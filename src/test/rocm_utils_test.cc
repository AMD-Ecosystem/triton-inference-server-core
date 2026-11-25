// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Advanced Micro Devices, Inc. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Tests for the ROCm device helpers and the ROCm paths of the memory
// managers that the upstream memory tests do not reach on one GPU.

#include <hip/hip_runtime_api.h>

#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "buffer_attributes.h"
#include "gtest/gtest.h"
#include "model_config_rocm.h"
#include "numa_utils.h"
#include "pinned_memory_manager.h"
#include "rocm_block_manager.h"
#include "rocm_memory_manager.h"
#include "rocm_utils.h"
#include "triton/common/sync_queue.h"
#include "triton_memory.h"

namespace tc = triton::core;

namespace {

// No device has this ID, so every HIP call made for it fails.
constexpr int kBadDevice = 1 << 20;
constexpr double kMinCc = TRITON_MIN_COMPUTE_CAPABILITY;

class RocmUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(hipSetDevice(0), hipSuccess); }
};

TEST_F(RocmUtilsTest, DeviceMemoryInfo)
{
  size_t free = 0, total = 0;
  ASSERT_TRUE(tc::GetDeviceMemoryInfo(0, &free, &total).IsOk());
  EXPECT_GT(total, 0u);
  EXPECT_LE(free, total);

  auto status = tc::GetDeviceMemoryInfo(kBadDevice, &free, &total);
  EXPECT_FALSE(status.IsOk());
  EXPECT_NE(status.Message().find("unable to get memory info"), std::string::npos)
      << status.Message();
  int device = -1;
  ASSERT_EQ(hipGetDevice(&device), hipSuccess);
  EXPECT_EQ(device, 0) << "the caller's device must be restored";
}

TEST_F(RocmUtilsTest, DeviceProperties)
{
  EXPECT_TRUE(tc::CheckGPUCompatibility(0, kMinCc).IsOk());
  EXPECT_FALSE(tc::CheckGPUCompatibility(0, 1000.0).IsOk());
  auto status = tc::CheckGPUCompatibility(kBadDevice, kMinCc);
  EXPECT_FALSE(status.IsOk());
  EXPECT_NE(
      status.Message().find("unable to get ROCM device properties"),
      std::string::npos)
      << status.Message();

  bool zero_copy = true;
  ASSERT_TRUE(tc::SupportsIntegratedZeroCopy(0, &zero_copy).IsOk());
  EXPECT_FALSE(tc::SupportsIntegratedZeroCopy(kBadDevice, &zero_copy).IsOk());

  std::set<int> gpus;
  ASSERT_TRUE(tc::GetSupportedGPUs(&gpus, kMinCc).IsOk());
  EXPECT_EQ(gpus.count(0), 1u);
}

TEST_F(RocmUtilsTest, ScopedSetDevice)
{
  {
    tc::ScopedSetDevice same(0);
    int device = -1;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    EXPECT_EQ(device, 0);
  }
  {
    // Switching to a missing device fails inside HIP; the guard must still
    // put device 0 back.
    tc::ScopedSetDevice other(kBadDevice);
  }
  int device = -1;
  ASSERT_EQ(hipGetDevice(&device), hipSuccess);
  EXPECT_EQ(device, 0);
}

TEST_F(RocmUtilsTest, CopyBuffer)
{
  constexpr size_t kCount = 1024;
  constexpr size_t kBytes = kCount * sizeof(int);
  std::vector<int> src(kCount), dst(kCount, 0), back(kCount, 0);
  std::iota(src.begin(), src.end(), 0);
  hipStream_t stream;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  void* dev = nullptr;
  ASSERT_EQ(hipMalloc(&dev, kBytes), hipSuccess);

  bool used = true;
  ASSERT_TRUE(tc::CopyBuffer(
                  "cpu to cpu", TRITONSERVER_MEMORY_CPU, 0,
                  TRITONSERVER_MEMORY_CPU, 0, kBytes, src.data(), dst.data(),
                  stream, &used)
                  .IsOk());
  EXPECT_FALSE(used);
  EXPECT_EQ(src, dst);

  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_TRUE(tc::CopyBuffer(
                  "cpu to cpu on stream", TRITONSERVER_MEMORY_CPU, 0,
                  TRITONSERVER_MEMORY_CPU_PINNED, 0, kBytes, src.data(),
                  dst.data(), stream, &used, true /* copy_on_stream */)
                  .IsOk());
  EXPECT_TRUE(used);
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
  EXPECT_EQ(src, dst);

  ASSERT_TRUE(tc::CopyBuffer(
                  "cpu to gpu", TRITONSERVER_MEMORY_CPU, 0,
                  TRITONSERVER_MEMORY_GPU, 0, kBytes, src.data(), dev, stream,
                  &used)
                  .IsOk());
  EXPECT_TRUE(used);
  ASSERT_TRUE(tc::CopyBuffer(
                  "gpu to cpu", TRITONSERVER_MEMORY_GPU, 0,
                  TRITONSERVER_MEMORY_CPU, 0, kBytes, dev, back.data(), stream,
                  &used)
                  .IsOk());
  EXPECT_TRUE(used);
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
  EXPECT_EQ(src, back);

  EXPECT_EQ(hipFree(dev), hipSuccess);
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(RocmUtilsTest, CopyBufferHandler)
{
  std::vector<char> src(64, 'x'), dst(64, 0);
  triton::common::SyncQueue<std::tuple<tc::Status, bool, void*>> queue;
  int marker = 0;
  tc::CopyBufferHandler(
      "handler", TRITONSERVER_MEMORY_CPU, 0, TRITONSERVER_MEMORY_CPU, 0,
      src.size(), src.data(), dst.data(), nullptr, &marker, &queue);
  auto result = queue.Get();
  EXPECT_TRUE(std::get<0>(result).IsOk());
  EXPECT_FALSE(std::get<1>(result));
  EXPECT_EQ(std::get<2>(result), &marker);
  EXPECT_EQ(src, dst);
}

TEST_F(RocmUtilsTest, DriverHelper)
{
  auto& helper = tc::CudaDriverHelper::GetInstance();
  ASSERT_TRUE(helper.IsAvailable());
  const char* message = nullptr;
  helper.CuGetErrorString(&message, hipErrorInvalidValue);
  ASSERT_NE(message, nullptr);
  EXPECT_GT(std::strlen(message), 0u);

  size_t granularity = 0;
  ASSERT_TRUE(tc::GetAllocationGranularity(granularity).IsOk());
  EXPECT_GT(granularity, 0u);
}

TEST_F(RocmUtilsTest, StreamPriority)
{
  int least = 0, greatest = 0;
  ASSERT_EQ(hipDeviceGetStreamPriorityRange(&least, &greatest), hipSuccess);
  EXPECT_EQ(
      tc::GetRocmStreamPriority(
          inference::ModelOptimizationPolicy::PRIORITY_MAX),
      greatest);
  EXPECT_EQ(
      tc::GetRocmStreamPriority(
          inference::ModelOptimizationPolicy::PRIORITY_MIN),
      least);
  EXPECT_EQ(
      tc::GetRocmStreamPriority(
          inference::ModelOptimizationPolicy::PRIORITY_DEFAULT),
      0);
}

TEST(RocmBufferAttributesTest, IpcHandle)
{
  char handle[64];
  for (int i = 0; i < 64; ++i) {
    handle[i] = static_cast<char>(i + 1);
  }
  tc::BufferAttributes attrs(256, TRITONSERVER_MEMORY_GPU, 0, handle);
  ASSERT_NE(attrs.CudaIpcHandle(), nullptr);
  EXPECT_EQ(std::memcmp(attrs.CudaIpcHandle(), handle, 64), 0);

  tc::BufferAttributes empty;
  EXPECT_EQ(empty.CudaIpcHandle(), nullptr);
  empty.SetCudaIpcHandle(handle);
  ASSERT_NE(empty.CudaIpcHandle(), nullptr);
  EXPECT_EQ(std::memcmp(empty.CudaIpcHandle(), handle, 64), 0);
}

class RocmMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_EQ(hipSetDevice(0), hipSuccess); }
  void TearDown() override
  {
    tc::CudaMemoryManager::Reset();
    tc::PinnedMemoryManager::Reset();
  }
};

TEST_F(RocmMemoryManagerTest, CreateTwiceKeepsFirstPool)
{
  tc::CudaMemoryManager::Options options(kMinCc, {{0, 1 << 20}});
  ASSERT_TRUE(tc::CudaMemoryManager::Create(options).IsOk());
  tc::CudaMemoryManager::Options bigger(kMinCc, {{0, 4 << 20}});
  EXPECT_TRUE(tc::CudaMemoryManager::Create(bigger).IsOk());

  void* ptr = nullptr;
  EXPECT_FALSE(tc::CudaMemoryManager::Alloc(&ptr, 2 << 20, 0).IsOk())
      << "the second Create must not replace the 1 MiB pool";
}

TEST_F(RocmMemoryManagerTest, NoPreallocatedPool)
{
  tc::CudaMemoryManager::Options options(kMinCc, {{0, 0}});
  ASSERT_TRUE(tc::CudaMemoryManager::Create(options).IsOk());
  void* ptr = nullptr;
  auto status = tc::CudaMemoryManager::Alloc(&ptr, 64, 0);
  EXPECT_FALSE(status.IsOk());
  EXPECT_NE(status.Message().find("no preallocated"), std::string::npos)
      << status.Message();
  EXPECT_FALSE(tc::CudaMemoryManager::Free(ptr, 0).IsOk());
}

TEST_F(RocmMemoryManagerTest, MissingDevice)
{
  tc::CudaMemoryManager::Options options(kMinCc, {{0, 1 << 20}});
  ASSERT_TRUE(tc::CudaMemoryManager::Create(options).IsOk());
  void* ptr = nullptr;
  EXPECT_FALSE(tc::CudaMemoryManager::Alloc(&ptr, 64, kBadDevice).IsOk());
  ASSERT_TRUE(tc::CudaMemoryManager::Alloc(&ptr, 64, 0).IsOk());
  EXPECT_FALSE(tc::CudaMemoryManager::Free(ptr, kBadDevice).IsOk());
  EXPECT_TRUE(tc::CudaMemoryManager::Free(ptr, 0).IsOk());
}

TEST_F(RocmMemoryManagerTest, SetGpuMemory)
{
  tc::CudaMemoryManager::Options options(kMinCc, {{0, 1 << 20}});
  ASSERT_TRUE(tc::CudaMemoryManager::Create(options).IsOk());
  tc::PinnedMemoryManager::Options pinned(1 << 20);
  ASSERT_TRUE(tc::PinnedMemoryManager::Create(pinned).IsOk());

  constexpr size_t kBytes = 4096;
  tc::AllocatedMemory memory(kBytes, TRITONSERVER_MEMORY_GPU, 0);
  TRITONSERVER_MemoryType type;
  int64_t type_id;
  char* buffer = memory.MutableBuffer(&type, &type_id);
  ASSERT_NE(buffer, nullptr);
  ASSERT_EQ(type, TRITONSERVER_MEMORY_GPU);
  ASSERT_TRUE(memory.SetMemory(0x5a).IsOk());

  std::vector<unsigned char> host(kBytes, 0);
  ASSERT_EQ(
      hipMemcpy(host.data(), buffer, kBytes, hipMemcpyDeviceToHost),
      hipSuccess);
  for (auto b : host) {
    ASSERT_EQ(b, 0x5a);
  }
}

// Setting a NUMA memory policy needs CAP_SYS_NICE inside a container.
TEST_F(RocmMemoryManagerTest, PinnedPoolOnNumaNode)
{
  triton::common::HostPolicyCmdlineConfigMap policy{
      {"gpu_0", {{"numa-node", "0"}}}};
  std::map<std::string, std::string> numa{{"numa-node", "0"}};
  if (!tc::SetNumaMemoryPolicy(numa).IsOk()) {
    GTEST_SKIP() << "cannot set a NUMA memory policy here";
  }
  ASSERT_TRUE(tc::ResetNumaMemoryPolicy().IsOk());
  tc::PinnedMemoryManager::Options options(1 << 20, policy);
  ASSERT_TRUE(tc::PinnedMemoryManager::Create(options).IsOk());
  void* ptr = nullptr;
  TRITONSERVER_MemoryType type = TRITONSERVER_MEMORY_CPU;
  ASSERT_TRUE(tc::PinnedMemoryManager::Alloc(
                  &ptr, 64, &type, false /* allow_nonpinned_fallback */)
                  .IsOk());
  EXPECT_EQ(type, TRITONSERVER_MEMORY_CPU_PINNED);
  EXPECT_TRUE(tc::PinnedMemoryManager::Free(ptr).IsOk());
  tc::PinnedMemoryManager::Reset();

  // A pinned pool larger than any host makes hipHostMalloc fail; the
  // manager is still created, without a pool.
  tc::PinnedMemoryManager::Options huge(1ULL << 50, policy);
  EXPECT_TRUE(tc::PinnedMemoryManager::Create(huge).IsOk());
}

class RocmBlockManagerTest : public ::testing::Test {
 protected:
  void TearDown() override { tc::CudaBlockManager::Reset(); }
};

TEST_F(RocmBlockManagerTest, UseBeforeCreate)
{
  std::unique_ptr<tc::Allocation> allocation;
  auto status = tc::CudaBlockManager::Allocate(1 << 20, allocation, 0);
  EXPECT_FALSE(status.IsOk());
  EXPECT_NE(status.Message().find("has not been created"), std::string::npos)
      << status.Message();
  tc::Allocation orphan(0);
  EXPECT_FALSE(tc::CudaBlockManager::Free(&orphan, 0).IsOk());
}

TEST_F(RocmBlockManagerTest, CreateTwice)
{
  ASSERT_TRUE(tc::CudaBlockManager::Create(kMinCc).IsOk());
  auto status = tc::CudaBlockManager::Create(kMinCc);
  EXPECT_FALSE(status.IsOk());
  EXPECT_NE(status.Message().find("already been created"), std::string::npos)
      << status.Message();
}

}  // namespace
