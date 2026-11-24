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
#include "rocm_cnmem.h"

#include <hip/hip_runtime_api.h>

#include <vector>

#include "gtest/gtest.h"

namespace tc = triton::core;

namespace {

class RocmCnmemTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  void TearDown() override { tc::cnmemFinalize(); }

  // Reserve 'size' bytes on device 0.
  tc::cnmemStatus_t Init(size_t size)
  {
    tc::cnmemDevice_t device{};
    device.device = 0;
    device.size = size;
    return tc::cnmemInit(1, &device, tc::CNMEM_FLAGS_CANNOT_GROW);
  }

  static const size_t G = tc::kCnmemGranularity;
};

TEST_F(RocmCnmemTest, InitRejectsBadArguments)
{
  tc::cnmemDevice_t device{};
  device.device = 0;
  device.size = G;
  EXPECT_EQ(
      tc::cnmemInit(0, &device, tc::CNMEM_FLAGS_CANNOT_GROW),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(
      tc::cnmemInit(1, nullptr, tc::CNMEM_FLAGS_CANNOT_GROW),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  // Growable pools are not supported.
  EXPECT_EQ(
      tc::cnmemInit(1, &device, tc::CNMEM_FLAGS_DEFAULT),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  // Zero-byte and duplicate devices are rejected.
  device.size = 0;
  EXPECT_EQ(
      tc::cnmemInit(1, &device, tc::CNMEM_FLAGS_CANNOT_GROW),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  tc::cnmemDevice_t twice[2] = {};
  twice[0].size = twice[1].size = G;
  EXPECT_EQ(
      tc::cnmemInit(2, twice, tc::CNMEM_FLAGS_CANNOT_GROW),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  // A device that does not exist cannot be reserved.
  device.device = 1 << 20;
  device.size = G;
  EXPECT_EQ(
      tc::cnmemInit(1, &device, tc::CNMEM_FLAGS_CANNOT_GROW),
      tc::CNMEM_STATUS_OUT_OF_MEMORY);
  // Nothing was left initialized by the failures.
  EXPECT_EQ(tc::cnmemFinalize(), tc::CNMEM_STATUS_NOT_INITIALIZED);
}

TEST_F(RocmCnmemTest, InitTwiceFails)
{
  ASSERT_EQ(Init(G), tc::CNMEM_STATUS_SUCCESS);
  EXPECT_EQ(Init(G), tc::CNMEM_STATUS_INVALID_ARGUMENT);
}

TEST_F(RocmCnmemTest, NotInitialized)
{
  void* ptr = nullptr;
  EXPECT_EQ(
      tc::cnmemMalloc(&ptr, G, nullptr), tc::CNMEM_STATUS_NOT_INITIALIZED);
  EXPECT_EQ(
      tc::cnmemFree(reinterpret_cast<void*>(G), nullptr),
      tc::CNMEM_STATUS_NOT_INITIALIZED);
  EXPECT_EQ(tc::cnmemFinalize(), tc::CNMEM_STATUS_NOT_INITIALIZED);
}

TEST_F(RocmCnmemTest, MallocArguments)
{
  ASSERT_EQ(Init(G), tc::CNMEM_STATUS_SUCCESS);
  EXPECT_EQ(
      tc::cnmemMalloc(nullptr, G, nullptr), tc::CNMEM_STATUS_INVALID_ARGUMENT);
  void* ptr = &ptr;
  EXPECT_EQ(tc::cnmemMalloc(&ptr, 0, nullptr), tc::CNMEM_STATUS_SUCCESS);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(tc::cnmemFree(nullptr, nullptr), tc::CNMEM_STATUS_SUCCESS);
}

TEST_F(RocmCnmemTest, AllocationsAreDeviceMemoryAndRounded)
{
  ASSERT_EQ(Init(4 * G), tc::CNMEM_STATUS_SUCCESS);
  void* a = nullptr;
  void* b = nullptr;
  ASSERT_EQ(tc::cnmemMalloc(&a, 1, nullptr), tc::CNMEM_STATUS_SUCCESS);
  ASSERT_EQ(tc::cnmemMalloc(&b, G + 1, nullptr), tc::CNMEM_STATUS_SUCCESS);
  // 1 byte takes one block and G + 1 bytes take two.
  EXPECT_EQ(static_cast<char*>(b) - static_cast<char*>(a), ptrdiff_t(G));
  void* c = nullptr;
  EXPECT_EQ(tc::cnmemMalloc(&c, 2 * G, nullptr), tc::CNMEM_STATUS_OUT_OF_MEMORY);
  ASSERT_EQ(tc::cnmemMalloc(&c, G, nullptr), tc::CNMEM_STATUS_SUCCESS);

  hipPointerAttribute_t attr;
  ASSERT_EQ(hipPointerGetAttributes(&attr, b), hipSuccess);
  EXPECT_EQ(attr.type, hipMemoryTypeDevice);
  EXPECT_EQ(hipMemset(b, 0x5a, G + 1), hipSuccess);
}

TEST_F(RocmCnmemTest, FreeRejectsForeignPointers)
{
  ASSERT_EQ(Init(2 * G), tc::CNMEM_STATUS_SUCCESS);
  void* a = nullptr;
  ASSERT_EQ(tc::cnmemMalloc(&a, 2 * G, nullptr), tc::CNMEM_STATUS_SUCCESS);
  char outside;
  EXPECT_EQ(tc::cnmemFree(&outside, nullptr), tc::CNMEM_STATUS_INVALID_ARGUMENT);
  // Inside the pool but not the start of an allocation.
  EXPECT_EQ(
      tc::cnmemFree(static_cast<char*>(a) + G, nullptr),
      tc::CNMEM_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(tc::cnmemFree(a, nullptr), tc::CNMEM_STATUS_SUCCESS);
  // A second free of the same pointer fails.
  EXPECT_EQ(tc::cnmemFree(a, nullptr), tc::CNMEM_STATUS_INVALID_ARGUMENT);
}

TEST_F(RocmCnmemTest, WrongDevice)
{
  int count = 0;
  ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
  ASSERT_EQ(Init(G), tc::CNMEM_STATUS_SUCCESS);
  if (count < 2) {
    GTEST_SKIP() << "needs a second device without a pool";
  }
  ASSERT_EQ(hipSetDevice(1), hipSuccess);
  void* ptr = nullptr;
  EXPECT_EQ(
      tc::cnmemMalloc(&ptr, G, nullptr), tc::CNMEM_STATUS_INVALID_ARGUMENT);
}

TEST_F(RocmCnmemTest, FreedBlocksMerge)
{
  ASSERT_EQ(Init(4 * G), tc::CNMEM_STATUS_SUCCESS);
  std::vector<void*> p(4, nullptr);
  for (auto& ptr : p) {
    ASSERT_EQ(tc::cnmemMalloc(&ptr, G, nullptr), tc::CNMEM_STATUS_SUCCESS);
  }
  void* big = nullptr;
  EXPECT_EQ(tc::cnmemMalloc(&big, G, nullptr), tc::CNMEM_STATUS_OUT_OF_MEMORY);

  // Free 1 and 2 (merging forward), then 0 (merging with the block after it)
  // and 3 (merging with the block before it): the pool is whole again.
  ASSERT_EQ(tc::cnmemFree(p[1], nullptr), tc::CNMEM_STATUS_SUCCESS);
  ASSERT_EQ(tc::cnmemFree(p[2], nullptr), tc::CNMEM_STATUS_SUCCESS);
  EXPECT_EQ(
      tc::cnmemMalloc(&big, 3 * G, nullptr), tc::CNMEM_STATUS_OUT_OF_MEMORY);
  ASSERT_EQ(tc::cnmemFree(p[0], nullptr), tc::CNMEM_STATUS_SUCCESS);
  ASSERT_EQ(tc::cnmemFree(p[3], nullptr), tc::CNMEM_STATUS_SUCCESS);
  ASSERT_EQ(tc::cnmemMalloc(&big, 4 * G, nullptr), tc::CNMEM_STATUS_SUCCESS);
  EXPECT_EQ(big, p[0]);
}

TEST_F(RocmCnmemTest, ErrorStrings)
{
  EXPECT_STREQ(tc::cnmemGetErrorString(tc::CNMEM_STATUS_SUCCESS), "success");
  EXPECT_STREQ(tc::cnmemGetErrorString(tc::CNMEM_STATUS_HIP_ERROR), "HIP error");
  EXPECT_STREQ(
      tc::cnmemGetErrorString(tc::CNMEM_STATUS_INVALID_ARGUMENT),
      "invalid argument");
  EXPECT_STREQ(
      tc::cnmemGetErrorString(tc::CNMEM_STATUS_NOT_INITIALIZED),
      "memory pool not initialized");
  EXPECT_STREQ(
      tc::cnmemGetErrorString(tc::CNMEM_STATUS_OUT_OF_MEMORY), "out of memory");
  EXPECT_STREQ(
      tc::cnmemGetErrorString(static_cast<tc::cnmemStatus_t>(-1)),
      "unknown error");
}

}  // namespace
