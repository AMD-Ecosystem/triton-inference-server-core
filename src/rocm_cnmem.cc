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
#ifdef TRITON_ENABLE_ROCM

#include "rocm_cnmem.h"

#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace triton { namespace core {

namespace {

size_t
RoundUp(size_t size)
{
  return (size + kCnmemGranularity - 1) / kCnmemGranularity * kCnmemGranularity;
}

class DevicePool {
 public:
  DevicePool(char* base, size_t size) : base_(base), size_(size)
  {
    free_[0] = size;
  }

  ~DevicePool() { static_cast<void>(hipFree(base_)); }

  char* Base() const { return base_; }

  cnmemStatus_t Alloc(void** ptr, size_t size)
  {
    size = RoundUp(size);
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second < size) {
        continue;
      }
      const size_t offset = it->first;
      const size_t remaining = it->second - size;
      free_.erase(it);
      if (remaining > 0) {
        free_[offset + size] = remaining;
      }
      used_[offset] = size;
      *ptr = base_ + offset;
      return CNMEM_STATUS_SUCCESS;
    }
    return CNMEM_STATUS_OUT_OF_MEMORY;
  }

  cnmemStatus_t Free(void* ptr)
  {
    char* p = static_cast<char*>(ptr);
    if ((p < base_) || (p >= base_ + size_)) {
      return CNMEM_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto used = used_.find(p - base_);
    if (used == used_.end()) {
      return CNMEM_STATUS_INVALID_ARGUMENT;
    }
    size_t offset = used->first;
    size_t size = used->second;
    used_.erase(used);

    // Merge with the free blocks on either side.
    auto next = free_.lower_bound(offset);
    if ((next != free_.end()) && (next->first == offset + size)) {
      size += next->second;
      next = free_.erase(next);
    }
    if (next != free_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == offset) {
        offset = prev->first;
        size += prev->second;
        free_.erase(prev);
      }
    }
    free_[offset] = size;
    return CNMEM_STATUS_SUCCESS;
  }

 private:
  char* const base_;
  const size_t size_;
  std::mutex mu_;
  // Offset -> size of the free blocks, kept merged.
  std::map<size_t, size_t> free_;
  // Offset -> size of the allocated blocks.
  std::unordered_map<size_t, size_t> used_;
};

std::mutex pools_mu;
std::map<int, std::unique_ptr<DevicePool>> pools;

cnmemStatus_t
CurrentPool(DevicePool** pool)
{
  int device;
  if (hipGetDevice(&device) != hipSuccess) {
    return CNMEM_STATUS_HIP_ERROR;
  }
  std::lock_guard<std::mutex> lock(pools_mu);
  if (pools.empty()) {
    return CNMEM_STATUS_NOT_INITIALIZED;
  }
  auto it = pools.find(device);
  if (it == pools.end()) {
    return CNMEM_STATUS_INVALID_ARGUMENT;
  }
  *pool = it->second.get();
  return CNMEM_STATUS_SUCCESS;
}

}  // namespace

cnmemStatus_t
cnmemInit(int num_devices, const cnmemDevice_t* devices, unsigned flags)
{
  if ((num_devices <= 0) || (devices == nullptr) ||
      ((flags & CNMEM_FLAGS_CANNOT_GROW) == 0)) {
    return CNMEM_STATUS_INVALID_ARGUMENT;
  }
  int current_device;
  if (hipGetDevice(&current_device) != hipSuccess) {
    return CNMEM_STATUS_HIP_ERROR;
  }
  std::map<int, std::unique_ptr<DevicePool>> created;
  cnmemStatus_t status = CNMEM_STATUS_SUCCESS;
  for (int i = 0; i < num_devices; ++i) {
    const size_t size = RoundUp(devices[i].size);
    void* base = nullptr;
    if ((size == 0) || (created.count(devices[i].device) != 0)) {
      status = CNMEM_STATUS_INVALID_ARGUMENT;
    } else if (
        (hipSetDevice(devices[i].device) != hipSuccess) ||
        (hipMalloc(&base, size) != hipSuccess)) {
      status = CNMEM_STATUS_OUT_OF_MEMORY;
    }
    if (status != CNMEM_STATUS_SUCCESS) {
      break;
    }
    created[devices[i].device].reset(
        new DevicePool(static_cast<char*>(base), size));
  }
  static_cast<void>(hipSetDevice(current_device));
  if (status != CNMEM_STATUS_SUCCESS) {
    return status;
  }

  std::lock_guard<std::mutex> lock(pools_mu);
  if (!pools.empty()) {
    return CNMEM_STATUS_INVALID_ARGUMENT;
  }
  pools = std::move(created);
  return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t
cnmemFinalize()
{
  std::lock_guard<std::mutex> lock(pools_mu);
  if (pools.empty()) {
    return CNMEM_STATUS_NOT_INITIALIZED;
  }
  pools.clear();
  return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t
cnmemMalloc(void** ptr, size_t size, hipStream_t /* stream */)
{
  if (ptr == nullptr) {
    return CNMEM_STATUS_INVALID_ARGUMENT;
  }
  if (size == 0) {
    *ptr = nullptr;
    return CNMEM_STATUS_SUCCESS;
  }
  DevicePool* pool = nullptr;
  auto status = CurrentPool(&pool);
  return (status == CNMEM_STATUS_SUCCESS) ? pool->Alloc(ptr, size) : status;
}

cnmemStatus_t
cnmemFree(void* ptr, hipStream_t /* stream */)
{
  if (ptr == nullptr) {
    return CNMEM_STATUS_SUCCESS;
  }
  DevicePool* pool = nullptr;
  auto status = CurrentPool(&pool);
  return (status == CNMEM_STATUS_SUCCESS) ? pool->Free(ptr) : status;
}

const char*
cnmemGetErrorString(cnmemStatus_t status)
{
  switch (status) {
    case CNMEM_STATUS_SUCCESS:
      return "success";
    case CNMEM_STATUS_HIP_ERROR:
      return "HIP error";
    case CNMEM_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case CNMEM_STATUS_NOT_INITIALIZED:
      return "memory pool not initialized";
    case CNMEM_STATUS_OUT_OF_MEMORY:
      return "out of memory";
  }
  return "unknown error";
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_ROCM
