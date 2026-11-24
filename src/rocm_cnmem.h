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
#pragma once

#ifdef TRITON_ENABLE_ROCM

#include <hip/hip_runtime_api.h>

#include <cstddef>

// The subset of the cnmem API that CudaMemoryManager uses, implemented with
// HIP so the ROCm build keeps the same fixed-size per-device memory pools.
// Each pool is one hipMalloc of the configured size, carved into blocks of
// kCnmemGranularity bytes; an allocation that does not fit fails instead of
// growing the pool. Allocation and free act on the current HIP device.

namespace triton { namespace core {

constexpr size_t kCnmemGranularity = 512;

enum cnmemStatus_t {
  CNMEM_STATUS_SUCCESS = 0,
  CNMEM_STATUS_HIP_ERROR,
  CNMEM_STATUS_INVALID_ARGUMENT,
  CNMEM_STATUS_NOT_INITIALIZED,
  CNMEM_STATUS_OUT_OF_MEMORY,
};

enum cnmemManagerFlags_t {
  CNMEM_FLAGS_DEFAULT = 0,
  CNMEM_FLAGS_CANNOT_GROW = 1,
};

struct cnmemDevice_t {
  int device;
  size_t size;
  int numStreams;
  hipStream_t* streams;
  size_t* streamSizes;
};

// Reserve a pool on each of 'devices'. Only CNMEM_FLAGS_CANNOT_GROW pools are
// supported. Fails, leaving no pool behind, if any reservation fails.
cnmemStatus_t cnmemInit(int num_devices, const cnmemDevice_t* devices, unsigned flags);

// Release every pool. Outstanding allocations become invalid.
cnmemStatus_t cnmemFinalize();

// Allocate 'size' bytes from the current device's pool. 'stream' is ignored.
cnmemStatus_t cnmemMalloc(void** ptr, size_t size, hipStream_t stream);

// Return 'ptr' to the current device's pool. Fails if the pool did not
// allocate it.
cnmemStatus_t cnmemFree(void* ptr, hipStream_t stream);

const char* cnmemGetErrorString(cnmemStatus_t status);

}}  // namespace triton::core

#endif  // TRITON_ENABLE_ROCM
