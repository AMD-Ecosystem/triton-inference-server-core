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
// GPU memory options and IPC handles through the public server API of a
// ROCm build.

#include <stdlib.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "triton/core/tritonserver.h"

namespace {

#define ASSERT_NO_ERR(X)                                     \
  do {                                                       \
    TRITONSERVER_Error* err__ = (X);                         \
    ASSERT_EQ(err__, nullptr) << TRITONSERVER_ErrorMessage(err__); \
  } while (false)

TEST(RocmServerApiTest, GpuMemoryOptions)
{
  char repo[] = "/tmp/rocm_server_api_test.XXXXXX";
  ASSERT_NE(mkdtemp(repo), nullptr);

  TRITONSERVER_ServerOptions* options = nullptr;
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsNew(&options));
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(options, repo));
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(
      options, TRITONSERVER_MODEL_CONTROL_EXPLICIT));
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(
      options, "/opt/tritonserver/backends"));
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
      options, "/opt/tritonserver/repoagents"));
  ASSERT_NO_ERR(
      TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(options, 0, 8 << 20));
  ASSERT_NO_ERR(
      TRITONSERVER_ServerOptionsSetCudaVirtualAddressSize(options, 0, 1 << 30));

  TRITONSERVER_Server* server = nullptr;
  ASSERT_NO_ERR(TRITONSERVER_ServerNew(&server, options));
  ASSERT_NO_ERR(TRITONSERVER_ServerOptionsDelete(options));
  bool live = false;
  ASSERT_NO_ERR(TRITONSERVER_ServerIsLive(server, &live));
  EXPECT_TRUE(live);
  ASSERT_NO_ERR(TRITONSERVER_ServerDelete(server));
  rmdir(repo);
}

TEST(RocmServerApiTest, BufferAttributesIpcHandle)
{
  char handle[64];
  for (int i = 0; i < 64; ++i) {
    handle[i] = static_cast<char>(64 - i);
  }
  TRITONSERVER_BufferAttributes* attrs = nullptr;
  ASSERT_NO_ERR(TRITONSERVER_BufferAttributesNew(&attrs));
  ASSERT_NO_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(attrs, handle));
  void* got = nullptr;
  ASSERT_NO_ERR(TRITONSERVER_BufferAttributesCudaIpcHandle(attrs, &got));
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(std::memcmp(got, handle, 64), 0);
  ASSERT_NO_ERR(TRITONSERVER_BufferAttributesDelete(attrs));
}

}  // namespace
