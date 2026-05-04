# Copyright 2025 The Stablehlo(Fractalyze) Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    PRIME_IR_COMMIT = "22da8066960ef469944dbf006ca09ea0c5d8a965"
    PRIME_IR_SHA256 = "4e71cc05fbb6f1344ee62b094a2807ebf7283af4a5f0a4e2bcd2d24aee8d9078"
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-{commit}".format(commit = PRIME_IR_COMMIT),
        urls = ["https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "prime_ir",
    #     path = "../prime-ir",
    # )
