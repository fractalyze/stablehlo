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
    PRIME_IR_COMMIT = "8950a652316a4bda1001f06780932f2e8b8a5c82"
    PRIME_IR_SHA256 = "f2821966ddaa83f2278daaae9157d1a69f43f721f918226a65fd55db0af996e9"
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
