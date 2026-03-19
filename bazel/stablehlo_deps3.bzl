# Copyright 2026 The StableHLO(Fractalyze) Authors.
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

"""StableHLO dependencies3."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("@prime_ir//bazel:prime_ir_deps.bzl", "prime_ir_deps")
load("@zk_dtypes//bazel:zk_dtypes_deps.bzl", "zk_dtypes_deps")

def stablehlo_deps3():
    """StableHLO dependencies3."""

    zk_dtypes_deps()
    prime_ir_deps()

    llvm_configure(name = "llvm-project")
