# Copyright 2026 The StableHLO Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The bzlmod half of the hermetic toolchain's sysroot declaration.

`WORKSPACE.bazel` calls `//third_party/sysroot:workspace.bzl`'s `repo()`
directly; this extension calls the same function, so both lanes fetch one
archive pinned in one place. It is a dev dependency of MODULE.bazel, like the
toolchain it feeds: which compiler builds StableHLO is the root module's choice,
not part of what a consumer resolves.
"""

load("//third_party/sysroot:workspace.bzl", sysroot = "repo")

def _toolchain_deps_impl(_module_ctx):
    sysroot()

toolchain_deps = module_extension(implementation = _toolchain_deps_impl)
