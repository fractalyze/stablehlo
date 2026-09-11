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

"""Provides the repo macro to import the Chromium sysroot the clang toolchain uses.

Both dependency lanes go through `repo()`: `WORKSPACE.bazel` calls it directly
and `//bazel:toolchain_deps.bzl` wraps it in a module extension, so the archive
and its digest have one home.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# The chromium sysroot URL is sha256-keyed with no file extension, so the digest
# doubles as the archive name.
SYSROOT_SHA256 = "52d61d4446ffebfaa3dda2cd02da4ab4876ff237853f46d273e7f9b666652e1d"

_BUILD_FILE_CONTENT = """filegroup(
    name = "sysroot",
    srcs = glob(
        ["**"],
        # systemd unit filenames contain literal backslash escapes, which
        # are invalid in bazel labels — and nothing in a compile or link
        # ever reads them.
        exclude = ["lib/systemd/**"],
    ),
    visibility = ["//visibility:public"],
)
"""

def repo():
    """Declares the debian bullseye amd64 sysroot the hermetic clang compiles against.

    Pinning it keeps worker glibc/libtinfo out of action outputs — the LLVM 20.x
    generic tarball otherwise resolves the non-hermetic host /usr. The filegroup
    name and extraction layout match toolchains_llvm's `sysroot` repo rule
    convention, reproduced via `http_archive` to avoid its aspect_bazel_lib /
    bsdtar dep chain — but with `glob(["**"])` where the convention uses srcs
    `["."]`: the latter feeds the repo ROOT into every compile action as a single
    directory artifact, which bazel flags ("dependency checking of directories is
    unsound") once per consuming target and cannot hash incrementally. Per-file
    artifacts are tracked soundly and keep logs quiet.
    """
    http_archive(
        name = "org_chromium_sysroot_linux_x64",
        build_file_content = _BUILD_FILE_CONTENT,
        sha256 = SYSROOT_SHA256,
        # Bazel cannot infer the archive format from the extensionless URL.
        type = "tar.xz",
        urls = ["https://commondatastorage.googleapis.com/chrome-linux-sysroot/" + SYSROOT_SHA256],
    )
