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

"""Checks that every module in the repository pins the same forks.

Neither prime_ir nor zk_dtypes is in a registry, so the bzlmod lane reaches them
through `archive_override`s in MODULE.bazel while the WORKSPACE lane reaches
prime_ir through an `http_archive` in `workspace.bzl` and zk_dtypes through
prime_ir's own. MODULE.bazel cannot `load()`, so no pin can be single-sourced
and the copies can drift — leaving the lanes building different revisions of the
dependencies the rest of the chain hangs on.

Three files carry a copy. `//third_party/prime_ir:workspace.bzl` and
`//MODULE.bazel` are the two lanes. `//bazel/bzlmod_consumer/MODULE.bazel` is
the third and the easiest to forget: overrides apply only in the root module,
and that fixture is root in its own lane, so its stale pin would not be
corrected by either of the others — it would just build stablehlo against an old
prime_ir. All three spell the pin behind `PRIME_IR_`- and
`ZK_DTYPES_`-prefixed variables so that one substitution finds it in any of
them, which is what lets `.github/workflows/pin-bump.yml` hand every path to the
same bump action. This test is the other half of that arrangement.

The zk_dtypes pin has no WORKSPACE-lane copy in this repository: the authority
is whatever prime_ir itself declares, which is read here out of
`@prime_ir//:MODULE.bazel`. A prime_ir bump that also moves zk_dtypes therefore
fails this test until the two MODULE.bazel files follow.

A digest is the same hash written two ways — `http_archive` takes hex,
`archive_override` takes base64 — so comparing one means converting first.
"""

import base64
import binascii
import re

from absl import flags
from absl.testing import absltest

_MODULE_BAZEL = flags.DEFINE_string(
    "module_bazel", None, "Path to this repository's MODULE.bazel."
)
_WORKSPACE_BZL = flags.DEFINE_string(
    "workspace_bzl", None, "Path to //third_party/prime_ir:workspace.bzl."
)
_PRIME_IR_MODULE_BAZEL = flags.DEFINE_string(
    "prime_ir_module_bazel", None, "Path to @prime_ir//:MODULE.bazel."
)
_CONSUMER_MODULE_BAZEL = flags.DEFINE_string(
    "consumer_module_bazel",
    None,
    "Path to //bazel/bzlmod_consumer/MODULE.bazel.",
)


def _commit_re(prefix):
  return re.compile(rf'{prefix}_COMMIT = "([0-9a-f]{{40}})"')


def _sha256_re(prefix):
  return re.compile(rf'{prefix}_SHA256 = "([0-9a-f]{{64}})"')


def _integrity_re(prefix):
  return re.compile(rf'{prefix}_INTEGRITY = "sha256-([A-Za-z0-9+/=]+)"')


def _search(pattern, contents, path):
  match = pattern.search(contents)
  if not match:
    raise AssertionError(f"{path} has no {pattern.pattern}")
  return match.group(1)


def _hex_as_integrity(digest):
  return base64.b64encode(binascii.unhexlify(digest)).decode()


class PinSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module_bazel = self._read(_MODULE_BAZEL.value)
    self.workspace_bzl = self._read(_WORKSPACE_BZL.value)
    self.prime_ir_module_bazel = self._read(_PRIME_IR_MODULE_BAZEL.value)
    self.consumer_module_bazel = self._read(_CONSUMER_MODULE_BAZEL.value)

  def _read(self, path):
    with open(path, encoding="utf-8") as f:
      return f.read()

  def test_prime_ir_commits_match(self):
    self.assertEqual(
        _search(_commit_re("PRIME_IR"), self.workspace_bzl, "workspace.bzl"),
        _search(_commit_re("PRIME_IR"), self.module_bazel, "MODULE.bazel"),
    )

  def test_prime_ir_hashes_match(self):
    self.assertEqual(
        _hex_as_integrity(
            _search(_sha256_re("PRIME_IR"), self.workspace_bzl, "workspace.bzl")
        ),
        _search(_integrity_re("PRIME_IR"), self.module_bazel, "MODULE.bazel"),
    )

  def test_zk_dtypes_override_tracks_prime_ir(self):
    for pattern in (_commit_re("ZK_DTYPES"), _integrity_re("ZK_DTYPES")):
      self.assertEqual(
          _search(pattern, self.prime_ir_module_bazel, "@prime_ir//:MODULE.bazel"),
          _search(pattern, self.module_bazel, "MODULE.bazel"),
      )

  def test_consumer_fixture_restates_both_overrides(self):
    for prefix in ("PRIME_IR", "ZK_DTYPES"):
      for pattern in (_commit_re(prefix), _integrity_re(prefix)):
        self.assertEqual(
            _search(pattern, self.module_bazel, "MODULE.bazel"),
            _search(
                pattern,
                self.consumer_module_bazel,
                "bazel/bzlmod_consumer/MODULE.bazel",
            ),
        )


if __name__ == "__main__":
  absltest.main()
