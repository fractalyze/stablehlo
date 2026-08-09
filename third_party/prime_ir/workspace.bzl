"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).

Override with a local checkout via `--override_repository=prime_ir=<path>`.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "29f9f08871c3c347b03a4c167c63ff0abbb6c64e"
PRIME_IR_SHA256 = "20caaaa8bdcfd817dddf260d2a5a71ce28ac9cfb0f8d95a1b40097b62e853f62"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
