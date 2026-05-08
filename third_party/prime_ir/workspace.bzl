"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "da33331616409d6db5f32e5d20fee6aaa41d1719"
PRIME_IR_SHA256 = "50e17798922801d5e5cd1f4a644eab379c81507b9605baf8a867d68bfa3e4c83"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
