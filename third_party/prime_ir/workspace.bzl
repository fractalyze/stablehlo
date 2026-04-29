"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "1c493a91025479fe7d714381967e37dd99dfffa2"
PRIME_IR_SHA256 = "b157f9c164439b9182c064cfbbb5865e4176ebaed68fb24acbf024b6447ea23f"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
