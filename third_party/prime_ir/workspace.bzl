"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).

Override with a local checkout via `--override_repository=prime_ir=<path>`.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "9ddaeb6163f4b830ec3bbdacf6482abf416d3a88"
PRIME_IR_SHA256 = "c83168325996b9d78919576173db4362900f58da74c6e6c9959a009ca875ebec"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
