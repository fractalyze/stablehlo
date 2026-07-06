"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).

Override with a local checkout via `--override_repository=prime_ir=<path>`.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "a783a98f010c9ccafea38e4ccd445ba3062930c2"
PRIME_IR_SHA256 = "123991638997e1d31a780478073c5b8fb071b69141420214645b429f8d86211b"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
