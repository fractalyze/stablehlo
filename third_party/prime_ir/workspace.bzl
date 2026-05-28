"""Provides the repo macro to import prime_ir.

prime_ir provides MLIR dialects for cryptographic computations
(Field, ModArith, EllipticCurve, Poly, TensorExt, ArithExt).

Pinned to fractalyze/prime-ir branch fix/function-outliner-internal-linkage
(the LLVM 815edc3 base + outliner linkage fixes + shaped tensor EC work
the proto tree builds against). For local proto iteration the sibling
checkout can be wired via
`--override_repository=prime_ir=/home/baz/proto/prime_ir_fork`.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

PRIME_IR_COMMIT = "07123d5cdc2c53dd79639487253308c8f76ac589"
PRIME_IR_SHA256 = "b8e9ae50df470b70ec1606a3d673e62f00e9b387d15055d880228f98a490ca7c"

def repo():
    http_archive(
        name = "prime_ir",
        sha256 = PRIME_IR_SHA256,
        strip_prefix = "prime-ir-" + PRIME_IR_COMMIT,
        urls = [
            "https://github.com/fractalyze/prime-ir/archive/{commit}.tar.gz".format(commit = PRIME_IR_COMMIT),
        ],
    )
