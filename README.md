# StableHLO for Cryptographic Applications

A modified version of [XLA StableHLO](https://github.com/openxla/stablehlo)
extended for cryptographic applications, particularly zero-knowledge (ZK) proof
systems.

## Overview

This project extends StableHLO with support for cryptographic primitives by
integrating with:

- **Field Dialect**: Finite field arithmetic (e.g., BabyBear prime field)
- **EllipticCurve Dialect**: Elliptic curve operations

These extensions enable StableHLO operations to work with cryptographic types,
allowing tensor operations over finite fields commonly used in ZK proof systems.

## Example

```mlir
!pf_babybear = !field.pf<2013265921 : i32, true>

func.func @reduce_sum(%x: tensor<15x!pf_babybear>) -> tensor<!pf_babybear> {
  %init = stablehlo.constant dense<0> : tensor<!pf_babybear>
  %0 = stablehlo.reduce(%x init: %init) applies stablehlo.add across dimensions = [0]
       : (tensor<15x!pf_babybear>, tensor<!pf_babybear>) -> tensor<!pf_babybear>
  return %0 : tensor<!pf_babybear>
}
```

## Build

### Prerequisites

- Bazel 7.x

### Build and Test

```bash
# Build all targets
bazel build //...

# Run tests
bazel test //...
```

### Generate compile_commands.json (for IDE support)

```bash
bazel run @hedron_compile_commands//:refresh_all
```

## Dependencies

- [LLVM/MLIR](https://github.com/llvm/llvm-project)
- [prime_ir](https://github.com/fractalyze/prime_ir): Field and EllipticCurve
  dialects
- [zk_dtypes](https://github.com/fractalyze/zk_dtypes): ZK data types

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
