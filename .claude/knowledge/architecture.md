# stablehlo Architecture

## Overview
Fork of XLA StableHLO extended with cryptographic type support.

## Dialect Extensions
- **Field Dialect** (`field.*`): Finite field arithmetic
  - Types: `!field.pf<prime : storage_type, montgomery>` (prime field)
  - Ops: `field.add`, `field.mul`, `field.sub`, `field.inverse`
- **EllipticCurve Dialect** (`ec.*`): EC operations
  - Types: `!ec.point<curve>`
  - Ops: EC point addition, scalar multiplication

## Key Design Decisions
- StableHLO tensor ops work with cryptographic types (e.g., `tensor<15x!pf_babybear>`)
- Field ops are more expensive than integer ops — prefer `arith.*` where possible
- Montgomery form types indicated by `true` flag in type definition

## Relationship to ZKX Pipeline
```
StableHLO (this repo) → ZKIR → PrimeIR → LLVM/GPU
```
