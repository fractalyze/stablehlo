# CLAUDE.md

## Project Overview
Fork of XLA StableHLO extended for cryptographic applications. Adds Field Dialect
(finite field arithmetic) and EllipticCurve Dialect for zero-knowledge proof systems.
Entry point of the ZKX compiler pipeline.

## Current Focus
Q2: E2E proving p99 ≤ 7s on 16 GPUs (excluding verification).
Sprint: E2E correctness — 5 test blocks Phase 1-3 bug-free.
Out of scope: Multi-zkVM 2nd backend, community building, internal tooling, external talks.

## Commands
- Build: `bazel build //...`
- Test: `bazel test //...`
- IDE support: `bazel run @hedron_compile_commands//:refresh_all`

## Why Decisions
- Fork over wrapper: deep integration of Field/EC dialects into tensor type system required modifying StableHLO core.
- `arith.*` preferred over `field.*`: integer ops are significantly cheaper on both CPU and GPU. Field ops only for genuine modular arithmetic.

## Rules
- Do NOT modify lowering passes affecting field arithmetic correctness without expert review.
- Do NOT use `field.*` ops when `arith.*` equivalents work — field ops are significantly more expensive.
- Follow upstream StableHLO code style and conventions.
- Always run `bazel test //...` before committing.

## Invisible Traps
- Field types use `!field.pf<prime : storage_type, montgomery>` — the `montgomery` flag changes arithmetic behavior silently.
- StableHLO tensor ops work with cryptographic types (e.g., `tensor<15x!pf_babybear>`). Type verifiers must handle these or they pass validation but produce wrong results.
- Pipeline position: **StableHLO** → ZKIR → PrimeIR → LLVM/GPU. Breaking changes here cascade to all downstream stages.

## Knowledge Files
Read ONLY when relevant to your current task:
@.claude/knowledge/architecture.md — Dialect extensions, type system
@.claude/knowledge/testing-guide.md — LIT test conventions
@.claude/knowledge/solutions.md — Past bug resolution patterns
