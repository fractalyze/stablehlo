# stablehlo Testing Guide

## Running Tests
```
bazel test //...
bazel test //stablehlo/tests:all
```

## LIT Test Format
```mlir
// RUN: stablehlo-opt %s --pass-name | FileCheck %s
// CHECK-LABEL: func @test_name
```

## Test Conventions
- One test file per pass or feature
- Include both positive and negative test cases
- Test field type interactions with standard StableHLO ops
