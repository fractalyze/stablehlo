// RUN: stablehlo-opt --prime-ir-to-stablehlo --split-input-file %s | FileCheck %s

// The reverse leg of the stablehlo↔prime-ir round trip: prime-ir field ops
// that survive canonicalization must be rebuilt as stablehlo ops or they hit
// the HLO export gate ("unsupported op for export to XLA"). field.inverse is
// the interesting case — it has no stablehlo counterpart and is rebuilt as
// divide(1, x), so the "1" constant must be materialized per field kind.

// CHECK-LABEL: func @inverse_prime_field
func.func @inverse_prime_field(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK: %[[ONE:.*]] = stablehlo.constant() <{value = dense<1> : tensor<4xi32>}>
  // CHECK: stablehlo.divide %[[ONE]], %arg0
  // CHECK-NOT: field.inverse
  %0 = field.inverse %a : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----
// Binary fields: the multiplicative identity is storage bit-pattern 1 in both
// the tower and the flat GHASH basis, so the constant is typed directly on
// the binary field (no base-prime-field resolution).

// CHECK-LABEL: func @inverse_binary_field_tower
func.func @inverse_binary_field_tower(%a: tensor<4x!field.bf<7>>)
    -> tensor<4x!field.bf<7>> {
  // CHECK: %[[ONE:.*]] = stablehlo.constant() <{value = dense<1> : tensor<4xi128>}> : () -> tensor<4x!field.bf<7>>
  // CHECK: stablehlo.divide %[[ONE]], %arg0 : tensor<4x!field.bf<7>>
  // CHECK-NOT: field.inverse
  %0 = field.inverse %a : tensor<4x!field.bf<7>>
  func.return %0 : tensor<4x!field.bf<7>>
}

// -----
// CHECK-LABEL: func @inverse_binary_field_ghash
func.func @inverse_binary_field_ghash(%a: tensor<3x!field.bf<7, ghash>>)
    -> tensor<3x!field.bf<7, ghash>> {
  // CHECK: %[[ONE:.*]] = stablehlo.constant() <{value = dense<1> : tensor<3xi128>}> : () -> tensor<3x!field.bf<7, ghash>>
  // CHECK: stablehlo.divide %[[ONE]], %arg0 : tensor<3x!field.bf<7, ghash>>
  // CHECK-NOT: field.inverse
  %0 = field.inverse %a : tensor<3x!field.bf<7, ghash>>
  func.return %0 : tensor<3x!field.bf<7, ghash>>
}

// -----
// Extension fields resolve the base prime field for the "1" constant; DivOp
// accepts the mixed PF/EF operand pairing.

// CHECK-LABEL: func @inverse_extension_field
func.func @inverse_extension_field(
    %a: tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>)
    -> tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>> {
  // CHECK: %[[ONE:.*]] = stablehlo.constant() <{value = dense<1> : tensor<2xi32>}>
  // CHECK: stablehlo.divide %[[ONE]], %arg0
  // CHECK-NOT: field.inverse
  %0 = field.inverse %a : tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>
  func.return %0 : tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>
}

// -----
// stablehlo is tensor-only: a bare scalar field.inverse has no stablehlo
// form and must be left for the prime-ir lowerings.

// CHECK-LABEL: func @inverse_scalar_stays
func.func @inverse_scalar_stays(%a: !field.bf<7, ghash>) -> !field.bf<7, ghash> {
  // CHECK: field.inverse
  // CHECK-NOT: stablehlo.divide
  %0 = field.inverse %a : !field.bf<7, ghash>
  func.return %0 : !field.bf<7, ghash>
}
