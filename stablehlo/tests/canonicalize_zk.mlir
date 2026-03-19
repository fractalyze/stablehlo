// RUN: stablehlo-opt %s -stablehlo-canonicalize -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Prime field constant folding
//===----------------------------------------------------------------------===//

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_add_constants_pf
func.func @fold_add_constants_pf() -> tensor<!pf7> {
  // 3 + 5 = 8 mod 7 = 1
  %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<!pf7>
  %1 = "stablehlo.constant"() <{value = dense<5> : tensor<i32>}> : () -> tensor<!pf7>
  %2 = stablehlo.add %0, %1 : tensor<!pf7>
  // CHECK: dense<1>
  return %2 : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_sub_constants_pf
func.func @fold_sub_constants_pf() -> tensor<!pf7> {
  // 3 - 5 = -2 mod 7 = 5
  %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<!pf7>
  %1 = "stablehlo.constant"() <{value = dense<5> : tensor<i32>}> : () -> tensor<!pf7>
  %2 = stablehlo.subtract %0, %1 : tensor<!pf7>
  // CHECK: dense<5>
  return %2 : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_mul_constants_pf
func.func @fold_mul_constants_pf() -> tensor<!pf7> {
  // 3 * 5 = 15 mod 7 = 1
  %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<!pf7>
  %1 = "stablehlo.constant"() <{value = dense<5> : tensor<i32>}> : () -> tensor<!pf7>
  %2 = stablehlo.multiply %0, %1 : tensor<!pf7>
  // CHECK: dense<1>
  return %2 : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_div_constants_pf
func.func @fold_div_constants_pf() -> tensor<!pf7> {
  // 6 / 3 = 2
  %0 = "stablehlo.constant"() <{value = dense<6> : tensor<i32>}> : () -> tensor<!pf7>
  %1 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<!pf7>
  %2 = stablehlo.divide %0, %1 : tensor<!pf7>
  // CHECK: dense<2>
  return %2 : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_negate_constant_pf
func.func @fold_negate_constant_pf() -> tensor<!pf7> {
  // -3 mod 7 = 4
  %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<!pf7>
  %1 = stablehlo.negate %0 : tensor<!pf7>
  // CHECK: dense<4>
  return %1 : tensor<!pf7>
}

// -----

//===----------------------------------------------------------------------===//
// Identity folding
//===----------------------------------------------------------------------===//

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_add_zero_identity
func.func @fold_add_zero_identity(%arg0: tensor<!pf7>) -> tensor<!pf7> {
  %zero = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf7>
  %result = stablehlo.add %arg0, %zero : tensor<!pf7>
  // CHECK-NOT: stablehlo.add
  // CHECK: return %arg0
  return %result : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_sub_zero_identity
func.func @fold_sub_zero_identity(%arg0: tensor<!pf7>) -> tensor<!pf7> {
  %zero = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf7>
  %result = stablehlo.subtract %arg0, %zero : tensor<!pf7>
  // CHECK-NOT: stablehlo.subtract
  // CHECK: return %arg0
  return %result : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_mul_one_identity
func.func @fold_mul_one_identity(%arg0: tensor<!pf7>) -> tensor<!pf7> {
  %one = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<!pf7>
  %result = stablehlo.multiply %arg0, %one : tensor<!pf7>
  // CHECK-NOT: stablehlo.multiply
  // CHECK: return %arg0
  return %result : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_mul_zero_absorb
func.func @fold_mul_zero_absorb(%arg0: tensor<!pf7>) -> tensor<!pf7> {
  %zero = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf7>
  %result = stablehlo.multiply %arg0, %zero : tensor<!pf7>
  // CHECK-NOT: stablehlo.multiply
  // CHECK: dense<0>
  return %result : tensor<!pf7>
}

// -----

!pf7 = !field.pf<7:i32>

// CHECK-LABEL: @fold_div_one_identity
func.func @fold_div_one_identity(%arg0: tensor<!pf7>) -> tensor<!pf7> {
  %one = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<!pf7>
  %result = stablehlo.divide %arg0, %one : tensor<!pf7>
  // CHECK-NOT: stablehlo.divide
  // CHECK: return %arg0
  return %result : tensor<!pf7>
}

// -----

//===----------------------------------------------------------------------===//
// Extension field constant folding
//===----------------------------------------------------------------------===//

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_add_constants_ef
func.func @fold_add_constants_ef() -> tensor<!EF2> {
  // [1, 2] + [3, 4] = [4, 6] in GF(7²)
  %0 = "stablehlo.constant"() <{value = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %1 = "stablehlo.constant"() <{value = dense<[3, 4]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %2 = stablehlo.add %0, %1 : tensor<!EF2>
  // CHECK: dense<[4, 6]>
  return %2 : tensor<!EF2>
}

// -----

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_negate_constant_ef
func.func @fold_negate_constant_ef() -> tensor<!EF2> {
  // -[1, 2] = [6, 5] in GF(7²)
  %0 = "stablehlo.constant"() <{value = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %1 = stablehlo.negate %0 : tensor<!EF2>
  // CHECK: dense<[6, 5]>
  return %1 : tensor<!EF2>
}
