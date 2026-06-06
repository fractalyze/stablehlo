// RUN: stablehlo-opt %s -stablehlo-canonicalize -split-input-file | FileCheck %s
// RUN: stablehlo-opt %s -stablehlo-aggressive-simplification -split-input-file | FileCheck %s --check-prefix=SIMP

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

// -----

//===----------------------------------------------------------------------===//
// Extension field identity folding
//===----------------------------------------------------------------------===//

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_add_zero_identity_ef
func.func @fold_add_zero_identity_ef(%arg0: tensor<!EF2>) -> tensor<!EF2> {
  %zero = "stablehlo.constant"() <{value = dense<[0, 0]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %result = stablehlo.add %arg0, %zero : tensor<!EF2>
  // CHECK-NOT: stablehlo.add
  // CHECK: return %arg0
  return %result : tensor<!EF2>
}

// -----

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_sub_zero_identity_ef
func.func @fold_sub_zero_identity_ef(%arg0: tensor<!EF2>) -> tensor<!EF2> {
  %zero = "stablehlo.constant"() <{value = dense<[0, 0]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %result = stablehlo.subtract %arg0, %zero : tensor<!EF2>
  // CHECK-NOT: stablehlo.subtract
  // CHECK: return %arg0
  return %result : tensor<!EF2>
}

// -----

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_mul_one_identity_ef
func.func @fold_mul_one_identity_ef(%arg0: tensor<!EF2>) -> tensor<!EF2> {
  // EF "1" is the constant-term embedding [1, 0].
  %one = "stablehlo.constant"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %result = stablehlo.multiply %arg0, %one : tensor<!EF2>
  // CHECK-NOT: stablehlo.multiply
  // CHECK: return %arg0
  return %result : tensor<!EF2>
}

// -----

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

// CHECK-LABEL: @fold_mul_zero_absorb_ef
func.func @fold_mul_zero_absorb_ef(%arg0: tensor<!EF2>) -> tensor<!EF2> {
  %zero = "stablehlo.constant"() <{value = dense<[0, 0]> : tensor<2xi32>}> : () -> tensor<!EF2>
  %result = stablehlo.multiply %arg0, %zero : tensor<!EF2>
  // CHECK-NOT: stablehlo.multiply
  // CHECK: dense<0>
  return %result : tensor<!EF2>
}

// -----

//===----------------------------------------------------------------------===//
// Iota over a field element type must not fold to a constant
//
// IotaOp_FoldScalarToZero (StablehloAggressiveSimplificationPatterns.td) folds
// a size-1 iota dim to constant(0) for int/float element types; the
// NotPrimeIrFieldOrEcType guard keeps it from firing on a field element type
// (a stablehlo.constant DenseElementsAttr can't carry a field type). The fold
// lives in the aggressive-simplification pattern set, not the prime-ir
// round-trip pipeline, so this case is pinned under the SIMP run line.
//===----------------------------------------------------------------------===//

!pf7 = !field.pf<7:i32>

// SIMP-LABEL: @iota_field_size1_not_folded
func.func @iota_field_size1_not_folded() -> tensor<1x!pf7> {
  %0 = stablehlo.iota dim = 0 : tensor<1x!pf7>
  // SIMP: stablehlo.iota
  // SIMP-NOT: stablehlo.constant
  return %0 : tensor<1x!pf7>
}

// -----

// Contrast case: the same size-1 iota over an int element type DOES fold to a
// zero constant under the aggressive-simplification run, confirming the guard
// (not the shape) is what blocks the field case above.

// SIMP-LABEL: @iota_int_size1_folds
func.func @iota_int_size1_folds() -> tensor<1xi32> {
  %0 = stablehlo.iota dim = 0 : tensor<1xi32>
  // SIMP-NOT: stablehlo.iota
  // SIMP: stablehlo.constant
  return %0 : tensor<1xi32>
}

// -----

//===----------------------------------------------------------------------===//
// EC group-law round-trip: non-constant args survive the prime-ir round-trip
// (ConvertEC*  →  canonicalize  →  ConvertEC*Back) unchanged.
//===----------------------------------------------------------------------===//

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: @ec_add_roundtrip
func.func @ec_add_roundtrip(%a: tensor<2x!jac>, %b: tensor<2x!jac>)
    -> tensor<2x!jac> {
  // CHECK: stablehlo.add %arg0, %arg1 : tensor<2x!{{.*}}>
  // CHECK-NOT: elliptic_curve.add
  %0 = stablehlo.add %a, %b : tensor<2x!jac>
  return %0 : tensor<2x!jac>
}

// -----

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: @ec_sub_roundtrip
func.func @ec_sub_roundtrip(%a: tensor<2x!jac>, %b: tensor<2x!jac>)
    -> tensor<2x!jac> {
  // CHECK: stablehlo.subtract %arg0, %arg1 : tensor<2x!{{.*}}>
  // CHECK-NOT: elliptic_curve.sub
  %0 = stablehlo.subtract %a, %b : tensor<2x!jac>
  return %0 : tensor<2x!jac>
}

// -----

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: @ec_negate_roundtrip
func.func @ec_negate_roundtrip(%a: tensor<2x!jac>) -> tensor<2x!jac> {
  // CHECK: stablehlo.negate %arg0 : tensor<2x!{{.*}}>
  // CHECK-NOT: elliptic_curve.negate
  %0 = stablehlo.negate %a : tensor<2x!jac>
  return %0 : tensor<2x!jac>
}

// -----

// scalar*point round-trips back to multiply with the scalar operand first
// (ConvertECScalarMulBack emits multiply(scalar, point)).

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: @ec_scalar_mul_roundtrip
func.func @ec_scalar_mul_roundtrip(%s: tensor<!sf>, %p: tensor<!jac>)
    -> tensor<!jac> {
  // CHECK: stablehlo.multiply %arg0, %arg1
  // CHECK-NOT: elliptic_curve.scalar_mul
  %0 = stablehlo.multiply %s, %p : (tensor<!sf>, tensor<!jac>) -> tensor<!jac>
  return %0 : tensor<!jac>
}

// -----

//===----------------------------------------------------------------------===//
// Tensor field constant shape matching (inverse)
//===----------------------------------------------------------------------===//

!pf_bb = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @divide_broadcast_field_constant
func.func @divide_broadcast_field_constant(%arg0: tensor<4x!pf_bb>) -> tensor<4x!pf_bb> {
  // Scalar field constant broadcast then divided — the crash pattern.
  // ConvertFieldInverseBack must create the "1" constant matching the
  // result shape, not scalar. Without the fix, stablehlo-canonicalize
  // crashes: 'stablehlo.constant' op inferred type(s) 'tensor<i32>'
  // incompatible with return type 'tensor<4x!pf>'.
  // CHECK: stablehlo.divide
  // CHECK: return
  %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<!pf_bb>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<!pf_bb>) -> tensor<4x!pf_bb>
  %2 = stablehlo.divide %1, %arg0 : tensor<4x!pf_bb>
  return %2 : tensor<4x!pf_bb>
}

// -----

!pf_bb = !field.pf<2013265921 : i32, true>
!ef4 = !field.ef<4x!pf_bb, 11:i32>

// CHECK-LABEL: @divide_broadcast_field_constant_ef
func.func @divide_broadcast_field_constant_ef(%arg0: tensor<4x!ef4>) -> tensor<4x!ef4> {
  // Exercises ConvertFieldInverseBack on tensor<4x!ef4>: the EF "1"
  // coefficients [1,0,0,0] must be replicated for every tensor element.
  // CHECK: stablehlo.divide
  // CHECK: return
  %0 = "stablehlo.constant"() <{value = dense<[1, 0, 0, 0]> : tensor<4xi32>}> : () -> tensor<!ef4>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<!ef4>) -> tensor<4x!ef4>
  %2 = stablehlo.divide %1, %arg0 : tensor<4x!ef4>
  return %2 : tensor<4x!ef4>
}

// -----

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!aff = !elliptic_curve.affine<#curve>
!jac = !elliptic_curve.jacobian<#curve>

// Point-representation converts must survive the round-trip as
// stablehlo.convert — a leaked elliptic_curve.convert_point_type has no
// HLO export and fails compilation.
// CHECK-LABEL: @ec_convert_roundtrip
func.func @ec_convert_roundtrip(%a: tensor<4x!aff>) -> tensor<4x!jac> {
  // CHECK: stablehlo.convert %arg0 : (tensor<4x!{{.*}}>) -> tensor<4x!{{.*}}>
  // CHECK-NOT: elliptic_curve.convert_point_type
  %0 = stablehlo.convert %a : (tensor<4x!aff>) -> tensor<4x!jac>
  return %0 : tensor<4x!jac>
}
