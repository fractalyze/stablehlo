// RUN: stablehlo-opt --stablehlo-to-prime-ir --split-input-file --verify-diagnostics %s | FileCheck %s

// Field arithmetic over a small prime: stablehlo ops with field-typed
// tensors rewrite to the matching field.* op. Non-field-typed ops (the
// final f32 case) are left untouched, proving the patterns gate on
// element type rather than on op kind.

// CHECK-LABEL: func @field_add
func.func @field_add(%a: tensor<4x!field.pf<7681:i32>>,
                     %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK: field.add
  // CHECK-NOT: stablehlo.add
  %0 = stablehlo.add %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_subtract
func.func @field_subtract(%a: tensor<4x!field.pf<7681:i32>>,
                          %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK: field.sub
  // CHECK-NOT: stablehlo.subtract
  %0 = stablehlo.subtract %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_multiply
func.func @field_multiply(%a: tensor<4x!field.pf<7681:i32>>,
                          %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK: field.mul
  // CHECK-NOT: stablehlo.multiply
  %0 = stablehlo.multiply %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// stablehlo.divide(x, y) over a field becomes x * inverse(y); there is no
// direct field.div, so the rewrite synthesizes the inverse explicitly.

// CHECK-LABEL: func @field_divide
func.func @field_divide(%a: tensor<4x!field.pf<7681:i32>>,
                        %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK-DAG: %[[INV:.*]] = field.inverse %arg1
  // CHECK-DAG: field.mul %arg0, %[[INV]]
  // CHECK-NOT: stablehlo.divide
  %0 = stablehlo.divide %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_negate
func.func @field_negate(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  // CHECK: field.negate
  // CHECK-NOT: stablehlo.negate
  %0 = stablehlo.negate %a : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// Elliptic-curve point arithmetic. Jacobian (and XYZZ) coordinates are
// closed under the EC group law; affine is NOT — affine+affine yields
// jacobian/xyzz, never affine. The pass forwards the StableHLO result
// type as-is, so the IR has to be expressed in a coordinate system that
// the EC dialect's verifier accepts.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @ec_add
func.func @ec_add(%a: tensor<2x!jac>, %b: tensor<2x!jac>)
    -> tensor<2x!jac> {
  // CHECK: elliptic_curve.add
  // CHECK-NOT: stablehlo.add
  %0 = stablehlo.add %a, %b : tensor<2x!jac>
  func.return %0 : tensor<2x!jac>
}

// -----

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @ec_negate
func.func @ec_negate(%a: tensor<2x!jac>) -> tensor<2x!jac> {
  // CHECK: elliptic_curve.negate
  // CHECK-NOT: stablehlo.negate
  %0 = stablehlo.negate %a : tensor<2x!jac>
  func.return %0 : tensor<2x!jac>
}

// -----

// scalar*point and point*scalar both lower to elliptic_curve.scalar_mul
// with scalar-first canonicalization. Crossing field and point element
// types in stablehlo.multiply is allowed by the mixed-type compat
// extension in Base.cpp.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @ec_scalar_mul_left
func.func @ec_scalar_mul_left(%s: tensor<!sf>, %p: tensor<!jac>)
    -> tensor<!jac> {
  // CHECK: elliptic_curve.scalar_mul %arg0, %arg1
  // CHECK-NOT: stablehlo.multiply
  %0 = stablehlo.multiply %s, %p : (tensor<!sf>, tensor<!jac>) -> tensor<!jac>
  func.return %0 : tensor<!jac>
}

// -----

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @ec_scalar_mul_right
func.func @ec_scalar_mul_right(%p: tensor<!jac>, %s: tensor<!sf>)
    -> tensor<!jac> {
  // CHECK: elliptic_curve.scalar_mul %arg1, %arg0
  // CHECK-NOT: stablehlo.multiply
  %0 = stablehlo.multiply %p, %s : (tensor<!jac>, tensor<!sf>) -> tensor<!jac>
  func.return %0 : tensor<!jac>
}

// -----

// PF × EF: prime field added to extension field over the same prime
// (compat allowed when PF is the EF's base field). The rewrite reuses
// the field.add op since both operands are FieldTypeInterface.

!pf = !field.pf<7:i32>
!ef = !field.ef<2x!pf, 6:i32>

// CHECK-LABEL: func @field_pf_ef_add
func.func @field_pf_ef_add(%a: tensor<4x!ef>, %b: tensor<4x!pf>)
    -> tensor<4x!ef> {
  // CHECK: field.add
  // CHECK-NOT: stablehlo.add
  %0 = stablehlo.add %a, %b : (tensor<4x!ef>, tensor<4x!pf>) -> tensor<4x!ef>
  func.return %0 : tensor<4x!ef>
}

// -----

// Negative case: float ops are not field-typed, so the pass leaves them
// alone. Guards against an over-eager pattern that ignores element type.

// CHECK-LABEL: func @float_add_unchanged
func.func @float_add_unchanged(%a: tensor<4xf32>, %b: tensor<4xf32>)
    -> tensor<4xf32> {
  // CHECK: stablehlo.add
  // CHECK-NOT: field.add
  %0 = stablehlo.add %a, %b : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// Non-affine inputs (jacobian here) leave the op untouched: prime-ir's
// PairingCheckOp requires affine, and upstream is expected to insert
// convert_point_type ops to materialize affine coordinates.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @pairing_check_jacobian_unchanged
func.func @pairing_check_jacobian_unchanged(%g1: tensor<4x!jac>,
                                            %g2: tensor<4x!jac>)
    -> tensor<i1> {
  // CHECK: stablehlo.pairing_check
  // CHECK-NOT: elliptic_curve.pairing_check
  %0 = stablehlo.pairing_check %g1, %g2
      : (tensor<4x!jac>, tensor<4x!jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// G1 affine × G1 affine (both over the prime base field) doesn't fire:
// prime-ir's PairingCheckOp requires the second operand to be over a
// degree-2 extension base field. The pre-check in ConvertPairingCheck
// avoids producing IR that the EC verifier would reject. The happy
// path (G1×G2 with proper Fp2 G2 curve) needs G2 PrimitiveType support;
// this case lands once G2 storage widths are added at the xla layer.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!aff = !elliptic_curve.affine<#curve>

// CHECK-LABEL: func @pairing_check_g1xg1_pf_unchanged
func.func @pairing_check_g1xg1_pf_unchanged(%g1: tensor<4x!aff>,
                                            %g2: tensor<4x!aff>)
    -> tensor<i1> {
  // CHECK: stablehlo.pairing_check
  // CHECK-NOT: elliptic_curve.pairing_check
  %0 = stablehlo.pairing_check %g1, %g2
      : (tensor<4x!aff>, tensor<4x!aff>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
