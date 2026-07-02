// Copyright 2026 The StableHLO(Fractalyze) Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//
// Verification tests for ZK-only StableHLO ops.
// These operations are not in the reference ops_stablehlo.mlir file.
// They are either ZK-specific or basic arithmetic/bitwise ops that
// the reference tests only as part of other op tests.

// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// =============================================================================
// AddOp
// =============================================================================

// CHECK-LABEL: func @add_i32
func.func @add_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// SubtractOp
// =============================================================================

// CHECK-LABEL: func @subtract_i32
func.func @subtract_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// MultiplyOp
// =============================================================================

// CHECK-LABEL: func @multiply_i32
func.func @multiply_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// DivideOp
// =============================================================================

// CHECK-LABEL: func @divide_i32
func.func @divide_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// RemainderOp
// =============================================================================

// CHECK-LABEL: func @remainder_i32
func.func @remainder_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// PowerOp
// =============================================================================

// CHECK-LABEL: func @power_i32
func.func @power_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.power %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: func @power_field_with_int_exponent
func.func @power_field_with_int_exponent(%base: tensor<4x!pf>, %exp: tensor<4xi32>) -> tensor<4x!pf> {
  // Field base with integer exponent (ZK power pattern)
  %0 = stablehlo.power %base, %exp : (tensor<4x!pf>, tensor<4xi32>) -> tensor<4x!pf>
  func.return %0 : tensor<4x!pf>
}

// -----

!pf2 = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: func @power_field_scalar
func.func @power_field_scalar(%base: tensor<!pf2>, %exp: tensor<i32>) -> tensor<!pf2> {
  // Scalar field power
  %0 = stablehlo.power %base, %exp : (tensor<!pf2>, tensor<i32>) -> tensor<!pf2>
  func.return %0 : tensor<!pf2>
}

// -----

// =============================================================================
// NegateOp
// =============================================================================

// CHECK-LABEL: func @negate_i32
func.func @negate_i32(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.negate %arg0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// SignOp
// =============================================================================

// CHECK-LABEL: func @sign_i32
func.func @sign_i32(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.sign %arg0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// CountLeadingZerosOp
// =============================================================================

// CHECK-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.count_leading_zeros %arg0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// PopulationCountOp
// =============================================================================

// CHECK-LABEL: func @popcnt
func.func @popcnt(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.popcnt %arg0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// NotOp
// =============================================================================

// CHECK-LABEL: func @not_i32
func.func @not_i32(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.not %arg0 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// MaximumOp
// =============================================================================

// CHECK-LABEL: func @maximum_i32
func.func @maximum_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// MinimumOp
// =============================================================================

// CHECK-LABEL: func @minimum_i32
func.func @minimum_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// AndOp
// =============================================================================

// CHECK-LABEL: func @and_i32
func.func @and_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @and_i1
func.func @and_i1(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// OrOp
// =============================================================================

// CHECK-LABEL: func @or_i32
func.func @or_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// XorOp
// =============================================================================

// CHECK-LABEL: func @xor_i32
func.func @xor_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// ShiftLeftOp
// =============================================================================

// CHECK-LABEL: func @shift_left
func.func @shift_left(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.shift_left %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// ShiftRightArithmeticOp
// =============================================================================

// CHECK-LABEL: func @shift_right_arithmetic
func.func @shift_right_arithmetic(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// ShiftRightLogicalOp
// =============================================================================

// CHECK-LABEL: func @shift_right_logical
func.func @shift_right_logical(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// BitReverseOp
// =============================================================================

// CHECK-LABEL: func @bit_reverse_1d
func.func @bit_reverse_1d(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = stablehlo.bit_reverse %arg0, dims = [0] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @bit_reverse_2d
func.func @bit_reverse_2d(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %0 = stablehlo.bit_reverse %arg0, dims = [0, 1] : tensor<4x8xi32>
  func.return %0 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: func @bit_reverse_no_dims
func.func @bit_reverse_no_dims(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = stablehlo.bit_reverse %arg0, dims = [] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----

func.func @bit_reverse_duplicate_dims(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{dimensions should be unique. Got: 0, 0}}
  %0 = stablehlo.bit_reverse %arg0, dims = [0, 0] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----

func.func @bit_reverse_negative_dim(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = stablehlo.bit_reverse %arg0, dims = [-1] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----

func.func @bit_reverse_out_of_range_dim(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{all dimensions should be between [0, 1). Got dimension: 1.}}
  %0 = stablehlo.bit_reverse %arg0, dims = [1] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// -----

func.func @bit_reverse_non_power_of_2(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  // expected-error @+1 {{dimension size must be a power of 2, got 6 for dimension 0.}}
  %0 = stablehlo.bit_reverse %arg0, dims = [0] : tensor<6xi32>
  func.return %0 : tensor<6xi32>
}

// -----

// =============================================================================
// ReduceOp
// =============================================================================

// CHECK-LABEL: func @reduce
func.func @reduce(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = "stablehlo.reduce"(%arg0, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {dimensions = array<i64: 1>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// -----

// =============================================================================
// EC Arithmetic — NegateOp
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>
!g1_jacobian = !elliptic_curve.jacobian<#g1_curve>
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @negate_ec_affine
func.func @negate_ec_affine(%arg0: tensor<4x!g1_affine>) -> tensor<4x!g1_affine> {
  %0 = stablehlo.negate %arg0 : tensor<4x!g1_affine>
  func.return %0 : tensor<4x!g1_affine>
}

// CHECK-LABEL: func @negate_ec_jacobian
func.func @negate_ec_jacobian(%arg0: tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian> {
  %0 = stablehlo.negate %arg0 : tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @negate_ec_xyzz
func.func @negate_ec_xyzz(%arg0: tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz> {
  %0 = stablehlo.negate %arg0 : tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// -----

// =============================================================================
// EC Arithmetic — AddOp (same type, pretty format)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_jacobian = !elliptic_curve.jacobian<#g1_curve>

// CHECK-LABEL: func @add_ec_jacobian
func.func @add_ec_jacobian(%a: tensor<4x!g1_jacobian>, %b: tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian> {
  %0 = stablehlo.add %a, %b : tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// -----

// =============================================================================
// EC Arithmetic — AddOp (mixed types, generic format)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>
!g1_jacobian = !elliptic_curve.jacobian<#g1_curve>
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @add_ec_affine_to_jacobian
func.func @add_ec_affine_to_jacobian(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_affine>) -> tensor<4x!g1_jacobian> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!g1_affine>, tensor<4x!g1_affine>) -> tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @add_ec_affine_to_xyzz
func.func @add_ec_affine_to_xyzz(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_affine>) -> tensor<4x!g1_xyzz> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!g1_affine>, tensor<4x!g1_affine>) -> tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// CHECK-LABEL: func @add_ec_affine_jacobian
func.func @add_ec_affine_jacobian(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!g1_affine>, tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @add_ec_xyzz_affine
func.func @add_ec_xyzz_affine(%a: tensor<4x!g1_xyzz>, %b: tensor<4x!g1_affine>) -> tensor<4x!g1_xyzz> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!g1_xyzz>, tensor<4x!g1_affine>) -> tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// -----

// =============================================================================
// EC Arithmetic — SubtractOp (same type, pretty format)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @subtract_ec_xyzz
func.func @subtract_ec_xyzz(%a: tensor<4x!g1_xyzz>, %b: tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz> {
  %0 = stablehlo.subtract %a, %b : tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// -----

// =============================================================================
// EC Arithmetic — SubtractOp (mixed types, generic format)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @subtract_ec_affine_xyzz
func.func @subtract_ec_affine_xyzz(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz> {
  %0 = "stablehlo.subtract"(%a, %b) : (tensor<4x!g1_affine>, tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// -----

// =============================================================================
// EC Arithmetic — G2 (extension field base)
// =============================================================================

!pf_g2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!qf = !field.ef<2x!pf_g2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !qf
!g2_affine = !elliptic_curve.affine<#g2_curve>

// CHECK-LABEL: func @add_ec_g2_affine
func.func @add_ec_g2_affine(%a: tensor<4x!g2_affine>, %b: tensor<4x!g2_affine>) -> tensor<4x!g2_affine> {
  // G2 affine+affine → affine is not valid; use jacobian result
  func.return %a : tensor<4x!g2_affine>
}

// -----

// =============================================================================
// EC Arithmetic — G2 add (generic format)
// =============================================================================

!pf_g2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!qf = !field.ef<2x!pf_g2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !qf
!g2_affine = !elliptic_curve.affine<#g2_curve>
!g2_jacobian = !elliptic_curve.jacobian<#g2_curve>

// CHECK-LABEL: func @add_ec_g2
func.func @add_ec_g2(%a: tensor<4x!g2_affine>, %b: tensor<4x!g2_affine>) -> tensor<4x!g2_jacobian> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!g2_affine>, tensor<4x!g2_affine>) -> tensor<4x!g2_jacobian>
  func.return %0 : tensor<4x!g2_jacobian>
}

// -----

// =============================================================================
// EC Compare — EQ/NE (positive tests)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>

// CHECK-LABEL: func @compare_ec_eq
func.func @compare_ec_eq(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_affine>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %a, %b : (tensor<4x!g1_affine>, tensor<4x!g1_affine>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// CHECK-LABEL: func @compare_ec_ne
func.func @compare_ec_ne(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_affine>) -> tensor<4xi1> {
  %0 = stablehlo.compare NE, %a, %b : (tensor<4x!g1_affine>, tensor<4x!g1_affine>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// EC Compare — ordered direction (negative test)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>

func.func @compare_ec_lt_invalid(%a: tensor<4x!g1_affine>, %b: tensor<4x!g1_affine>) -> tensor<4xi1> {
  // expected-error @+1 {{EC point types only support EQ and NE comparisons}}
  %0 = stablehlo.compare LT, %a, %b : (tensor<4x!g1_affine>, tensor<4x!g1_affine>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// EC Add — invalid type combination (negative test)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>

func.func @add_ec_affine_affine_invalid(%a: tensor<!g1_affine>, %b: tensor<!g1_affine>) -> tensor<!g1_affine> {
  // expected-error @+1 {{invalid EC point type combination for binary operation}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<!g1_affine>, tensor<!g1_affine>) -> tensor<!g1_affine>
  func.return %0 : tensor<!g1_affine>
}

// -----

// =============================================================================
// ExtensionField Compare — ordered direction (negative test)
// =============================================================================

!pf =!field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ef2 = !field.ef<2x!pf, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

func.func @compare_ef_lt_invalid(%a: tensor<4x!ef2>, %b: tensor<4x!ef2>) -> tensor<4xi1> {
  // expected-error @+1 {{extension field types only support EQ and NE comparisons}}
  %0 = stablehlo.compare LT, %a, %b : (tensor<4x!ef2>, tensor<4x!ef2>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// NttOp — Number Theoretic Transform
// =============================================================================

!pf_babybear_mont = !field.pf<2013265921:i32, true>

// CHECK-LABEL: func @fft_1d
func.func @fft_1d(%x: tensor<1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = NTT, length = 1024 : tensor<1024x!pf_babybear_mont>
  func.return %0 : tensor<1024x!pf_babybear_mont>
}

// CHECK-LABEL: func @ifft_1d
func.func @ifft_1d(%x: tensor<1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = INTT, length = 1024 : tensor<1024x!pf_babybear_mont>
  func.return %0 : tensor<1024x!pf_babybear_mont>
}

// CHECK-LABEL: func @fft_2d
func.func @fft_2d(%x: tensor<1024x16x!pf_babybear_mont>) -> tensor<1024x16x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = NTT, length = 1024 : tensor<1024x16x!pf_babybear_mont>
  func.return %0 : tensor<1024x16x!pf_babybear_mont>
}

// CHECK-LABEL: func @ifft_2d
func.func @ifft_2d(%x: tensor<1024x16x!pf_babybear_mont>) -> tensor<1024x16x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = INTT, length = 1024 : tensor<1024x16x!pf_babybear_mont>
  func.return %0 : tensor<1024x16x!pf_babybear_mont>
}

// CHECK-LABEL: func @ntt_with_generator
func.func @ntt_with_generator(%x: tensor<1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = NTT, length = 1024, generator = 5 : tensor<1024x!pf_babybear_mont>
  func.return %0 : tensor<1024x!pf_babybear_mont>
}

// CHECK-LABEL: func @negacyclic_ntt_1d
func.func @negacyclic_ntt_1d(%x: tensor<1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = NEGACYCLIC_NTT, length = 1024 : tensor<1024x!pf_babybear_mont>
  func.return %0 : tensor<1024x!pf_babybear_mont>
}

// CHECK-LABEL: func @negacyclic_intt_1d
func.func @negacyclic_intt_1d(%x: tensor<1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont> {
  %0 = stablehlo.ntt %x, type = NEGACYCLIC_INTT, length = 1024 : tensor<1024x!pf_babybear_mont>
  func.return %0 : tensor<1024x!pf_babybear_mont>
}

// -----

// =============================================================================
// EC Scalar Multiplication — field × EC point (positive tests)
// =============================================================================

!pf_g1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf_g1 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_g1
!g1_affine = !elliptic_curve.affine<#g1_curve>
!g1_jacobian = !elliptic_curve.jacobian<#g1_curve>
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @multiply_field_ec_affine
func.func @multiply_field_ec_affine(%a: tensor<4x!sf_g1>, %b: tensor<4x!g1_affine>) -> tensor<4x!g1_jacobian> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!sf_g1>, tensor<4x!g1_affine>) -> tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @multiply_ec_affine_field
func.func @multiply_ec_affine_field(%a: tensor<4x!g1_affine>, %b: tensor<4x!sf_g1>) -> tensor<4x!g1_jacobian> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!g1_affine>, tensor<4x!sf_g1>) -> tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @multiply_field_ec_jacobian
func.func @multiply_field_ec_jacobian(%a: tensor<4x!sf_g1>, %b: tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!sf_g1>, tensor<4x!g1_jacobian>) -> tensor<4x!g1_jacobian>
  func.return %0 : tensor<4x!g1_jacobian>
}

// CHECK-LABEL: func @multiply_field_ec_xyzz
func.func @multiply_field_ec_xyzz(%a: tensor<4x!sf_g1>, %b: tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!sf_g1>, tensor<4x!g1_xyzz>) -> tensor<4x!g1_xyzz>
  func.return %0 : tensor<4x!g1_xyzz>
}

// -----

// =============================================================================
// PairingCheckOp — BN254 multi-pairing check
// =============================================================================

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_affine = !elliptic_curve.affine<#g1_curve>

!PF_G2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!QF = !field.ef<2x!PF_G2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
!g2_affine = !elliptic_curve.affine<#g2_curve>

// CHECK-LABEL: func @pairing_check_bn254
func.func @pairing_check_bn254(%g1: tensor<4x!g1_affine>, %g2: tensor<4x!g2_affine>) -> tensor<i1> {
  %0 = stablehlo.pairing_check %g1, %g2 : (tensor<4x!g1_affine>, tensor<4x!g2_affine>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// =============================================================================
// MsmOp — BN254 multi-scalar multiplication
// =============================================================================

!BN254_Fr = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

// CHECK-LABEL: func @msm_bn254_defaults
func.func @msm_bn254_defaults(%scalars: tensor<1024x!BN254_Fr>, %bases: tensor<1024x!g1_affine>) -> tensor<!g1_affine> {
  %0 = stablehlo.msm %scalars, %bases : (tensor<1024x!BN254_Fr>, tensor<1024x!g1_affine>) -> tensor<!g1_affine>
  func.return %0 : tensor<!g1_affine>
}

// CHECK-LABEL: func @msm_bn254_with_config
func.func @msm_bn254_with_config(%scalars: tensor<1024x!BN254_Fr>, %bases: tensor<1024x!g1_affine>) -> tensor<!g1_xyzz> {
  %0 = stablehlo.msm %scalars, %bases {window_bits = 16 : i32, precompute_factor = 2 : i32, bitsize = 253 : i32} : (tensor<1024x!BN254_Fr>, tensor<1024x!g1_affine>) -> tensor<!g1_xyzz>
  func.return %0 : tensor<!g1_xyzz>
}

// CHECK-LABEL: func @msm_bn254_batched
func.func @msm_bn254_batched(%scalars: tensor<2048x!BN254_Fr>, %bases: tensor<1024x!g1_affine>) -> tensor<2x!g1_xyzz> {
  %0 = stablehlo.msm %scalars, %bases {window_bits = 16 : i32, batch_size = 2 : i32, are_points_shared = true} : (tensor<2048x!BN254_Fr>, tensor<1024x!g1_affine>) -> tensor<2x!g1_xyzz>
  func.return %0 : tensor<2x!g1_xyzz>
}

// -----

// =============================================================================
// PF × EF Arithmetic — prime_field × extension_field (positive tests)
// =============================================================================

!pf_bb = !field.pf<2013265921:i32>
!ef_bb4 = !field.ef<4x!pf_bb, 11:i32>

// CHECK-LABEL: func @add_pf_ef
func.func @add_pf_ef(%a: tensor<4x!pf_bb>, %b: tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!pf_bb>, tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @add_ef_pf
func.func @add_ef_pf(%a: tensor<4x!ef_bb4>, %b: tensor<4x!pf_bb>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!ef_bb4>, tensor<4x!pf_bb>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @subtract_pf_ef
func.func @subtract_pf_ef(%a: tensor<4x!pf_bb>, %b: tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.subtract"(%a, %b) : (tensor<4x!pf_bb>, tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @subtract_ef_pf
func.func @subtract_ef_pf(%a: tensor<4x!ef_bb4>, %b: tensor<4x!pf_bb>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.subtract"(%a, %b) : (tensor<4x!ef_bb4>, tensor<4x!pf_bb>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @multiply_pf_ef
func.func @multiply_pf_ef(%a: tensor<4x!pf_bb>, %b: tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!pf_bb>, tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @multiply_ef_pf
func.func @multiply_ef_pf(%a: tensor<4x!ef_bb4>, %b: tensor<4x!pf_bb>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!ef_bb4>, tensor<4x!pf_bb>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @divide_pf_ef
func.func @divide_pf_ef(%a: tensor<4x!pf_bb>, %b: tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.divide"(%a, %b) : (tensor<4x!pf_bb>, tensor<4x!ef_bb4>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// CHECK-LABEL: func @divide_ef_pf
func.func @divide_ef_pf(%a: tensor<4x!ef_bb4>, %b: tensor<4x!pf_bb>) -> tensor<4x!ef_bb4> {
  %0 = "stablehlo.divide"(%a, %b) : (tensor<4x!ef_bb4>, tensor<4x!pf_bb>) -> tensor<4x!ef_bb4>
  func.return %0 : tensor<4x!ef_bb4>
}

// -----

// =============================================================================
// i128 / i256 integer ops (extended widths for BN254 bit decomposition)
// =============================================================================

// CHECK-LABEL: func @and_i256
func.func @and_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @or_i256
func.func @or_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @xor_i256
func.func @xor_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @shift_left_i256
func.func @shift_left_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.shift_left %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @shift_right_arithmetic_i256
func.func @shift_right_arithmetic_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @shift_right_logical_i256
func.func @shift_right_logical_i256(%arg0: tensor<4xi256>, %arg1: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @and_i128
func.func @and_i128(%arg0: tensor<4xi128>, %arg1: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// CHECK-LABEL: func @shift_right_logical_i128
func.func @shift_right_logical_i128(%arg0: tensor<4xi128>, %arg1: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// CHECK-LABEL: func @convert_i256_to_pf
!pf_bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
func.func @convert_i256_to_pf(%arg0: tensor<4xi256>) -> tensor<4x!pf_bn254> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi256>) -> tensor<4x!pf_bn254>
  func.return %0 : tensor<4x!pf_bn254>
}

// -----

// CHECK-LABEL: func @convert_pf_to_i256
!pf_bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
func.func @convert_pf_to_i256(%arg0: tensor<4x!pf_bn254>) -> tensor<4xi256> {
  %0 = stablehlo.convert %arg0 : (tensor<4x!pf_bn254>) -> tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// -----

// CHECK-LABEL: func @power_pf_i256
!pf_bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
func.func @power_pf_i256(%base: tensor<!pf_bn254>, %exp: tensor<i256>) -> tensor<!pf_bn254> {
  %0 = "stablehlo.power"(%base, %exp) : (tensor<!pf_bn254>, tensor<i256>) -> tensor<!pf_bn254>
  func.return %0 : tensor<!pf_bn254>
}

// -----

// CHECK-LABEL: func @convert_i128_to_pf
!pf_bn254 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
func.func @convert_i128_to_pf(%arg0: tensor<4xi128>) -> tensor<4x!pf_bn254> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi128>) -> tensor<4x!pf_bn254>
  func.return %0 : tensor<4x!pf_bn254>
}

// -----

// =============================================================================
// AddOp — PF + EF where PF is NOT the EF's base field (negative)
// =============================================================================

// The PF+EF fast path in verifyAddOp requires ef.getBaseField() == pf. With an
// unrelated PF the match fails and control falls through to the homogeneous
// element-type check, which rejects the mismatch.

!pf_a = !field.pf<2013265921:i32>
!pf_other = !field.pf<2130706433:i32>
!ef_a4 = !field.ef<4x!pf_a, 11:i32>

func.func @add_pf_ef_wrong_base(%a: tensor<4x!pf_other>, %b: tensor<4x!ef_a4>) -> tensor<4x!ef_a4> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!pf_other>, tensor<4x!ef_a4>) -> tensor<4x!ef_a4>
  func.return %0 : tensor<4x!ef_a4>
}

// -----

// =============================================================================
// AddOp — PF + EF with result element = PF, not the EF (negative)
// =============================================================================

// Even when the PF is the EF's base field, the result must keep the wider EF
// element type (resEl == rhsEl). A PF result fails the fast path and falls
// through to the homogeneous element-type check.

!pf_bb = !field.pf<2013265921:i32>
!ef_bb4 = !field.ef<4x!pf_bb, 11:i32>

func.func @add_pf_ef_result_pf(%a: tensor<4x!pf_bb>, %b: tensor<4x!ef_bb4>) -> tensor<4x!pf_bb> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!pf_bb>, tensor<4x!ef_bb4>) -> tensor<4x!pf_bb>
  func.return %0 : tensor<4x!pf_bb>
}

// -----

// =============================================================================
// AddOp — two points on DIFFERENT curves (negative)
// =============================================================================

!pf_c = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#curve_a = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_c
#curve_b = #elliptic_curve.sw<0:i256, 1:i256, (2:i256, 3:i256)> : !pf_c
!affine_a = !elliptic_curve.affine<#curve_a>
!affine_b = !elliptic_curve.affine<#curve_b>
!jacobian_a = !elliptic_curve.jacobian<#curve_a>

func.func @add_ec_different_curves(%a: tensor<4x!affine_a>, %b: tensor<4x!affine_b>) -> tensor<4x!jacobian_a> {
  // expected-error @+1 {{EC operands and result must be on the same curve}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<4x!affine_a>, tensor<4x!affine_b>) -> tensor<4x!jacobian_a>
  func.return %0 : tensor<4x!jacobian_a>
}

// -----

// =============================================================================
// AddOp — homogeneous element-type mismatch i32 + f32 (negative)
// =============================================================================

func.func @add_i32_f32_mismatch(%a: tensor<4xi32>, %b: tensor<4xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{op requires the same element type for all operands and results}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<4xi32>, tensor<4xf32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// MultiplyOp — scalar × point with NON-point result (negative)
// =============================================================================

// MulOp carries InferTypeOpInterface: inferMulOp resolves field×point to the
// point operand type, so a non-point explicit result is rejected at the
// infer-vs-result reconciliation before verifyMulOp's scalar-mult check runs.

!pf_m = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf_m = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#curve_m = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_m
!affine_m = !elliptic_curve.affine<#curve_m>

func.func @multiply_scalar_point_nonpoint_result(%a: tensor<4x!sf_m>, %b: tensor<4x!affine_m>) -> tensor<4x!sf_m> {
  // expected-error @+1 {{scalar multiplication result must be an EC point on the same curve as the point operand}}
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!sf_m>, tensor<4x!affine_m>) -> tensor<4x!sf_m>
  func.return %0 : tensor<4x!sf_m>
}

// -----

// =============================================================================
// MultiplyOp — scalar × point with result point on DIFFERENT curve (negative)
// =============================================================================

// Same infer-then-reconcile path: inferMulOp picks the point operand's curve;
// a result point on a different curve fails reconciliation before the verifier.

!pf_m2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!sf_m2 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#curve_m2 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_m2
#curve_m2b = #elliptic_curve.sw<0:i256, 1:i256, (2:i256, 3:i256)> : !pf_m2
!affine_m2 = !elliptic_curve.affine<#curve_m2>
!jacobian_m2b = !elliptic_curve.jacobian<#curve_m2b>

func.func @multiply_scalar_point_wrong_curve(%a: tensor<4x!sf_m2>, %b: tensor<4x!affine_m2>) -> tensor<4x!jacobian_m2b> {
  // expected-error @+1 {{scalar multiplication result must be an EC point on the same curve as the point operand}}
  %0 = "stablehlo.multiply"(%a, %b) : (tensor<4x!sf_m2>, tensor<4x!affine_m2>) -> tensor<4x!jacobian_m2b>
  func.return %0 : tensor<4x!jacobian_m2b>
}

// -----

// =============================================================================
// PowerOp — field base + int exponent, result element ≠ base (negative)
// =============================================================================

!pf_p = !field.pf<2013265921:i32>
!pf_p_other = !field.pf<2130706433:i32>

func.func @power_field_result_mismatch(%base: tensor<4x!pf_p>, %exp: tensor<4xi32>) -> tensor<4x!pf_p_other> {
  // expected-error @+1 {{stablehlo.power result element type must match base}}
  %0 = "stablehlo.power"(%base, %exp) : (tensor<4x!pf_p>, tensor<4xi32>) -> tensor<4x!pf_p_other>
  func.return %0 : tensor<4x!pf_p_other>
}

// -----

// =============================================================================
// PowerOp — i32 base with f32 exponent, homogeneous mismatch (negative)
// =============================================================================

func.func @power_i32_f32_mismatch(%base: tensor<4xi32>, %exp: tensor<4xf32>) -> tensor<4xi32> {
  // expected-error @+1 {{stablehlo.power requires a compatible element type for base, exponent, and result}}
  %0 = "stablehlo.power"(%base, %exp) : (tensor<4xi32>, tensor<4xf32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// PairingCheckOp — unrelated curves, no bilinear pair (negative)
// =============================================================================

// G1 and G2 are both prime-field curves with different curve constants: neither
// the same-curve branch nor the (PF, EF-over-PF) bilinear-pair branch matches.

!pf_pc = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#curve_pc1 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_pc
#curve_pc2 = #elliptic_curve.sw<0:i256, 1:i256, (2:i256, 3:i256)> : !pf_pc
!g1_pc = !elliptic_curve.affine<#curve_pc1>
!g2_pc = !elliptic_curve.affine<#curve_pc2>

func.func @pairing_check_unrelated_curves(%g1: tensor<4x!g1_pc>, %g2: tensor<4x!g2_pc>) -> tensor<i1> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{pairing_check operands must be on the same curve, or form a bilinear pair (G1 on a prime field, G2 on a degree>=2 extension of that prime field)}}
  %0 = stablehlo.pairing_check %g1, %g2 : (tensor<4x!g1_pc>, tensor<4x!g2_pc>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// =============================================================================
// MsmOp — rank-2 operands (negative)
// =============================================================================

!Fr_msm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#curve_msm = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!affine_msm = !elliptic_curve.affine<#curve_msm>

func.func @msm_rank2_operands(%scalars: tensor<4x8x!Fr_msm>, %bases: tensor<4x8x!affine_msm>) -> tensor<!affine_msm> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{msm requires rank-1 operands; got ranks 2 and 2.}}
  %0 = stablehlo.msm %scalars, %bases : (tensor<4x8x!Fr_msm>, tensor<4x8x!affine_msm>) -> tensor<!affine_msm>
  func.return %0 : tensor<!affine_msm>
}

// -----

// =============================================================================
// MsmOp — are_points_shared, scalars length ≠ batch*bases (negative)
// =============================================================================

!Fr_msm2 = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#curve_msm2 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!affine_msm2 = !elliptic_curve.affine<#curve_msm2>
!xyzz_msm2 = !elliptic_curve.xyzz<#curve_msm2>

func.func @msm_shared_bad_length(%scalars: tensor<2000x!Fr_msm2>, %bases: tensor<1024x!affine_msm2>) -> tensor<2x!xyzz_msm2> {
  // expected-error @+2 {{failed to infer returned types}}
  // expected-error @+1 {{msm with batch_size=2 and are_points_shared=true requires scalars length == batch_size * bases length; got 2000 and 1024.}}
  %0 = stablehlo.msm %scalars, %bases {batch_size = 2 : i32, are_points_shared = true} : (tensor<2000x!Fr_msm2>, tensor<1024x!affine_msm2>) -> tensor<2x!xyzz_msm2>
  func.return %0 : tensor<2x!xyzz_msm2>
}

// -----

// =============================================================================
// BitReverseOp — over a field element type (positive)
// =============================================================================

!pf_br = !field.pf<2013265921:i32>

// CHECK-LABEL: func @bit_reverse_field
func.func @bit_reverse_field(%arg0: tensor<8x!pf_br>) -> tensor<8x!pf_br> {
  %0 = stablehlo.bit_reverse %arg0, dims = [0] : tensor<8x!pf_br>
  func.return %0 : tensor<8x!pf_br>
}

// -----

// =============================================================================
// ExtensionField Arithmetic — EF×EF, EF÷EF, power, negate (positive)
// =============================================================================

!pf_ef = !field.pf<2013265921:i32>
!ef4 = !field.ef<4x!pf_ef, 11:i32>

// CHECK-LABEL: func @multiply_ef_ef
func.func @multiply_ef_ef(%a: tensor<4x!ef4>, %b: tensor<4x!ef4>) -> tensor<4x!ef4> {
  %0 = stablehlo.multiply %a, %b : tensor<4x!ef4>
  func.return %0 : tensor<4x!ef4>
}

// CHECK-LABEL: func @divide_ef_ef
func.func @divide_ef_ef(%a: tensor<4x!ef4>, %b: tensor<4x!ef4>) -> tensor<4x!ef4> {
  %0 = stablehlo.divide %a, %b : tensor<4x!ef4>
  func.return %0 : tensor<4x!ef4>
}

// CHECK-LABEL: func @power_ef_int
func.func @power_ef_int(%base: tensor<4x!ef4>, %exp: tensor<4xi32>) -> tensor<4x!ef4> {
  %0 = stablehlo.power %base, %exp : (tensor<4x!ef4>, tensor<4xi32>) -> tensor<4x!ef4>
  func.return %0 : tensor<4x!ef4>
}

// CHECK-LABEL: func @negate_ef
func.func @negate_ef(%arg0: tensor<4x!ef4>) -> tensor<4x!ef4> {
  %0 = stablehlo.negate %arg0 : tensor<4x!ef4>
  func.return %0 : tensor<4x!ef4>
}

// -----

// =============================================================================
// CompareOp — prime field EQ and LT (both positive; PF is unrestricted)
// =============================================================================

!pf_cmp = !field.pf<2013265921:i32>

// CHECK-LABEL: func @compare_pf_eq
func.func @compare_pf_eq(%a: tensor<4x!pf_cmp>, %b: tensor<4x!pf_cmp>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %a, %b : (tensor<4x!pf_cmp>, tensor<4x!pf_cmp>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// CHECK-LABEL: func @compare_pf_lt
func.func @compare_pf_lt(%a: tensor<4x!pf_cmp>, %b: tensor<4x!pf_cmp>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %a, %b : (tensor<4x!pf_cmp>, tensor<4x!pf_cmp>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// CompareOp — extension field EQ (positive)
// =============================================================================

!pf_cmp_ef = !field.pf<2013265921:i32>
!ef_cmp = !field.ef<4x!pf_cmp_ef, 11:i32>

// CHECK-LABEL: func @compare_ef_eq
func.func @compare_ef_eq(%a: tensor<4x!ef_cmp>, %b: tensor<4x!ef_cmp>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %a, %b : (tensor<4x!ef_cmp>, tensor<4x!ef_cmp>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// =============================================================================
// SelectOp — over EC point tensors (positive)
// =============================================================================

!pf_sel = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#curve_sel = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_sel
!jacobian_sel = !elliptic_curve.jacobian<#curve_sel>

// CHECK-LABEL: func @select_ec_jacobian
func.func @select_ec_jacobian(%pred: tensor<4xi1>, %a: tensor<4x!jacobian_sel>, %b: tensor<4x!jacobian_sel>) -> tensor<4x!jacobian_sel> {
  %0 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4x!jacobian_sel>
  func.return %0 : tensor<4x!jacobian_sel>
}

// -----

// =============================================================================
// IotaOp — over an extension field element type (positive)
// =============================================================================

!pf_iota = !field.pf<2013265921:i32>
!ef_iota = !field.ef<4x!pf_iota, 11:i32>

// CHECK-LABEL: func @iota_ef
func.func @iota_ef() -> tensor<8x!ef_iota> {
  %0 = stablehlo.iota dim = 0 : tensor<8x!ef_iota>
  func.return %0 : tensor<8x!ef_iota>
}

// -----

// =============================================================================
// PowerOp — EC-point base rejected by ODS operand constraint (negative)
// =============================================================================

!pf_pow_ec = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#curve_pow_ec = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !pf_pow_ec
!affine_pow_ec = !elliptic_curve.affine<#curve_pow_ec>

func.func @power_ec_base_invalid(%base: tensor<4x!affine_pow_ec>, %exp: tensor<4xi32>) -> tensor<4x!affine_pow_ec> {
  // expected-error @+1 {{operand #0 must be ranked tensor of}}
  %0 = "stablehlo.power"(%base, %exp) : (tensor<4x!affine_pow_ec>, tensor<4xi32>) -> tensor<4x!affine_pow_ec>
  func.return %0 : tensor<4x!affine_pow_ec>
}

// -----

// =============================================================================
// Wide-int arithmetic — i256 / i128 (positive)
// =============================================================================

// CHECK-LABEL: func @add_i256
func.func @add_i256(%a: tensor<4xi256>, %b: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.add %a, %b : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @subtract_i256
func.func @subtract_i256(%a: tensor<4xi256>, %b: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.subtract %a, %b : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @multiply_i256
func.func @multiply_i256(%a: tensor<4xi256>, %b: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.multiply %a, %b : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @divide_i256
func.func @divide_i256(%a: tensor<4xi256>, %b: tensor<4xi256>) -> tensor<4xi256> {
  %0 = stablehlo.divide %a, %b : tensor<4xi256>
  func.return %0 : tensor<4xi256>
}

// CHECK-LABEL: func @add_i128
func.func @add_i128(%a: tensor<4xi128>, %b: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.add %a, %b : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// CHECK-LABEL: func @subtract_i128
func.func @subtract_i128(%a: tensor<4xi128>, %b: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.subtract %a, %b : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// CHECK-LABEL: func @multiply_i128
func.func @multiply_i128(%a: tensor<4xi128>, %b: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.multiply %a, %b : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// CHECK-LABEL: func @divide_i128
func.func @divide_i128(%a: tensor<4xi128>, %b: tensor<4xi128>) -> tensor<4xi128> {
  %0 = stablehlo.divide %a, %b : tensor<4xi128>
  func.return %0 : tensor<4xi128>
}

// -----

// =============================================================================
// Wide-int constants — i256 / i128 (positive)
// =============================================================================

// CHECK-LABEL: func @constant_i256
func.func @constant_i256() -> tensor<2xi256> {
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi256>
  func.return %0 : tensor<2xi256>
}

// CHECK-LABEL: func @constant_i128
func.func @constant_i128() -> tensor<2xi128> {
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi128>
  func.return %0 : tensor<2xi128>
}

// -----

// =============================================================================
// Unsigned wide-int arithmetic — ui128 / ui256 (positive; HLO_UInt admits both)
// =============================================================================

// CHECK-LABEL: func @add_ui128
func.func @add_ui128(%a: tensor<4xui128>, %b: tensor<4xui128>) -> tensor<4xui128> {
  %0 = stablehlo.add %a, %b : tensor<4xui128>
  func.return %0 : tensor<4xui128>
}

// CHECK-LABEL: func @add_ui256
func.func @add_ui256(%a: tensor<4xui256>, %b: tensor<4xui256>) -> tensor<4xui256> {
  %0 = stablehlo.add %a, %b : tensor<4xui256>
  func.return %0 : tensor<4xui256>
}
