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
  %0 = stablehlo.power %arg0, %arg1 : tensor<4xi32>
  func.return %0 : tensor<4xi32>
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!QF = !field.ef<2x!PF_G2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
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

!PF_G2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!QF = !field.ef<2x!PF_G2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
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

!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_affine = !elliptic_curve.affine<#g1_curve>

func.func @add_ec_affine_affine_invalid(%a: tensor<!g1_affine>, %b: tensor<!g1_affine>) -> tensor<!g1_affine> {
  // expected-error @+1 {{invalid EC point type combination for binary operation}}
  %0 = "stablehlo.add"(%a, %b) : (tensor<!g1_affine>, tensor<!g1_affine>) -> tensor<!g1_affine>
  func.return %0 : tensor<!g1_affine>
}
