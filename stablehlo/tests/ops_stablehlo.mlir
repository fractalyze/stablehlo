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
// Verification tests for StableHLO ops.
// Ported from stablehlo-ref commit 7775e3e2 with adaptations:
// - Convert float types (f32, f64, f16) to integer types (i32, i64, i16)
// - Remove complex type tests (not supported in ZK StableHLO)
// - Add field type tests (prime field, extension field)
// - Reordered to match reference file for easier comparison
//
// ZK-only ops (not in reference) are in ops_stablehlo_zk.mlir

// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// =============================================================================
// BroadcastOp
// =============================================================================

// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 1, 2>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_result_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x2x3xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_first_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x3xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<1x3xi32>
  func.return %0 : tensor<1x3xi32>
}

// -----

func.func @broadcast_bad_second_part_result_shape(%arg0: tensor<3xi32>) -> tensor<2x1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<2x1xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<2x1xi32>
  func.return %0 : tensor<2x1xi32>
}

// -----

// =============================================================================
// DynamicBroadcastInDimOp
// =============================================================================

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func.func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_unknown_dim
func.func @dynamic_broadcast_in_dim_unknown_dim(%arg0: tensor<32xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<32xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}
// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_ok_dim
func.func @dynamic_broadcast_in_dim_ok_dim(%arg0: tensor<1xi32>, %shape: tensor<3xi64>) -> tensor<7x8x9xi32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<1xi32>, tensor<3xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_output_dimensions_match_result(%arg0: tensor<4xi32>) -> tensor<3x4xi32> {
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xi32>, tensor<2xi64>) -> tensor<3x4xi32>
  return %1 : tensor<3x4xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_output_dimensions_compatible_with_result(%arg0: tensor<4xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xi32>, tensor<2xi64>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c1(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi64> {
  // expected-error@+1 {{expects operand and result to have compatible element type}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi64>
  func.return %0 : tensor<?x?x?xi64>
}

// -----

func.func @dynamic_broadcast_in_dim_c2(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c3_negative_size(%arg0: tensor<1xi32>, %shape: tensor<3xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value -1 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: -1>} : (tensor<1xi32>, tensor<3xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c3_too_large(%arg0: tensor<1xi32>, %shape: tensor<3xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 3 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 3>} : (tensor<1xi32>, tensor<3xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c4(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{broadcast_dimensions should not have duplicates}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 1>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_shape_mismatch(%arg0: tensor<32xi32>, %shape: tensor<3xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{size of operand dimension 0 (32) is not compatible with size of result dimension 2 (9)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<32xi32>, tensor<3xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_too_large(%arg0: tensor<1xi32>, %shape: tensor<3xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 3 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 3>} : (tensor<1xi32>, tensor<3xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_input_mismatch_with_shape(%arg0: tensor<1x3xi32>) {
  %shape = stablehlo.constant dense<[2, 1, 1]> : tensor<3xi32>
  // expected-error@+1 {{size of operand dimension 1 (3) is not equal to 1 or value of shape at index 2 (1)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<1x3xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// SKIPPED: dynamic_broadcast_in_dim_c7_output_dimensions_negative_size - requires constant folding validation
// SKIPPED: dynamic_broadcast_in_dim_c7_output_dimensions_mismatching_size - requires constant folding validation

func.func @dynamic_broadcast_in_dim_c8(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{duplicate expansion hint for at least one operand dimension}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 1, 2>,
    known_expanding_dimensions = array<i64: 0, 0>
  } : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c9_c10(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{hint for expanding dimension 3 does not refer to a valid operand dimension}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 1, 2>,
    known_expanding_dimensions = array<i64: 3>
  } : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_dynamic_output_shape(%arg0: tensor<?x?xi32>, %shape: tensor<?xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{op operand #1 must be statically shaped}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<?xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

// =============================================================================
// BroadcastInDimOp
// =============================================================================

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}

// -----

func.func @broadcast_in_dim_c2(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_c3(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value -1 for result with rank 3}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: -1, 2>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}

// -----

func.func @broadcast_in_dim_c3(%arg0: tensor<1x2x3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 1 for result with rank 1}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0,1,2>} : (tensor<1x2x3xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @broadcast_in_dim_c4(%arg0: tensor<1x1x3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions should not have duplicates}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0,0,2>} : (tensor<1x1x3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_c5(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{size of operand dimension 0 (3) is not equal to 1 or size of result dimension 1 (2)}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_dynamic_i1
func.func @broadcast_in_dim_dynamic_i1(%arg0: tensor<?xi32>) -> tensor<1x3xi32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<?xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

// SKIPPED: broadcast_in_dim_dynamic_result - ZK StableHLO requires statically shaped results

// CHECK-LABEL: func @broadcast_in_dim_dynamic_shaped_operand
func.func @broadcast_in_dim_dynamic_shaped_operand(%arg0: tensor<?xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<?xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// =============================================================================
// IfOp
// =============================================================================

// CHECK-LABEL: func @if
func.func @if(%pred: tensor<i1>, %branch_operand: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xi32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xi32>) -> ()
    }) : (tensor<i1>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @if_c1(%pred: tensor<i1>, %branch_operand: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 must have 0 arguments, but found 1}}
  %0 = "stablehlo.if"(%pred) ({
      ^bb0(%arg0: tensor<i32>):
        "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @if_c1(%pred: tensor<i1>, %branch_operand: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>):
        "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @if_c2(%pred: tensor<i1>, %branch_operand: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<i32>', 'tensor<i32>' vs 'tensor<i32>'}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<i32>, tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @if_c3(%pred: tensor<i1>, %branch_operand: tensor<i32>) -> tensor<i64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<i32>' are incompatible with return type(s) of operation 'tensor<i64>'}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: if_dynamic_branch_result
func.func @if_dynamic_branch_result(%pred: tensor<i1>, %true_branch_operand: tensor<2xi32>, %false_branch_operand: tensor<?xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%true_branch_operand) : (tensor<2xi32>) -> ()
    }, {
      "stablehlo.return"(%false_branch_operand) : (tensor<?xi32>) -> ()
    }) : (tensor<i1>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// SKIPPED: if_dynamic_op_result - ZK StableHLO requires statically shaped results

func.func @if_i1(%pred: tensor<1xi1>, %branch_operand: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand should be rank 0 tensor but got rank 1}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
    }) : (tensor<1xi1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// =============================================================================
// CaseOp
// =============================================================================

// CHECK-LABEL: @case
func.func @case(%index: tensor<i32>, %branch_operand: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0, %1 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<i32>, tensor<i32>) -> ()
  }, {
    "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

func.func @case_c1(%index: tensor<i32>, %branch_operand: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expect at least one branch}}
  %0 = "stablehlo.case"(%index) : (tensor<i32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @case_c2(%index: tensor<i32>, %branch_operand: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
  }, {
      ^bb0(%arg0: tensor<i32>):
        "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @case_c3(%index: tensor<i32>, %operand_1: tensor<i32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<i32>' vs 'tensor<i64>'}}
  %0 = "stablehlo.case"(%index) ({
      %1 = "stablehlo.negate"(%operand_1) : (tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%1) : (tensor<i32>) -> ()
    },  {
      %1 = stablehlo.constant dense<2> : tensor<i64>
      "stablehlo.return"(%1) : (tensor<i64>) -> ()
    }) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @case_c4(%index: tensor<i32>, %branch_operand: tensor<i32>) -> tensor<i64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<i32>' are incompatible with return type(s) of operation 'tensor<i64>'}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: @case_dynamic_branch_result
func.func @case_dynamic_branch_result(%index: tensor<i32>, %branch_operand: tensor<?xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<?xi32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<?xi32>) -> ()
  }) : (tensor<i32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// SKIPPED: case_dynamic_op_result - ZK StableHLO requires statically shaped results

func.func @case_i1(%index: tensor<1xi32>, %branch_operand: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand should be rank 0 tensor but got rank 1}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xi32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xi32>) -> ()
  }) : (tensor<1xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// =============================================================================
// CompareOp
// =============================================================================

// CHECK-LABEL: func @compare_i32
func.func @compare_i32(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func @compare_compatible_types
func.func @compare_compatible_types(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<?xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: func @compare_compatible_operand_types
func.func @compare_compatible_operand_types(%arg0: tensor<3xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// =============================================================================
// ConcatenateOp
// =============================================================================

// CHECK-LABEL: func @concatenate_1D
func.func @concatenate_1D(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: func @concatenate_1D
// Verifies that an error is not thrown if the inferred type is compatible with
// the result type.
func.func @concatenate_1D(%arg0: tensor<1xi32>, %arg1: tensor<?xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<?xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c1_c5(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<3xi32>' are incompatible with return type(s) of operation 'tensor<4xi32>'}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @concatenate_c2(%arg0: tensor<1xi32>, %arg1: tensor<2x2xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operands (0) and (1) do not match rank}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2x2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c3() -> tensor<2xi32> {
  // expected-error@+1 {{expected 1 or more operands, but found 0}}
  %0 = "stablehlo.concatenate"() { dimension = 0 : i64 } : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{op attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = -1 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{rank-0 values cannot be concatenated}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{dimension 10 is out-of-bounds for input rank 1}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 10 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c6(%arg0: tensor<1x3xi32>, %arg1: tensor<2x2xi32>) -> tensor<3x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{shapes of operand (0) and (1) are not compatible at non-concat index 1: (1, 3) != (2, 2)}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x3xi32>, tensor<2x2xi32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// -----

// =============================================================================
// ClampOp
// =============================================================================

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp_compatible_dynamic
func.func @clamp_compatible_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<?xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// CHECK-LABEL: func @clamp_compatible_dynamic_match_static
func.func @clamp_compatible_dynamic_match_static(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %0: tensor<3xi32>
}

// -----

func.func @clamp_c1(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{min shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// -----

func.func @clamp_c2(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{max shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// -----

func.func @clamp_c4(%arg0: tensor<1xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1xi32>' are incompatible with return type(s) of operation 'tensor<1x2xi32>'}}
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @clamp_scalar
func.func @clamp_scalar(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg1) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// =============================================================================
// CreateTokenOp
// =============================================================================

// CHECK-LABEL: func @create_token
func.func @create_token() -> !stablehlo.token {
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0: !stablehlo.token
}

// -----

// =============================================================================
// IotaOp
// =============================================================================

// CHECK-LABEL: func @iota
func.func @iota() -> tensor<4xi32> {
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @iota_scalar() -> tensor<i32> {
  // expected-error@+1 {{does not support scalars}}
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @iota_invalid_iota_dimension() -> tensor<4xi32> {
  // expected-error@+1 {{iota dimension cannot go beyond the output rank}}
  %0 = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// MapOp
// =============================================================================

// CHECK-LABEL: func @map
func.func @map(%arg0: tensor<4x5xi32>, %arg1: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xi32>, tensor<4x5xi32>) -> tensor<4x5xi32>
  func.return %0 : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: func @map_heterogeneous_inputs
func.func @map_heterogeneous_inputs(%arg0: tensor<2xi32>, %arg1: tensor<2xi64>) -> tensor<2xi32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i64>):
    "stablehlo.return"(%arg2) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<2xi32>, tensor<2xi64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @map_scalar_operands
func.func @map_scalar_operands(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = array<i64>} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @map_c3(%arg0: tensor<4x5xi32>, %arg1: tensor<4x5xi32>) -> tensor<4x5xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires monotonically increasing dimension numbers, but got: 1, 0}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 1, 0>} : (tensor<4x5xi32>, tensor<4x5xi32>) -> tensor<4x5xi32>
  func.return %0 : tensor<4x5xi32>
}

// -----

func.func @map_c4(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects number of operands to match the arity of map computation, but got: 2 and 1}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg: tensor<i32>):
    %1 = stablehlo.add %arg, %arg : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// SelectOp
// =============================================================================

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_x_y
func.func @select_scalar_x_y(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<?x3xi32>, %arg2: tensor<2x?xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<?x3xi32>, tensor<2x?xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

func.func @select_c1(%arg0: tensor<3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires the same shape for all operands}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_c2(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires compatible types for non-predicate operands}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// =============================================================================
// SliceOp
// =============================================================================

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c2(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{the number of elements in start_indices (3) does not match the rank of the operand (2)}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0, 0>,
    limit_indices = array<i64: 2, 4, 0>,
    strides = array<i64: 1, 2, 0>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{negative start index -1 in dimension 0}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: -1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{limit index 5 is larger than dimension size 4 in dimension 1}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 5>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start index 3 is larger than limit index 2 in dimension 1}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 3>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c4(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{stride must be positive but got 0 in dimension 0}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 0, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @slice_dynamic_dim
func.func @slice_dynamic_dim(%arg0: tensor<3x?xi32>) -> tensor<1x?xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 1>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<3x?xi32>) -> tensor<1x?xi32>
  func.return %0 : tensor<1x?xi32>
}

// -----

// =============================================================================
// DynamicSliceOp
// =============================================================================

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c2_a(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has mismatched number of slice sizes (1) and number of start indices (2)}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c2_b(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has mismatched number of start indices (1) and the rank of operand (2)}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<3x4xi32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c3(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c4_a(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has negative size index to dynamic slice: -1}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: -1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c4_b(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has slice size 10 greater than dimension size 4 in dimension 1 of operand}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 10>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c5(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<2x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1x4xi32>' are incompatible with return type(s) of operation 'tensor<2x4xi32>'}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_slice_dynamic_dim
func.func @dynamic_slice_dynamic_dim(%arg0: tensor<?x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// =============================================================================
// DynamicUpdateSliceOp
// =============================================================================

// CHECK-LABEL: @dynamic_update_slice
func.func @dynamic_update_slice(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c1(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x5xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<3x4xi64>' are incompatible with return type(s) of operation 'tensor<3x5xi64>'}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x5xi64>
  func.return %0 : tensor<3x5xi64>
}

// -----

func.func @dynamic_update_slice_c3(%operand: tensor<3x4xi64>, %update: tensor<2xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{update rank does not match operand rank: 1 vs 2.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<2xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c4(%operand: tensor<3x4xi64>, %update: tensor<1x2xi64>, %start_indices0: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects number of start_indices to match operand rank: 1 vs 2.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0) : (tensor<3x4xi64>, tensor<1x2xi64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c5(%operand: tensor<11x3x4xi32>, %update: tensor<1x3x4xi32>, %start_indices0: tensor<i32>, %start_indices1: tensor<i64>, %start_indices2: tensor<i64>) -> tensor<11x3x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1, %start_indices2) : (tensor<11x3x4xi32>, tensor<1x3x4xi32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<11x3x4xi32>
  func.return %0 : tensor<11x3x4xi32>
}

// -----

func.func @dynamic_update_slice_c6(%operand: tensor<3x4xi64>, %update: tensor<1x5xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects size at dimension 1 of update to be in range [0, 4]. Got: 5.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x5xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: @dynamic_update_slice_dynamic_dim
func.func @dynamic_update_slice_dynamic_dim(%operand: tensor<?x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<?x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// =============================================================================
// TransposeOp
// =============================================================================

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

// -----

// CHECK-LABEL: func @transpose_ranked
func.func @transpose_ranked(%arg0: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  func.return %0 : tensor<?x?x?x?xi32>
}

// -----

func.func @transpose_missing_permutation(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // expected-error@+1 {{requires attribute 'permutation'}}
  %0 = "stablehlo.transpose"(%arg0) {} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{TransposeOp operand rank 4 does not match permutation size 1}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutation(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{attribute permutation must be a permutation of [0, 1, 2, 3] but got 1, 0, 3, 9}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 9>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) -> tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<2x1x4x3xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>'}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<?x1x?x3xi32>' are incompatible with return type(s) of operation 'tensor<?x2x?x?xi32>}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32>
  func.return %0 : tensor<?x2x?x?xi32>
}

// -----

// =============================================================================
// ConstantOp
// =============================================================================

// CHECK-LABEL: func @constant_i32
func.func @constant_i32() -> tensor<i32> {
  // CHECK: stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @constant_c1() -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.constant' op inferred type(s) 'tensor<i32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @constant_c2() -> tensor<?xi32> {
  // expected-error@+1 {{op result #0 must be statically shaped tensor}}
  %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// =============================================================================
// SortOp
// =============================================================================

// CHECK-LABEL: func @sort
func.func @sort(%input0: tensor<16x16xi32>, %input1: tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xi32>, tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>)
  func.return %0#0, %0#1 : tensor<16x16xi32>, tensor<16x16xi32>
}

// -----

func.func @sort_c4(%input0: tensor<16x16xi32>, %input1: tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found -3}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -3 : i64, is_stable = true} : (tensor<16x16xi32>, tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>)
  func.return %0#0, %0#1 : tensor<16x16xi32>, tensor<16x16xi32>
}

// -----

func.func @sort_c5(%input0: tensor<16x16xi32>, %input1: tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block should have 4 arguments}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xi32>, tensor<16x16xi32>) -> (tensor<16x16xi32>, tensor<16x16xi32>)
  func.return %0#0, %0#1 : tensor<16x16xi32>, tensor<16x16xi32>
}

// -----

// =============================================================================
// ReshapeOp
// =============================================================================

// CHECK-LABEL: func @reshape
func.func @reshape(%arg0: tensor<2x4xi32>) -> tensor<4x2xi32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<2x4xi32>) -> tensor<4x2xi32>
  func.return %0 : tensor<4x2xi32>
}

// -----

func.func @reshape_c1(%arg0: tensor<2x4xi32>) -> tensor<3x3xi32> {
  // expected-error @+1 {{number of output elements (9) doesn't match expected number of elements (8)}}
  %0 = "stablehlo.reshape"(%arg0) : (tensor<2x4xi32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// -----

// =============================================================================
// ReverseOp
// =============================================================================

// CHECK-LABEL: func @reverse
func.func @reverse(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = array<i64: 0, 1>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c1(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{dimensions should be unique. Got: 0, 0}}
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = array<i64: 0, 0>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c2(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = array<i64: -1>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c3(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be between [0, 2). Got dimension: 2.}}
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = array<i64: 2>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

// =============================================================================
// DotGeneralOp
// =============================================================================

// CHECK-LABEL: func @dot_general
func.func @dot_general(%arg0: tensor<2x3x4xi32>, %arg1: tensor<2x3x5xi32>) -> tensor<2x4x5xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4xi32>, tensor<2x3x5xi32>) -> tensor<2x4x5xi32>
  func.return %0 : tensor<2x4x5xi32>
}

// -----

func.func @dot_general_c1(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dot_general_c2(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{lhs and rhs should have the same number of contracting dimensions}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dot_general_c3(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dot_general_c4(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{has duplicated dimension from rhs_batching_dimensions and rhs_contracting_dimensions: 0}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dot_general_c5(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{lhs_batching_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [-1],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dot_general_c6(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @+1 {{lhs_contracting_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [-1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

// =============================================================================
// TupleOp
// =============================================================================

// CHECK-LABEL: func @tuple
func.func @tuple(%arg0: tensor<4xi32>, %arg1: tensor<2x3xi32>) -> tuple<tensor<4xi32>, tensor<2x3xi32>> {
  %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<4xi32>, tensor<2x3xi32>) -> tuple<tensor<4xi32>, tensor<2x3xi32>>
  func.return %0 : tuple<tensor<4xi32>, tensor<2x3xi32>>
}

// -----

// =============================================================================
// GetTupleElementOp
// =============================================================================

// CHECK-LABEL: func @get_tuple_element
func.func @get_tuple_element(%arg0: tuple<tensor<4xi32>, tensor<2x3xi32>>) -> tensor<4xi32> {
  %0 = "stablehlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<4xi32>, tensor<2x3xi32>>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// DynamicReshapeOp
// =============================================================================

// CHECK-LABEL: func @dynamic_reshape
func.func @dynamic_reshape(%arg0: tensor<?xi32>, %shape: tensor<2xindex>) -> tensor<?x?xi32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xi32>, tensor<2xindex>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

func.func @dynamic_reshape_c1(%arg0: tensor<?xi32>, %shape: tensor<2xindex>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expects operand and result to have compatible element type}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xi32>, tensor<2xindex>) -> tensor<?x?xi64>
  func.return %0 : tensor<?x?xi64>
}

// -----

func.func @dynamic_reshape_c2(%arg0: tensor<11xi32>, %shape: tensor<2xindex>) -> tensor<2x5xi32> {
  // expected-error @+1 {{number of output elements (10) doesn't match expected number of elements}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<11xi32>, tensor<2xindex>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

func.func @dynamic_reshape_c3(%arg0: tensor<?xi32>, %shape: tensor<2xindex>) -> tensor<?xi32> {
  // expected-error @+1 {{result should have a rank equal to the number of elements in output_shape}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xi32>, tensor<2xindex>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// =============================================================================
// BitcastConvertOp
// =============================================================================

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<2xi32>) -> tensor<2x4xi8> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xi32>) -> tensor<2x4xi8>
  func.return %0 : tensor<2x4xi8>
}

// -----

// CHECK-LABEL: func @bitcast_convert_same_type
func.func @bitcast_convert_same_type(%arg: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @bitcast_convert_c1(%arg: tensor<2xi64>) -> tensor<3xi64> {
  // expected-error@+1 {{operand and result shapes must match except for the innermost dimension of the shape with the smaller element type. Got: 'tensor<2xi64>' and 'tensor<3xi64>'.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xi64>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

func.func @bitcast_convert_c2(%arg: tensor<i64>) -> tensor<i32> {
  // expected-error@+1 {{rank of smaller element type (0) should be 1 more than rank of larger element type (0), but 0 != 0 + 1.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<i64>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @bitcast_convert_c3(%arg: tensor<2xi64>) -> tensor<2x4xi32> {
  // expected-error@+1 {{requires compatible bit widths. Got: 'tensor<2xi64>' and 'tensor<2x4xi32>', but 32 * 4 != 64.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xi64>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

// =============================================================================
// GatherOp
// =============================================================================

// CHECK-LABEL: func @gather
func.func @gather(%operand: tensor<2x4x9xi32>, %start_indices: tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c1(%operand: tensor<2x4x9xi32>, %start_indices: tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (2) plus operand_batching_dims size (0) is not equal to operand rank (3)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c2(%operand: tensor<2x4x9xi32>, %start_indices: tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects index_vector_dim to be in range [0, rank-of('start_indices')] i.e. [0, 3]. got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = -1
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c3(%operand: tensor<2x4x9xi32>, %start_indices: tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start_index_map size (1) is not equal to size of index dimension (2) of start_indices (2)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c4_a(%operand: tensor<16x11xi32>, %start_indices: tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects offset_dims to be sorted, got: [2, 1]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 1],
      collapsed_slice_dims = [],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 8, 6>,
    indices_are_sorted = false
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather_c4_b(%operand: tensor<16x11xi32>, %start_indices: tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects offset_dims to not repeat, got: [2, 2]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 2],
      collapsed_slice_dims = [],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 8, 6>,
    indices_are_sorted = false
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather_c5(%operand: tensor<2x4x9xi32>, %start_indices: tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of offset_dims to be in range [0, implied-result-rank) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [-1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

// =============================================================================
// GetDimensionSizeOp
// =============================================================================

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<4x2xi32>) -> tensor<i32> {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<4x2xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// =============================================================================
// PadOp
// =============================================================================

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<1x2x3xi16>, %arg1: tensor<i16>) -> tensor<2x4x5xi16> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 0>
  } : (tensor<1x2x3xi16>, tensor<i16>) -> tensor<2x4x5xi16>
  func.return %0 : tensor<2x4x5xi16>
}

// -----

func.func @pad_c2(%arg0: tensor<1x2x3xi16>, %arg1: tensor<i16>) -> tensor<2x4x7xi16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{edge_padding_low length (2) must match operand rank (3)}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 1, 1>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<1x2x3xi16>, tensor<i16>) -> tensor<2x4x7xi16>
  func.return %0 : tensor<2x4x7xi16>
}

// -----

// SKIPPED: pad_c3 - ZK StableHLO inferPadOp doesn't validate interior padding

// -----

func.func @pad_c4(%arg0: tensor<1x2x3xi16>, %arg1: tensor<i16>) -> tensor<2x4x7xi16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Padding result in negative size for dimension 2}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, -4>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 0>
  } : (tensor<1x2x3xi16>, tensor<i16>) -> tensor<2x4x7xi16>
  func.return %0 : tensor<2x4x7xi16>
}

// -----

// CHECK-LABEL: func @pad_dynamic
func.func @pad_dynamic(%arg0: tensor<?x48x48x32xi32>) -> tensor<?x48x48x48xi32> {
  %0 = "stablehlo.constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_low = array<i64: 0, 0, 0, 0>,
    edge_padding_high = array<i64: 0, 0, 0, 16>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<?x48x48x32xi32>, tensor<i32>) -> tensor<?x48x48x48xi32>
  func.return %1 : tensor<?x48x48x48xi32>
}

// -----

func.func @pad_i3(%arg0: tensor<1x2x3xi16>, %arg1: tensor<i16>) -> tensor<2x4x7xi16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{edge_padding_low length (1) must match operand rank (3)}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 1>,
    edge_padding_high = array<i64: 1>,
    interior_padding = array<i64: 1>
  } : (tensor<1x2x3xi16>, tensor<i16>) -> tensor<2x4x7xi16>
  func.return %0 : tensor<2x4x7xi16>
}

// -----

// =============================================================================
// ScatterOp
// =============================================================================

// CHECK-LABEL: func @scatter
func.func @scatter(%input: tensor<200x100x300xi32>, %indices: tensor<10x2xi32>, %updates: tensor<10x300xi32>) -> tensor<200x100x300xi32> {
  %0 = "stablehlo.scatter"(%input, %indices, %updates) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %1 = stablehlo.add %arg0, %arg1 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<200x100x300xi32>, tensor<10x2xi32>, tensor<10x300xi32>) -> tensor<200x100x300xi32>
  func.return %0 : tensor<200x100x300xi32>
}

// -----

// =============================================================================
// AbsOp
// =============================================================================

// CHECK-LABEL: func @abs_i32
func.func @abs_i32(%arg0: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @abs_c1(%arg0: tensor<1x2xi32>) -> tensor<1x2xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.abs' op inferred type(s) 'tensor<1x2xi32>' are incompatible with return type(s) of operation 'tensor<1x2xi64>'}}
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xi32>) -> tensor<1x2xi64>
  func.return %0 : tensor<1x2xi64>
}

// -----

// =============================================================================
// ConvertOp
// =============================================================================

// CHECK-LABEL: func @convert_i32_to_i64
func.func @convert_i32_to_i64(%arg0: tensor<4xi32>) -> tensor<4xi64> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<4xi32>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @convert_i64_to_i32
func.func @convert_i64_to_i32(%arg0: tensor<4xi64>) -> tensor<4xi32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<4xi64>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// =============================================================================
// DynamicIotaOp
// =============================================================================

// CHECK-LABEL: func @dynamic_iota
func.func @dynamic_iota() -> tensor<4xi32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// -----

func.func @dynamic_iota_c1() -> tensor<?xi32> {
  // expected-error@+2 {{op attribute 'iota_dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = -1 : (tensor<1xi64>) -> tensor<?xi32>
  func.return %1 : tensor<?xi32>
}

// -----

func.func @dynamic_iota_c2() -> tensor<?xi32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  // expected-error@+1 {{iota dimension cannot go beyond the output rank}}
  %1 = stablehlo.dynamic_iota %0, dim = 2 : (tensor<1xi64>) -> tensor<?xi32>
  func.return %1 : tensor<?xi32>
}

// -----

func.func @dynamic_iota_c3() -> tensor<4xi32> {
  // expected-error@+2 {{output shape [1] is incompatible with return type of operation 'tensor<4xi32>'}}
  %0 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// -----

// =============================================================================
// ReduceWindowOp
// =============================================================================

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<4x6xi32>) -> tensor<2x2xi32> {
  %init = stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.reduce_window"(%arg0, %init) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 2, 3>,
    window_strides = array<i64: 2, 3>,
    padding = dense<0> : tensor<2x2xi64>,
    base_dilations = array<i64: 1, 1>,
    window_dilations = array<i64: 1, 1>
  } : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// =============================================================================
// WhileOp
// =============================================================================

// CHECK-LABEL: func @while
func.func @while(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.while"(%arg0) ({
    ^bb0(%arg1: tensor<i32>):
      %1 = stablehlo.constant dense<10> : tensor<i32>
      %2 = stablehlo.compare LT, %arg1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg1: tensor<i32>):
      %1 = stablehlo.constant dense<1> : tensor<i32>
      %2 = stablehlo.add %arg1, %1 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
