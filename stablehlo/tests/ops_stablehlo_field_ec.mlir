// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Tier C.5 — prime-ir field/EC element types are admitted by stablehlo
// op verifiers alongside FP/int/complex (float coexistence). Widening is
// done in-place by adding HLO_Field/HLO_EC to the element-type lists in
// the existing tensor unions in Base.td, so every consumer gets it
// without per-op .td edits.

// -----
// Arithmetic over prime fields.

// CHECK-LABEL: func @field_add
// CHECK: stablehlo.add
func.func @field_add(%a: tensor<4x!field.pf<7681:i32>>,
                     %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.add %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_subtract
// CHECK: stablehlo.subtract
func.func @field_subtract(%a: tensor<4x!field.pf<7681:i32>>,
                          %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.subtract %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_multiply
// CHECK: stablehlo.multiply
func.func @field_multiply(%a: tensor<4x!field.pf<7681:i32>>,
                          %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.multiply %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_divide
// CHECK: stablehlo.divide
func.func @field_divide(%a: tensor<4x!field.pf<7681:i32>>,
                        %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.divide %a, %b : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_negate
// CHECK: stablehlo.negate
func.func @field_negate(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.negate %a : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----
// Extension fields (F_p^2) flow through the same widened constraints.

// CHECK-LABEL: func @field_extension_add
// CHECK: stablehlo.add
func.func @field_extension_add(%a: tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>,
                               %b: tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>)
    -> tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>> {
  %0 = stablehlo.add %a, %b : tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>
  func.return %0 : tensor<2x!field.ef<2x!field.pf<7:i32>, 6:i32>>
}

// -----
// Float ops alongside field ops in the same module — coexistence policy.

// CHECK-LABEL: func @field_pf_with_float_coexistence
// CHECK: stablehlo.add{{.*}}tensor<4xf32>
// CHECK: stablehlo.add
func.func @field_pf_with_float_coexistence(%fa: tensor<4xf32>, %fb: tensor<4xf32>,
                                           %ga: tensor<4x!field.pf<7681:i32>>,
                                           %gb: tensor<4x!field.pf<7681:i32>>)
    -> (tensor<4xf32>, tensor<4x!field.pf<7681:i32>>) {
  %0 = stablehlo.add %fa, %fb : tensor<4xf32>
  %1 = stablehlo.add %ga, %gb : tensor<4x!field.pf<7681:i32>>
  func.return %0, %1 : tensor<4xf32>, tensor<4x!field.pf<7681:i32>>
}

// -----
// Structural ops.

// CHECK-LABEL: func @field_reshape
// CHECK: stablehlo.reshape
func.func @field_reshape(%a: tensor<2x4x!field.pf<7681:i32>>) -> tensor<8x!field.pf<7681:i32>> {
  %0 = stablehlo.reshape %a : (tensor<2x4x!field.pf<7681:i32>>) -> tensor<8x!field.pf<7681:i32>>
  func.return %0 : tensor<8x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_transpose
// CHECK: stablehlo.transpose
func.func @field_transpose(%a: tensor<3x2x!field.pf<7681:i32>>) -> tensor<2x3x!field.pf<7681:i32>> {
  %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<3x2x!field.pf<7681:i32>>) -> tensor<2x3x!field.pf<7681:i32>>
  func.return %0 : tensor<2x3x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_broadcast_in_dim
// CHECK: stablehlo.broadcast_in_dim
func.func @field_broadcast_in_dim(%a: tensor<3x!field.pf<7681:i32>>) -> tensor<2x3x!field.pf<7681:i32>> {
  %0 = stablehlo.broadcast_in_dim %a, dims = [1] : (tensor<3x!field.pf<7681:i32>>) -> tensor<2x3x!field.pf<7681:i32>>
  func.return %0 : tensor<2x3x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_concatenate
// CHECK: stablehlo.concatenate
func.func @field_concatenate(%a: tensor<2x!field.pf<7681:i32>>,
                             %b: tensor<3x!field.pf<7681:i32>>)
    -> tensor<5x!field.pf<7681:i32>> {
  %0 = stablehlo.concatenate %a, %b, dim = 0
      : (tensor<2x!field.pf<7681:i32>>, tensor<3x!field.pf<7681:i32>>)
      -> tensor<5x!field.pf<7681:i32>>
  func.return %0 : tensor<5x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_slice
// CHECK: stablehlo.slice
func.func @field_slice(%a: tensor<8x!field.pf<7681:i32>>) -> tensor<3x!field.pf<7681:i32>> {
  %0 = stablehlo.slice %a [2:5] : (tensor<8x!field.pf<7681:i32>>) -> tensor<3x!field.pf<7681:i32>>
  func.return %0 : tensor<3x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_select
// CHECK: stablehlo.select
func.func @field_select(%pred: tensor<4xi1>,
                        %a: tensor<4x!field.pf<7681:i32>>,
                        %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.select %pred, %a, %b : tensor<4xi1>, tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_iota
// CHECK: stablehlo.iota
func.func @field_iota() -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.iota dim = 0 : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_pad
// CHECK: stablehlo.pad
func.func @field_pad(%a: tensor<2x3x!field.pf<7681:i32>>,
                     %pad: tensor<!field.pf<7681:i32>>)
    -> tensor<5x9x!field.pf<7681:i32>> {
  %0 = stablehlo.pad %a, %pad,
       low = [0, 1], high = [2, 1], interior = [1, 2]
       : (tensor<2x3x!field.pf<7681:i32>>, tensor<!field.pf<7681:i32>>)
       -> tensor<5x9x!field.pf<7681:i32>>
  func.return %0 : tensor<5x9x!field.pf<7681:i32>>
}

// -----
// Control flow carries field tensors through regions.

// CHECK-LABEL: func @field_if
// CHECK: stablehlo.if
func.func @field_if(%pred: tensor<i1>,
                    %a: tensor<4x!field.pf<7681:i32>>,
                    %b: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = "stablehlo.if"(%pred) ({
    "stablehlo.return"(%a) : (tensor<4x!field.pf<7681:i32>>) -> ()
  }, {
    "stablehlo.return"(%b) : (tensor<4x!field.pf<7681:i32>>) -> ()
  }) : (tensor<i1>) -> tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_while
// CHECK: stablehlo.while
func.func @field_while(%init: tensor<4x!field.pf<7681:i32>>,
                       %cnt: tensor<i64>, %lim: tensor<i64>)
    -> tensor<4x!field.pf<7681:i32>> {
  %r:2 = stablehlo.while(%i = %cnt, %x = %init)
      : tensor<i64>, tensor<4x!field.pf<7681:i32>>
  cond {
    %c = stablehlo.compare LT, %i, %lim : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %nx = stablehlo.add %x, %x : tensor<4x!field.pf<7681:i32>>
    %ni = stablehlo.add %i, %i : tensor<i64>
    stablehlo.return %ni, %nx : tensor<i64>, tensor<4x!field.pf<7681:i32>>
  }
  func.return %r#1 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_optimization_barrier
// CHECK: stablehlo.optimization_barrier
func.func @field_optimization_barrier(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.optimization_barrier %a : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_ntt_forward
// CHECK: stablehlo.ntt %{{.*}}, type = NTT, length = 4
func.func @field_ntt_forward(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.ntt %a, type = NTT, length = 4
      : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// CHECK-LABEL: func @field_ntt_inverse
// CHECK: stablehlo.ntt %{{.*}}, type = INTT, length = 4
func.func @field_ntt_inverse(%a: tensor<4x!field.pf<7681:i32>>)
    -> tensor<4x!field.pf<7681:i32>> {
  %0 = stablehlo.ntt %a, type = INTT, length = 4
      : tensor<4x!field.pf<7681:i32>>
  func.return %0 : tensor<4x!field.pf<7681:i32>>
}

// -----

// Operand element type is constrained at parse time to a field tensor.

func.func @ntt_non_field_rejected(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{operand #0 must be ranked tensor of}}
  %0 = "stablehlo.ntt"(%a) <{ntt_type = #stablehlo<ntt_type NTT>, ntt_length = 4 : i64}>
      : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// stablehlo.pairing_check happy path. BN254 G1xG1 is semantically wrong
// (real pairing wants G1xG2) but exercises the IR machinery; G1xG2
// typing locks down once G2 PrimitiveTypes land.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @pairing_check
// CHECK: stablehlo.pairing_check
func.func @pairing_check(%g1: tensor<4x!jac>, %g2: tensor<4x!jac>)
    -> tensor<i1> {
  %0 = stablehlo.pairing_check %g1, %g2
      : (tensor<4x!jac>, tensor<4x!jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----
// Verifier rejects: rank mismatch — pairing_check requires rank-1.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

func.func @pairing_check_rank_rejected(%g1: tensor<2x4x!jac>,
                                       %g2: tensor<2x4x!jac>)
    -> tensor<i1> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires rank-1 operands}}
  %0 = stablehlo.pairing_check %g1, %g2
      : (tensor<2x4x!jac>, tensor<2x4x!jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----
// Verifier rejects: length mismatch between the two operand tensors.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

func.func @pairing_check_length_rejected(%g1: tensor<4x!jac>,
                                         %g2: tensor<8x!jac>)
    -> tensor<i1> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires matching operand lengths}}
  %0 = stablehlo.pairing_check %g1, %g2
      : (tensor<4x!jac>, tensor<8x!jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----
// Verifier rejects: non-EC operand element type. Uses the generic op
// form because the assembly format would type-check the operand against
// HLO_ECTensor at parse time before the verifier ever runs.

func.func @pairing_check_non_ec_rejected(%g1: tensor<4xf32>, %g2: tensor<4xf32>)
    -> tensor<i1> {
  // expected-error@+1 {{operand #0 must be ranked tensor of}}
  %0 = "stablehlo.pairing_check"(%g1, %g2)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----
// stablehlo.msm happy path: rank-1 scalars × rank-1 bases reduces to a
// scalar EC point.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @msm_single
// CHECK: stablehlo.msm
func.func @msm_single(%scalars: tensor<4x!field.pf<7681:i32>>,
                      %bases: tensor<4x!jac>) -> tensor<!jac> {
  %0 = stablehlo.msm %scalars, %bases
      : (tensor<4x!field.pf<7681:i32>>, tensor<4x!jac>) -> tensor<!jac>
  func.return %0 : tensor<!jac>
}

// -----
// stablehlo.msm batched: batch_size=2 over an 8-element operand returns
// a 2-element tensor of points.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

// CHECK-LABEL: func @msm_batched
// CHECK: stablehlo.msm
func.func @msm_batched(%scalars: tensor<8x!field.pf<7681:i32>>,
                       %bases: tensor<8x!jac>) -> tensor<2x!jac> {
  %0 = "stablehlo.msm"(%scalars, %bases) <{batch_size = 2 : i32}>
      : (tensor<8x!field.pf<7681:i32>>, tensor<8x!jac>) -> tensor<2x!jac>
  func.return %0 : tensor<2x!jac>
}

// -----
// Verifier rejects: length mismatch between scalars and bases.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

func.func @msm_length_rejected(%scalars: tensor<4x!field.pf<7681:i32>>,
                               %bases: tensor<8x!jac>) -> tensor<!jac> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires matching operand lengths}}
  %0 = stablehlo.msm %scalars, %bases
      : (tensor<4x!field.pf<7681:i32>>, tensor<8x!jac>) -> tensor<!jac>
  func.return %0 : tensor<!jac>
}

// -----
// Verifier rejects: batch_size that doesn't divide the operand length.

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!jac = !elliptic_curve.jacobian<#curve>

func.func @msm_uneven_batch_rejected(%scalars: tensor<7x!field.pf<7681:i32>>,
                                     %bases: tensor<7x!jac>) -> tensor<2x!jac> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{batch_size 2 must divide the operand length 7}}
  %0 = "stablehlo.msm"(%scalars, %bases) <{batch_size = 2 : i32}>
      : (tensor<7x!field.pf<7681:i32>>, tensor<7x!jac>) -> tensor<2x!jac>
  func.return %0 : tensor<2x!jac>
}

// -----
// Verifier rejects: non-field scalar operand. Uses generic op form
// because the assembly format would type-check at parse time.

func.func @msm_non_field_scalar_rejected(%scalars: tensor<4xf32>,
                                          %bases: tensor<4xf32>) -> tensor<f32> {
  // expected-error@+1 {{operand #0 must be ranked tensor of}}
  %0 = "stablehlo.msm"(%scalars, %bases)
      : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
