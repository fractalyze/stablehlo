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
