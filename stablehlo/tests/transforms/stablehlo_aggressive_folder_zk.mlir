// RUN: stablehlo-opt --stablehlo-aggressive-folder=fold-op-element-limit=100 --split-input-file --verify-diagnostics %s | FileCheck %s

// Adjacent-splat concatenate folding over field / elliptic-curve element types.
// The matched splat's value attr is storage-typed (an i256/i32 integer tensor),
// so the merged constant must be built with the operand's field/EC *result*
// type, not the storage type. Otherwise it lowers to a storage-typed constant
// feeding a field-typed concatenate, which fails shape inference downstream
// ("Cannot concatenate arrays with different element types: PALLAS_SF vs S256").

!pf7 = !field.pf<7 : i32>

// CHECK-LABEL: func.func @fold_concatenate_splat_leading_field
func.func @fold_concatenate_splat_leading_field(%arg0: tensor<1x!pf7>) -> tensor<3x!pf7> {
  // The two leading splats merge into one constant whose value attr stays
  // storage-typed (tensor<2xi32>) but whose result type is the field type.
  // CHECK:      %[[CST:.+]] = stablehlo.constant() <{value = dense<1> : tensor<2xi32>}> : () -> tensor<2x!{{.+}}>
  // CHECK-NEXT: stablehlo.concatenate %[[CST]], %arg0, dim = 0
  %cst0 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1x!pf7>
  %0 = stablehlo.concatenate %cst0, %cst0, %arg0, dim = 0 : (tensor<1x!pf7>, tensor<1x!pf7>, tensor<1x!pf7>) -> tensor<3x!pf7>
  return %0 : tensor<3x!pf7>
}

// -----

!pf7 = !field.pf<7 : i32>

// CHECK-LABEL: func.func @fold_concatenate_splat_middle_field
func.func @fold_concatenate_splat_middle_field(%arg0: tensor<1x!pf7>) -> tensor<4x!pf7> {
  // CHECK:      %[[CST:.+]] = stablehlo.constant() <{value = dense<1> : tensor<2xi32>}> : () -> tensor<2x!{{.+}}>
  // CHECK-NEXT: stablehlo.concatenate %arg0, %[[CST]], %arg0, dim = 0
  %cst0 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1x!pf7>
  %0 = stablehlo.concatenate %arg0, %cst0, %cst0, %arg0, dim = 0 : (tensor<1x!pf7>, tensor<1x!pf7>, tensor<1x!pf7>, tensor<1x!pf7>) -> tensor<4x!pf7>
  return %0 : tensor<4x!pf7>
}

// -----

#curve = #elliptic_curve.sw<0 : i256, 3 : i256, (1 : i256, 2 : i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583 : i256>
!ec = !elliptic_curve.affine<#curve>

// CHECK-LABEL: func.func @fold_concatenate_splat_leading_ec
func.func @fold_concatenate_splat_leading_ec(%arg0: tensor<1x!ec>) -> tensor<3x!ec> {
  // EC-point constants encode coordinates as trailing storage dims, so the
  // value attr resizes along the leading (concatenation) dim only — here
  // tensor<1x2xi256> -> tensor<2x2xi256> — while the result stays the EC type.
  // CHECK:      %[[CST:.+]] = stablehlo.constant() <{value = dense<1> : tensor<2x2xi256>}> : () -> tensor<2x!{{.+}}>
  // CHECK-NEXT: stablehlo.concatenate %[[CST]], %arg0, dim = 0
  %cst0 = "stablehlo.constant"() <{value = dense<1> : tensor<1x2xi256>}> : () -> tensor<1x!ec>
  %0 = stablehlo.concatenate %cst0, %cst0, %arg0, dim = 0 : (tensor<1x!ec>, tensor<1x!ec>, tensor<1x!ec>) -> tensor<3x!ec>
  return %0 : tensor<3x!ec>
}
