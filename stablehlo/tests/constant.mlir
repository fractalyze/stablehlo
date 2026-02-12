// RUN: stablehlo-opt %s -split-input-file | FileCheck %s

func.func private @constant_with_i32(%x: tensor<15xi32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK: @constant_with_i32

// -----

!pf_babybear = !field.pf<2013265921 : i32, true>

func.func @constant_with_babybear() -> tensor<!pf_babybear> {
  %0 = stablehlo.constant dense<0> : tensor<!pf_babybear>
  return %0 : tensor<!pf_babybear>
}

// CHECK: @constant_with_babybear

// -----

!PF = !field.pf<2013265921 : i32, true>
!EF2 = !field.ef<2x!PF, 11:i32>

func.func @constant_with_ef2_scalar() -> tensor<!EF2> {
  %0 = stablehlo.constant dense<[1, 2]> : tensor<!EF2>
  return %0 : tensor<!EF2>
}

// CHECK: @constant_with_ef2_scalar

// -----

!PF = !field.pf<2013265921 : i32, true>
!EF2 = !field.ef<2x!PF, 11:i32>

func.func @constant_with_ef2_1d() -> tensor<2x!EF2> {
  %0 = stablehlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x!EF2>
  return %0 : tensor<2x!EF2>
}

// CHECK: @constant_with_ef2_1d

// -----

!PF = !field.pf<7 : i32>
!QF = !field.ef<2x!PF, 6:i32>
!TowerF6 = !field.ef<3x!QF, 2:i32>

func.func @constant_with_tower_ef6() -> tensor<!TowerF6> {
  %0 = stablehlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<!TowerF6>
  return %0 : tensor<!TowerF6>
}

// CHECK: @constant_with_tower_ef6
