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

!PF = !field.pf<7:i32>
!Fp2 = !field.ef<2x!PF, 6:i32>
!TowerF6 = !field.ef<3x!Fp2, 2:i32>

func.func @constant_with_tower_ext_field() -> tensor<!TowerF6> {
  %0 = stablehlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<!TowerF6>
  return %0 : tensor<!TowerF6>
}

// CHECK: @constant_with_tower_ext_field

// -----

// EC point constant — BN254 G1 affine (scalar)
!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_affine = !elliptic_curve.affine<#g1_curve>

func.func @constant_with_g1_affine() -> tensor<!g1_affine> {
  %0 = "stablehlo.constant"() <{value = dense<[1, 2]> : tensor<2xi256>}> : () -> tensor<!g1_affine>
  return %0 : tensor<!g1_affine>
}

// CHECK: @constant_with_g1_affine

// -----

// EC point constant — BN254 G1 affine (1D tensor)
!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_affine = !elliptic_curve.affine<#g1_curve>

func.func @constant_with_g1_affine_1d() -> tensor<2x!g1_affine> {
  %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi256>}> : () -> tensor<2x!g1_affine>
  return %0 : tensor<2x!g1_affine>
}

// CHECK: @constant_with_g1_affine_1d

// -----

// EC point constant — BN254 G1 jacobian (scalar)
!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_jacobian = !elliptic_curve.jacobian<#g1_curve>

func.func @constant_with_g1_jacobian() -> tensor<!g1_jacobian> {
  %0 = "stablehlo.constant"() <{value = dense<[1, 2, 1]> : tensor<3xi256>}> : () -> tensor<!g1_jacobian>
  return %0 : tensor<!g1_jacobian>
}

// CHECK: @constant_with_g1_jacobian

// -----

// EC point constant — BN254 G1 xyzz (scalar)
!PF_G1 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#g1_curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF_G1
!g1_xyzz = !elliptic_curve.xyzz<#g1_curve>

func.func @constant_with_g1_xyzz() -> tensor<!g1_xyzz> {
  %0 = "stablehlo.constant"() <{value = dense<[1, 2, 1, 1]> : tensor<4xi256>}> : () -> tensor<!g1_xyzz>
  return %0 : tensor<!g1_xyzz>
}

// CHECK: @constant_with_g1_xyzz

// -----

// EC point constant — BN254 G2 affine (scalar, Fp² base field)
!PF_G2 = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!QF = !field.ef<2x!PF_G2, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

#g2a = dense<[0, 0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2_curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
!g2_affine = !elliptic_curve.affine<#g2_curve>

func.func @constant_with_g2_affine() -> tensor<!g2_affine> {
  %0 = "stablehlo.constant"() <{value = dense<[[1, 0], [2, 0]]> : tensor<2x2xi256>}> : () -> tensor<!g2_affine>
  return %0 : tensor<!g2_affine>
}

// CHECK: @constant_with_g2_affine
