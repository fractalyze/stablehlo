// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version='target=1.12.0' --verify-diagnostics --split-input-file %s

// Fork-added VHLO ops/types/enums all floor at minVersion 1.13.0, so any
// target below that floor (here 1.12.0) must refuse to downgrade. One
// artifact per split chunk: the ZK ops fail on their own op, the field/EC
// element types and extended-width integers fail on the carrying vhlo.func_v1
// because the type itself is illegal below the floor. Type/attr aliases are
// file-scoped (cannot live inside the module), so each chunk declares the
// curve it needs above the wrapping module.

// stablehlo.ntt (+ #vhlo<ntt_type_v1 ...> enum attr) — vhlo.ntt_v1 floor 1.13.0.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @ntt(%arg0: tensor<8x!field.pf<2130706433 : i32, true>>) -> tensor<8x!field.pf<2130706433 : i32, true>> {
  %0 = stablehlo.ntt %arg0, type = NTT, length = 8 : tensor<8x!field.pf<2130706433 : i32, true>>
  func.return %0 : tensor<8x!field.pf<2130706433 : i32, true>>
}
}

// -----

// stablehlo.msm — vhlo.msm_v1 floor 1.13.0.
!BN254_Fr = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#bn254_g1 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!bn254_affine = !elliptic_curve.affine<#bn254_g1>
!bn254_xyzz = !elliptic_curve.xyzz<#bn254_g1>
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @msm(%scalars: tensor<2048x!BN254_Fr>, %bases: tensor<1024x!bn254_affine>) -> tensor<2x!bn254_xyzz> {
  %0 = stablehlo.msm %scalars, %bases {window_bits = 16 : i32, precompute_factor = 2 : i32, bitsize = 253 : i32, batch_size = 2 : i32, are_points_shared = true} : (tensor<2048x!BN254_Fr>, tensor<1024x!bn254_affine>) -> tensor<2x!bn254_xyzz>
  func.return %0 : tensor<2x!bn254_xyzz>
}
}

// -----

// stablehlo.pairing_check — vhlo.pairing_check_v1 floor 1.13.0.
#bn254_pc = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!bn254_jac = !elliptic_curve.jacobian<#bn254_pc>
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @pairing_check(%g1: tensor<4x!bn254_jac>, %g2: tensor<4x!bn254_jac>) -> tensor<i1> {
  %0 = stablehlo.pairing_check %g1, %g2 : (tensor<4x!bn254_jac>, tensor<4x!bn254_jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
}

// -----

// stablehlo.bit_reverse — vhlo.bit_reverse_v1 floor 1.13.0.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
func.func @bit_reverse(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{failed to legalize operation 'vhlo.bit_reverse_v1' that was explicitly marked illegal}}
  %0 = stablehlo.bit_reverse %arg0, dims = [0] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}
}

// -----

// !vhlo.pf_v1 (prime field) floor 1.13.0 — the element type makes the
// carrying vhlo.func_v1 illegal below the floor.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @pf_type(%arg0: tensor<4x!field.pf<7681:i32>>) -> tensor<4x!field.pf<7681:i32>> {
  func.return %arg0 : tensor<4x!field.pf<7681:i32>>
}
}

// -----

// !vhlo.ef_v1 (extension field) floor 1.13.0.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @ef_type(%arg0: tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>>) -> tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>> {
  func.return %arg0 : tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>>
}
}

// -----

// !vhlo.affine_v1 (EC affine point) floor 1.13.0.
#bn254_a = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ec_affine = !elliptic_curve.affine<#bn254_a>
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @affine_type(%arg0: tensor<4x!ec_affine>) -> tensor<4x!ec_affine> {
  func.return %arg0 : tensor<4x!ec_affine>
}
}

// -----

// !vhlo.jacobian_v1 (EC jacobian point) floor 1.13.0.
#bn254_j = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ec_jacobian = !elliptic_curve.jacobian<#bn254_j>
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @jacobian_type(%arg0: tensor<4x!ec_jacobian>) -> tensor<4x!ec_jacobian> {
  func.return %arg0 : tensor<4x!ec_jacobian>
}
}

// -----

// !vhlo.xyzz_v1 (EC xyzz point) floor 1.13.0.
#bn254_z = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ec_xyzz = !elliptic_curve.xyzz<#bn254_z>
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @xyzz_type(%arg0: tensor<4x!ec_xyzz>) -> tensor<4x!ec_xyzz> {
  func.return %arg0 : tensor<4x!ec_xyzz>
}
}

// -----

// !vhlo.i128_v1 (extended-width integer) floor 1.13.0.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @i128_type(%arg0: tensor<4xi128>) -> tensor<4xi128> {
  func.return %arg0 : tensor<4xi128>
}
}

// -----

// !vhlo.i256_v1 (extended-width integer) floor 1.13.0.
// expected-error @+1 {{failed to convert VHLO to v1.12.0}}
module {
// expected-error @+1 {{failed to legalize operation 'vhlo.func_v1' that was explicitly marked illegal}}
func.func @i256_type(%arg0: tensor<4xi256>) -> tensor<4xi256> {
  func.return %arg0 : tensor<4xi256>
}
}
