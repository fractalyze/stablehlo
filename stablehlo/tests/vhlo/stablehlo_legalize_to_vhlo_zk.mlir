// Regression coverage for fork-added VHLO ops/types/enums and the
// downgrade-to-WEEK_4-target path that `jax.export` exercises.
//
// Verifies three things in lock-step (one regression → one breakage):
//   1. `stablehlo-legalize-to-vhlo` rewrites prime_ir element types
//      into their `vhlo.pf_v1` / `vhlo.ef_v1` twins (NOT the pre-twin
//      pass-through where the original `!field.pf<...>` rode through).
//   2. `vhlo-to-version target=1.13.9` keeps the fork-added op
//      (`vhlo.ntt_v1`), its enum attr (`#vhlo<ntt_type_v1 NTT>`), and
//      the field twin types legal at JAX's WEEK_4 anchor. Patch is
//      dropped, so the effective target is 1.13.0.
//   3. The full VHLO `--serialize --target=1.13.9` + `--deserialize`
//      portable-artifact round-trip preserves these ops + types.
//      Exercises the bytecode encoders for the field twin types
//      (kPrimeFieldV1Type=47, kExtensionFieldV1Type=48).

// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --mlir-print-op-generic %s | FileCheck %s --check-prefix=LEGALIZE
// RUN: stablehlo-opt --stablehlo-legalize-to-vhlo --vhlo-to-version=target=1.13.9 %s 2>&1 | FileCheck %s --check-prefix=DOWNGRADE
// RUN: stablehlo-translate --serialize --target=1.13.9 %s | stablehlo-translate --deserialize | FileCheck %s --check-prefix=ROUNDTRIP

// LEGALIZE-LABEL: "op_ntt_pf"
// LEGALIZE: "vhlo.ntt_v1"
// LEGALIZE-SAME: !vhlo.pf_v1<2130706433 : i32, true>
// DOWNGRADE-LABEL: vhlo.func_v1 @op_ntt_pf
// DOWNGRADE: vhlo.ntt_v1
// DOWNGRADE-SAME: ntt_type_v1
// DOWNGRADE-SAME: !vhlo.pf_v1<2130706433 : i32, true>
// ROUNDTRIP-DAG: !pf_koalabear_mont = !field.pf<2130706433 : i32, true>
// ROUNDTRIP-DAG: !pf_babybear = !field.pf<2013265921 : i32>
// ROUNDTRIP-LABEL: @op_ntt_pf
// ROUNDTRIP: stablehlo.ntt
// ROUNDTRIP-SAME: !pf_koalabear_mont
func.func @op_ntt_pf(%arg0: tensor<8x!field.pf<2130706433 : i32, true>>) -> tensor<8x!field.pf<2130706433 : i32, true>> {
  %0 = stablehlo.ntt %arg0, type = NTT, length = 8 : tensor<8x!field.pf<2130706433 : i32, true>>
  func.return %0 : tensor<8x!field.pf<2130706433 : i32, true>>
}

// Extension field round-trip + downgrade (covers vhlo.ef_v1 bytecode codes).
// LEGALIZE-LABEL: "op_ef_passthrough"
// LEGALIZE: !vhlo.ef_v1<4 x !vhlo.pf_v1<2013265921 : i32, false>, 11 : i32>
// DOWNGRADE-LABEL: vhlo.func_v1 @op_ef_passthrough
// DOWNGRADE: vhlo.add_v1
// DOWNGRADE-SAME: !vhlo.ef_v1<4 x !vhlo.pf_v1<2013265921 : i32, false>, 11 : i32>
// ROUNDTRIP-LABEL: @op_ef_passthrough
// ROUNDTRIP: stablehlo.add
// ROUNDTRIP-SAME: !field.ef<4x!pf_babybear, 11 : i32>
func.func @op_ef_passthrough(
    %arg0: tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>>
  ) -> tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>> {
  %0 = stablehlo.add %arg0, %arg0
    : tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>>
  func.return %0
    : tensor<4x!field.ef<4x!field.pf<2013265921 : i32>, 11 : i32>>
}

// INTT-typed transform — crosses the second NttType enum value through the
// bytecode round-trip (op_ntt_pf above carries NTT; this carries INTT).
// LEGALIZE-LABEL: "op_intt_pf"
// LEGALIZE: "vhlo.ntt_v1"
// LEGALIZE-SAME: INTT
// DOWNGRADE-LABEL: vhlo.func_v1 @op_intt_pf
// DOWNGRADE: vhlo.ntt_v1
// DOWNGRADE-SAME: INTT
// ROUNDTRIP-LABEL: @op_intt_pf
// ROUNDTRIP: stablehlo.ntt
// ROUNDTRIP-SAME: type = INTT
func.func @op_intt_pf(%arg0: tensor<8x!field.pf<2130706433 : i32, true>>) -> tensor<8x!field.pf<2130706433 : i32, true>> {
  %0 = stablehlo.ntt %arg0, type = INTT, length = 8 : tensor<8x!field.pf<2130706433 : i32, true>>
  func.return %0 : tensor<8x!field.pf<2130706433 : i32, true>>
}

// stablehlo.bit_reverse — fork-added op, dims attr rides the round-trip.
// LEGALIZE-LABEL: "op_bit_reverse"
// LEGALIZE: "vhlo.bit_reverse_v1"
// DOWNGRADE-LABEL: vhlo.func_v1 @op_bit_reverse
// DOWNGRADE: vhlo.bit_reverse_v1
// ROUNDTRIP-LABEL: @op_bit_reverse
// ROUNDTRIP: stablehlo.bit_reverse
// ROUNDTRIP-SAME: dims = [0]
func.func @op_bit_reverse(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = stablehlo.bit_reverse %arg0, dims = [0] : tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// stablehlo.msm with its full attr set (window_bits / precompute_factor /
// bitsize / batch_size / are_points_shared) crossing vhlo.msm_v1 bytecode.
// Carries AffineV1 bases + XYZZV1 result through the round-trip.
!BN254_Fr = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
#bn254_g1 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!bn254_affine = !elliptic_curve.affine<#bn254_g1>
!bn254_xyzz = !elliptic_curve.xyzz<#bn254_g1>

// LEGALIZE-LABEL: "op_msm"
// LEGALIZE: "vhlo.msm_v1"
// DOWNGRADE-LABEL: vhlo.func_v1 @op_msm
// DOWNGRADE: vhlo.msm_v1
// ROUNDTRIP-LABEL: @op_msm
// ROUNDTRIP: stablehlo.msm
func.func @op_msm(%scalars: tensor<2048x!BN254_Fr>, %bases: tensor<1024x!bn254_affine>) -> tensor<2x!bn254_xyzz> {
  %0 = stablehlo.msm %scalars, %bases {window_bits = 16 : i32, precompute_factor = 2 : i32, bitsize = 253 : i32, batch_size = 2 : i32, are_points_shared = true} : (tensor<2048x!BN254_Fr>, tensor<1024x!bn254_affine>) -> tensor<2x!bn254_xyzz>
  func.return %0 : tensor<2x!bn254_xyzz>
}

// stablehlo.pairing_check — fork-added op + JacobianV1 in the signature.
#bn254_pc = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!bn254_jac = !elliptic_curve.jacobian<#bn254_pc>

// LEGALIZE-LABEL: "op_pairing_check"
// LEGALIZE: "vhlo.pairing_check_v1"
// DOWNGRADE-LABEL: vhlo.func_v1 @op_pairing_check
// DOWNGRADE: vhlo.pairing_check_v1
// ROUNDTRIP-LABEL: @op_pairing_check
// ROUNDTRIP: stablehlo.pairing_check
func.func @op_pairing_check(%g1: tensor<4x!bn254_jac>, %g2: tensor<4x!bn254_jac>) -> tensor<i1> {
  %0 = stablehlo.pairing_check %g1, %g2 : (tensor<4x!bn254_jac>, tensor<4x!bn254_jac>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// EC point types in op signatures so AffineV1 / JacobianV1 / XYZZV1 each
// legalize and ride the bytecode round-trip via a plain negate.
#bn254_ec = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ec_affine = !elliptic_curve.affine<#bn254_ec>
!ec_jacobian = !elliptic_curve.jacobian<#bn254_ec>
!ec_xyzz = !elliptic_curve.xyzz<#bn254_ec>

// LEGALIZE-LABEL: "op_negate_affine"
// LEGALIZE: "vhlo.negate_v1"
// LEGALIZE-SAME: !vhlo.affine_v1
// DOWNGRADE-LABEL: vhlo.func_v1 @op_negate_affine
// DOWNGRADE: vhlo.negate_v1
// DOWNGRADE-SAME: !vhlo.affine_v1
// ROUNDTRIP-LABEL: @op_negate_affine
// ROUNDTRIP: stablehlo.negate
func.func @op_negate_affine(%arg0: tensor<4x!ec_affine>) -> tensor<4x!ec_affine> {
  %0 = stablehlo.negate %arg0 : tensor<4x!ec_affine>
  func.return %0 : tensor<4x!ec_affine>
}

// LEGALIZE-LABEL: "op_negate_jacobian"
// LEGALIZE: "vhlo.negate_v1"
// LEGALIZE-SAME: !vhlo.jacobian_v1
// DOWNGRADE-LABEL: vhlo.func_v1 @op_negate_jacobian
// DOWNGRADE: vhlo.negate_v1
// DOWNGRADE-SAME: !vhlo.jacobian_v1
// ROUNDTRIP-LABEL: @op_negate_jacobian
// ROUNDTRIP: stablehlo.negate
func.func @op_negate_jacobian(%arg0: tensor<4x!ec_jacobian>) -> tensor<4x!ec_jacobian> {
  %0 = stablehlo.negate %arg0 : tensor<4x!ec_jacobian>
  func.return %0 : tensor<4x!ec_jacobian>
}

// LEGALIZE-LABEL: "op_negate_xyzz"
// LEGALIZE: "vhlo.negate_v1"
// LEGALIZE-SAME: !vhlo.xyzz_v1
// DOWNGRADE-LABEL: vhlo.func_v1 @op_negate_xyzz
// DOWNGRADE: vhlo.negate_v1
// DOWNGRADE-SAME: !vhlo.xyzz_v1
// ROUNDTRIP-LABEL: @op_negate_xyzz
// ROUNDTRIP: stablehlo.negate
func.func @op_negate_xyzz(%arg0: tensor<4x!ec_xyzz>) -> tensor<4x!ec_xyzz> {
  %0 = stablehlo.negate %arg0 : tensor<4x!ec_xyzz>
  func.return %0 : tensor<4x!ec_xyzz>
}
