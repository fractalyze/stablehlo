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
