// Round-trip via the in-memory legalize passes pins two recent
// VhloLegalizeToStablehlo convertSpecial fixes and the VHLO twin types
// for prime_ir field/EC element types.
//
// * BitReverseOpV1.dimensions: convertSpecial must mirror ReverseOp's
//   entry. Without it the default reverse path decodes the dimensions
//   attribute to DenseTypedElementsAttr, BitReverseOp's
//   setPropertiesFromAttr fails the dyn_cast, and the no-error
//   emit-error function_ref (nullptr in the no-fail build entry point)
//   gets dereferenced.
//
// * NttOpV1.ntt_length: StablehloOps.td defines it as a scalar
//   ConfinedAttr<I64Attr, [IntNonNegative]>. The previous convertSpecial
//   treated it as a DenseI64Array, silently emitting an array attribute
//   on the reverse leg that the stablehlo verifier then rejected.
//
// * The NTT path also requires VHLO twin types
//   (PrimeFieldV1Type/ExtensionFieldV1Type/AffineV1Type/...) so that
//   `RankedTensorV1Type::verify` accepts the field element type as
//   "from VHLO". The opaque pass-through that preceded these twin types
//   was rejected by the verifier.

// RUN: stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-legalize-to-stablehlo \
// RUN:   | FileCheck %s

!pf_babybear = !field.pf<2013265921 : i32>

// CHECK-LABEL: func.func @bit_reverse_roundtrip
// CHECK-SAME: tensor<16xi32>
// CHECK: stablehlo.bit_reverse %{{.*}}, dims = [0] : tensor<16xi32>
func.func @bit_reverse_roundtrip(%arg0: tensor<16xi32>) -> tensor<16xi32> {
  %0 = "stablehlo.bit_reverse"(%arg0) {
    dimensions = array<i64: 0>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// VHLO_NttOpV1 requires `generator`, so each NTT here gives one explicitly.
// The point of these cases is that `ntt_length` survives the round-trip as
// a scalar (not as a dense array, which the previous convertSpecial
// incorrectly produced).
// CHECK-LABEL: func.func @ntt_length_scalar_fwd
// CHECK: stablehlo.ntt %{{.*}}, type = NTT, length = 16
// CHECK-NOT: length = dense
// CHECK-NOT: length = array
func.func @ntt_length_scalar_fwd(%arg0: tensor<16x!pf_babybear>)
    -> tensor<16x!pf_babybear> {
  %0 = stablehlo.ntt %arg0, type = NTT, length = 16, generator = 5
      : tensor<16x!pf_babybear>
  return %0 : tensor<16x!pf_babybear>
}

// CHECK-LABEL: func.func @ntt_length_scalar_inv
// CHECK: stablehlo.ntt %{{.*}}, type = INTT, length = 16
// CHECK-NOT: length = dense
// CHECK-NOT: length = array
func.func @ntt_length_scalar_inv(%arg0: tensor<16x!pf_babybear>)
    -> tensor<16x!pf_babybear> {
  %0 = stablehlo.ntt %arg0, type = INTT, length = 16, generator = 5
      : tensor<16x!pf_babybear>
  return %0 : tensor<16x!pf_babybear>
}

// Power-of-two length other than 16 — guards against the off-by-one where
// the old DenseI64Array path encoded a 1-element array that happened to
// equal `length` and so escaped scalar checks.
// CHECK-LABEL: func.func @ntt_length_scalar_1024
// CHECK: stablehlo.ntt %{{.*}}, type = NTT, length = 1024
// CHECK-NOT: length = dense
// CHECK-NOT: length = array
func.func @ntt_length_scalar_1024(%arg0: tensor<1024x!pf_babybear>)
    -> tensor<1024x!pf_babybear> {
  %0 = stablehlo.ntt %arg0, type = NTT, length = 1024, generator = 5
      : tensor<1024x!pf_babybear>
  return %0 : tensor<1024x!pf_babybear>
}
