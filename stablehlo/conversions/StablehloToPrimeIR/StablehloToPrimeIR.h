/* Copyright 2026 The StableHLO(Fractalyze) Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIR_STABLEHLOTOPRIMEIR_H_
#define STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIR_STABLEHLOTOPRIMEIR_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::stablehlo {

/// Field / EC arithmetic patterns only. Round-trippable: for every prime-ir
/// op produced here, PrimeIRToStablehlo has an inverse pattern. Used by the
/// `stablehlo-canonicalize` pipeline, which folds arithmetic through
/// prime-ir's algebraic rules and converts back to StableHLO.
void populateStablehloToPrimeIRArithPatterns(RewritePatternSet &patterns);

/// Non-reversible lowerings: `stablehlo.ntt` -> `poly.ntt`,
/// `stablehlo.pairing_check` -> `elliptic_curve.pairing_check`,
/// `stablehlo.msm` -> `scalar_mul + add` chain. These ops don't have an
/// inverse in PrimeIRToStablehlo and must not run inside a round-trip
/// pipeline like `stablehlo-canonicalize`.
void populateStablehloToPrimeIRLoweringPatterns(RewritePatternSet &patterns);

/// All patterns: arith + non-reversible. Used by the standalone
/// `stablehlo-to-prime-ir` pass for end-of-frontend lowering.
void populateStablehloToPrimeIRPatterns(RewritePatternSet &patterns);

}  // namespace mlir::stablehlo

#endif  // STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIR_STABLEHLOTOPRIMEIR_H_
