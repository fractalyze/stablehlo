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

#ifndef STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIIR_H_
#define STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIIR_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::stablehlo {

/// Populate patterns for converting field/EC-typed StableHLO ops to prime-ir
/// ops.
void populateStablehloToPrimeIRPatterns(RewritePatternSet &patterns);

} // namespace mlir::stablehlo

#endif // STABLEHLO_CONVERSIONS_STABLEHLOTOPRIMEIIR_H_
