/* Copyright 2022 The StableHLO Authors.
Copyright 2026 The StableHLO(Fractalyze) Authors.

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

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::stablehlo {

void createStablehloDeserializePipeline(OpPassManager &pm) {
  // Convert VHLO(version x.y.z) --> VHLO(current).
  pm.addPass(
      createVhloToVersionPass({vhlo::Version::getCurrentVersion().toString()}));

  // Convert VHLO --> StableHLO. Will not fail within compatibility window.
  pm.addPass(createVhloLegalizeToStablehloPass());
}

void createStablehloCanonicalizePipeline(OpPassManager &pm) {
  // Step 1: Convert field/EC-typed StableHLO ops to prime-ir dialect ops.
  pm.addPass(createStablehloToPrimeIRPass());

  // Step 2: Run MLIR canonicalizer to pick up prime-ir's canonicalization
  // patterns (strength reduction, algebraic identities, distributivity, etc.)
  pm.addPass(createCanonicalizerPass());

  // Step 3: Convert prime-ir ops back to StableHLO, expanding specialized ops
  // (double, square, inverse) into StableHLO equivalents.
  pm.addPass(createPrimeIRToStablehloPass());
}

void registerPassPipelines() {
  PassPipelineRegistration<>("stablehlo-deserialize",
                             "Run an example pipeline.",
                             createStablehloDeserializePipeline);
  PassPipelineRegistration<>(
      "stablehlo-canonicalize",
      "Canonicalize field/EC-typed StableHLO ops via prime-ir dialect.",
      createStablehloCanonicalizePipeline);
}

} // namespace mlir::stablehlo
