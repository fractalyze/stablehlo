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

#ifndef STABLEHLO_TRANSFORMS_PASSES_H_
#define STABLEHLO_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/Version.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/Passes.h.inc"

void populateVhloToStablehloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

void populateVhloToVersionPatterns(RewritePatternSet *patterns,
                                   TypeConverter *converter,
                                   MLIRContext *context);

void createStablehloDeserializePipeline(OpPassManager &pm);

void registerPassPipelines();

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TRANSFORMS_PASSES_H_
