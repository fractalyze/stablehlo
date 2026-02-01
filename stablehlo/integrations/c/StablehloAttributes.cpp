/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
   Copyright 2025 The StableHLO(Fractalyze) Authors.

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

#include "stablehlo/integrations/c/StablehloAttributes.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx, intptr_t nUpdateWindowDims,
    const int64_t *updateWindowDims, intptr_t nInsertedWindowDims,
    const int64_t *insertedWindowDims, intptr_t nInputBatchingDims,
    const int64_t *inputBatchingDims, intptr_t nScatterIndicesBatchingDims,
    const int64_t *scatterIndicesBatchingDims,
    intptr_t nScatteredDimsToOperandDims,
    const int64_t *scatteredDimsToOperandDims, int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::ScatterDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(updateWindowDims, nUpdateWindowDims),
      llvm::ArrayRef(insertedWindowDims, nInsertedWindowDims),
      llvm::ArrayRef(inputBatchingDims, nInputBatchingDims),
      llvm::ArrayRef(scatterIndicesBatchingDims, nScatterIndicesBatchingDims),
      llvm::ArrayRef(scatteredDimsToOperandDims, nScatteredDimsToOperandDims),
      indexVectorDim));
}

bool stablehloAttributeIsAScatterDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr));
}

intptr_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()
      .size();
}

int64_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(MlirAttribute attr,
                                                        intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getUpdateWindowDims()[pos];
}

intptr_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()
      .size();
}

int64_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(MlirAttribute attr,
                                                          intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInsertedWindowDims()[pos];
}

intptr_t
stablehloScatterDimensionNumbersGetInputBatchingDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchingDims()
      .size();
}

int64_t
stablehloScatterDimensionNumbersGetInputBatchingDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getInputBatchingDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterIndicesBatchingDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterIndicesBatchingDims()[pos];
}

intptr_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()
      .size();
}

int64_t stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getScatterDimsToOperandDims()[pos];
}

int64_t stablehloScatterDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::ScatterDimensionNumbersAttr>(unwrap(attr))
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nOperandBatchingDims, const int64_t *operandBatchingDims,
    intptr_t nStartIndicesBatchingDims, const int64_t *startIndicesBatchingDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim) {
  return wrap(mlir::stablehlo::GatherDimensionNumbersAttr::get(
      unwrap(ctx), llvm::ArrayRef(offsetDims, nOffsetDims),
      llvm::ArrayRef(collapsedSliceDims, nCollapsedSliceDims),
      llvm::ArrayRef(operandBatchingDims, nOperandBatchingDims),
      llvm::ArrayRef(startIndicesBatchingDims, nStartIndicesBatchingDims),
      llvm::ArrayRef(startIndexMap, nStartIndexMap), indexVectorDim));
}

bool stablehloAttributeIsAGatherDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr));
}

intptr_t stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOffsetDims()[pos];
}

intptr_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()
      .size();
}

int64_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getCollapsedSliceDims()[pos];
}

intptr_t
stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOperandBatchingDims()
      .size();
}

int64_t
stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(MlirAttribute attr,
                                                          intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getOperandBatchingDims()[pos];
}

intptr_t stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndicesBatchingDims()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndicesBatchingDims()[pos];
}

intptr_t
stablehloGatherDimensionNumbersGetStartIndexMapSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()
      .size();
}

int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getStartIndexMap()[pos];
}

int64_t stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::GatherDimensionNumbersAttr>(unwrap(attr))
      .getIndexVectorDim();
}

//===----------------------------------------------------------------------===//
// DotDimensionNumbers
//===----------------------------------------------------------------------===//

MlirAttribute stablehloDotDimensionNumbersGet(
    MlirContext ctx, intptr_t nLhsBatchingDimensions,
    const int64_t *lhsBatchingDimensions, intptr_t nRhsBatchingDimensions,
    const int64_t *rhsBatchingDimensions, intptr_t nLhsContractingDimensions,
    const int64_t *lhsContractingDimensions, intptr_t nRhsContractingDimensions,
    const int64_t *rhsContractingDimensions) {
  return wrap(mlir::stablehlo::DotDimensionNumbersAttr::get(
      unwrap(ctx),
      llvm::ArrayRef(lhsBatchingDimensions, nLhsBatchingDimensions),
      llvm::ArrayRef(rhsBatchingDimensions, nRhsBatchingDimensions),
      llvm::ArrayRef(lhsContractingDimensions, nLhsContractingDimensions),
      llvm::ArrayRef(rhsContractingDimensions, nRhsContractingDimensions)));
}

bool stablehloAttributeIsADotDimensionNumbers(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr));
}

intptr_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()
      .size();
}

int64_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsBatchingDimensions()[pos];
}

intptr_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()
      .size();
}

int64_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsBatchingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()
      .size();
}

int64_t
stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getLhsContractingDimensions()[pos];
}

intptr_t stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
    MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()
      .size();
}

int64_t
stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return llvm::cast<mlir::stablehlo::DotDimensionNumbersAttr>(unwrap(attr))
      .getRhsContractingDimensions()[pos];
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonDirectionAttrGet(MlirContext ctx,
                                                  MlirStringRef value) {
  std::optional<mlir::stablehlo::ComparisonDirection> comparisonDirection =
      mlir::stablehlo::symbolizeComparisonDirection(unwrap(value));
  if (!comparisonDirection)
    llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::ComparisonDirectionAttr::get(
      unwrap(ctx), comparisonDirection.value()));
}

bool stablehloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr));
}

MlirStringRef stablehloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonDirection(
      llvm::cast<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr))
          .getValue()));
}

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTypeExtensionsGet(MlirContext ctx, intptr_t nBounds,
                                         const int64_t *bounds) {
  return wrap(mlir::stablehlo::TypeExtensionsAttr::get(
      unwrap(ctx), llvm::ArrayRef(bounds, nBounds)));
}

bool stablehloAttributeIsTypeExtensions(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr));
}

intptr_t stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()
      .size();
}

int64_t stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()[pos];
}
