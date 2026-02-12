/* Copyright 2024 The StableHLO Authors.
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

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/VhloOps.td"
include "mlir/IR/CommonAttrConstraints.td"
include "stablehlo/dialect/VhloEnums.td"
include "stablehlo/dialect/VhloAttrs.td"

// ZK: All version upgrade/downgrade patterns removed.
// Only V1 ops exist in the ZK dialect - no V2 ops to convert between.
