/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/dialect/Register.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir {
namespace stablehlo {

void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::quant::QuantDialect,
                  mlir::sparse_tensor::SparseTensorDialect>();
  registry.insert<mlir::chlo::ChloDialect,
                  mlir::stablehlo::StablehloDialect,
                  mlir::vhlo::VhloDialect>();
  // stablehlo's tensor type constraints accept prime-ir field
  // and elliptic-curve element types alongside the FP/int/complex set,
  // so any tool consuming a stablehlo module that mentions them must
  // load these dialects. ModArith is bridged for lazy linking but
  // intentionally not registered here yet.
  registry.insert<mlir::prime_ir::field::FieldDialect,
                  mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();
  // clang-format on
}

}  // namespace stablehlo
}  // namespace mlir
