/* Copyright 2026 The StableHLO Authors. All Rights Reserved.

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

// Asserts that stablehlo resolves, compiles and links as a non-root Bazel
// module. The dialect exercise is deliberately trivial: stablehlo's own suite
// covers behaviour, and what is under test here is dependency resolution.
//
// Loading the dialects reaches what a consumer cannot declare for itself: MLIR,
// which arrives from the patched @llvm-project that prime_ir's module extension
// fetches, and the prime_ir dialects that :register links in so that the
// symbols @prime_ir//third_party/llvm-project:linalg_type_support.patch leaves
// for the final link are satisfied.

#include "gtest/gtest.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace {

TEST(ConsumeStablehloTest, DialectsLoad) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  auto *dialect = context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  ASSERT_NE(dialect, nullptr);
  EXPECT_EQ(dialect->getNamespace(), "stablehlo");
}

TEST(ConsumeStablehloTest, PassPipelinesRegister) {
  // Reaches :stablehlo_passes, the third target riscv-witness names.
  mlir::stablehlo::registerPassPipelines();
}

}  // namespace
