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

#include "stablehlo/conversions/PrimeIRToStablehlo/PrimeIRToStablehlo.h"

#include "mlir/IR/PatternMatch.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {

namespace {

//===----------------------------------------------------------------------===//
// Field → StableHLO conversion patterns
//===----------------------------------------------------------------------===//

/// field.add → stablehlo.add
struct ConvertFieldAddBack : public OpRewritePattern<prime_ir::field::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  }
};

/// field.sub → stablehlo.subtract
struct ConvertFieldSubBack : public OpRewritePattern<prime_ir::field::SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::SubOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<SubtractOp>(op, op.getType(), op.getLhs(),
                                            op.getRhs());
    return success();
  }
};

/// field.mul → stablehlo.multiply
struct ConvertFieldMulBack : public OpRewritePattern<prime_ir::field::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::MulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  }
};

/// field.negate → stablehlo.negate
struct ConvertFieldNegBack
    : public OpRewritePattern<prime_ir::field::NegateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::NegateOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NegOp>(op, op.getType(), op.getInput());
    return success();
  }
};

/// field.double(x) → stablehlo.add(x, x)
struct ConvertFieldDoubleBack
    : public OpRewritePattern<prime_ir::field::DoubleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::DoubleOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getInput(),
                                       op.getInput());
    return success();
  }
};

/// field.square(x) → stablehlo.multiply(x, x)
struct ConvertFieldSquareBack
    : public OpRewritePattern<prime_ir::field::SquareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::SquareOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getInput(),
                                       op.getInput());
    return success();
  }
};

/// field.inverse(x) → stablehlo.divide(1, x)
struct ConvertFieldInverseBack
    : public OpRewritePattern<prime_ir::field::InverseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::InverseOp op,
                                PatternRewriter &rewriter) const override {
    Type elemType = getElementTypeOrSelf(op.getType());

    // Resolve the base prime field type — for PF it's the type itself,
    // for EF we extract the underlying prime field.
    prime_ir::field::PrimeFieldType pfType;
    if (auto pf = dyn_cast<prime_ir::field::PrimeFieldType>(elemType))
      pfType = pf;
    else if (auto ef = dyn_cast<prime_ir::field::ExtensionFieldType>(elemType))
      pfType = ef.getBasePrimeField();
    else
      return failure();

    // Build a base-field "1" constant. DivOp supports mixed PF / EF
    // operands (div_c1 compatibility), so a scalar PF "1" works for both.
    auto fieldOne = prime_ir::field::FieldOperation(uint64_t{1}, pfType);
    APInt val = static_cast<APInt>(fieldOne);
    auto resultShape = cast<ShapedType>(op.getType()).getShape();
    auto oneAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(resultShape, pfType.getStorageType()), {val});

    auto oneType = RankedTensorType::get(resultShape, pfType);
    auto one = rewriter.create<ConstantOp>(op.getLoc(), oneType, oneAttr);
    rewriter.replaceOpWithNewOp<DivOp>(op, op.getType(), one, op.getInput());
    return success();
  }
};

/// field.constant → stablehlo.constant
struct ConvertFieldConstantBack
    : public OpRewritePattern<prime_ir::field::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::field::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Attribute value = op.getValue();
    // field.constant may use IntegerAttr (scalar prime field) or
    // DenseIntElementsAttr (tensor/extension field). StableHLO constants
    // require ElementsAttr. Wrap IntegerAttr into DenseIntElementsAttr
    // matching the result shape (handles splat tensor field.constants
    // from prime-ir constant folding).
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      auto resultType = cast<ShapedType>(op.getType());
      auto tensorType =
          RankedTensorType::get(resultType.getShape(), intAttr.getType());
      value = DenseIntElementsAttr::get(tensorType, {intAttr.getValue()});
    }
    auto elementsAttr = dyn_cast<ElementsAttr>(value);
    if (!elementsAttr)
      return failure();
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// EC → StableHLO conversion patterns
//===----------------------------------------------------------------------===//

/// elliptic_curve.add → stablehlo.add
struct ConvertECAddBack
    : public OpRewritePattern<prime_ir::elliptic_curve::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::elliptic_curve::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  }
};

/// elliptic_curve.sub → stablehlo.subtract
struct ConvertECSubBack
    : public OpRewritePattern<prime_ir::elliptic_curve::SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::elliptic_curve::SubOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<SubtractOp>(op, op.getType(), op.getLhs(),
                                            op.getRhs());
    return success();
  }
};

/// elliptic_curve.negate → stablehlo.negate
struct ConvertECNegBack
    : public OpRewritePattern<prime_ir::elliptic_curve::NegateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::elliptic_curve::NegateOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<NegOp>(op, op.getType(), op.getInput());
    return success();
  }
};

/// elliptic_curve.double(P) → stablehlo.add(P, P)
struct ConvertECDoubleBack
    : public OpRewritePattern<prime_ir::elliptic_curve::DoubleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::elliptic_curve::DoubleOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getInput(),
                                       op.getInput());
    return success();
  }
};

/// elliptic_curve.scalar_mul → stablehlo.multiply
struct ConvertECScalarMulBack
    : public OpRewritePattern<prime_ir::elliptic_curve::ScalarMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(prime_ir::elliptic_curve::ScalarMulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getScalar(),
                                       op.getPoint());
    return success();
  }
};

} // namespace

void populatePrimeIRToStablehloPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // Field → StableHLO
  patterns
      .add<ConvertFieldAddBack, ConvertFieldSubBack, ConvertFieldMulBack,
           ConvertFieldNegBack, ConvertFieldDoubleBack, ConvertFieldSquareBack,
           ConvertFieldInverseBack, ConvertFieldConstantBack>(ctx);
  // EC → StableHLO
  patterns.add<ConvertECAddBack, ConvertECSubBack, ConvertECNegBack,
               ConvertECDoubleBack, ConvertECScalarMulBack>(ctx);
}

} // namespace mlir::stablehlo
