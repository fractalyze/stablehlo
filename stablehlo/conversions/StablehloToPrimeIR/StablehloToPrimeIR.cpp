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

#include "stablehlo/conversions/StablehloToPrimeIR/StablehloToPrimeIR.h"

#include "mlir/IR/PatternMatch.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {

using PrimeFieldType = prime_ir::field::PrimeFieldType;
using ExtensionFieldType = prime_ir::field::ExtensionFieldType;
using BinaryFieldType = prime_ir::field::BinaryFieldType;
using FieldTypeInterface = prime_ir::field::FieldTypeInterface;
using PointTypeInterface = prime_ir::elliptic_curve::PointTypeInterface;

namespace {

/// Check if the element type of a value is a field type.
bool hasFieldElementType(Value v) {
  return isa<FieldTypeInterface>(getElementTypeOrSelf(v.getType()));
}

/// Check if the element type of a value is an EC point type.
bool hasECElementType(Value v) {
  return isa<PointTypeInterface>(getElementTypeOrSelf(v.getType()));
}

//===----------------------------------------------------------------------===//
// Field type conversion patterns
//===----------------------------------------------------------------------===//

/// stablehlo.add (field) → field.add
struct ConvertFieldAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::AddOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

/// stablehlo.subtract (field) → field.sub
struct ConvertFieldSub : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::SubOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

/// stablehlo.multiply (field) → field.mul
struct ConvertFieldMul : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs()) || !hasFieldElementType(op.getRhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::MulOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

/// stablehlo.divide (field) → field.mul(x, field.inverse(y))
struct ConvertFieldDiv : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs()))
      return failure();
    auto inv = rewriter.create<prime_ir::field::InverseOp>(
        op.getLoc(), op.getRhs().getType(), op.getRhs());
    rewriter.replaceOpWithNewOp<prime_ir::field::MulOp>(op, op.getType(),
                                                        op.getLhs(), inv);
    return success();
  }
};

/// stablehlo.negate (field) → field.negate
struct ConvertFieldNeg : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getOperand()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::NegateOp>(op, op.getType(),
                                                           op.getOperand());
    return success();
  }
};

/// stablehlo.constant (field) → field.constant
struct ConvertFieldConstant : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Type elemType = getElementTypeOrSelf(op.getType());
    if (!isa<FieldTypeInterface>(elemType))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::ConstantOp>(op, op.getType(),
                                                             op.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// EC type conversion patterns
//===----------------------------------------------------------------------===//

/// stablehlo.add (EC) → elliptic_curve.add
struct ConvertECAdd : public OpRewritePattern<AddOp> {
  // Lower priority so field patterns match first.
  ConvertECAdd(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getLhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::AddOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

/// stablehlo.subtract (EC) → elliptic_curve.sub
struct ConvertECSub : public OpRewritePattern<SubtractOp> {
  ConvertECSub(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getLhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::SubOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

/// stablehlo.negate (EC) → elliptic_curve.negate
struct ConvertECNeg : public OpRewritePattern<NegOp> {
  ConvertECNeg(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getOperand()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::NegateOp>(
        op, op.getType(), op.getOperand());
    return success();
  }
};

/// stablehlo.multiply(scalar, point) → elliptic_curve.scalar_mul
struct ConvertECScalarMul : public OpRewritePattern<MulOp> {
  ConvertECScalarMul(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type lhsElemType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsElemType = getElementTypeOrSelf(op.getRhs().getType());

    // scalar * point
    if (isa<FieldTypeInterface>(lhsElemType) &&
        isa<PointTypeInterface>(rhsElemType)) {
      rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::ScalarMulOp>(
          op, op.getType(), op.getLhs(), op.getRhs());
      return success();
    }
    // point * scalar (swap)
    if (isa<PointTypeInterface>(lhsElemType) &&
        isa<FieldTypeInterface>(rhsElemType)) {
      rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::ScalarMulOp>(
          op, op.getType(), op.getRhs(), op.getLhs());
      return success();
    }
    return failure();
  }
};

} // namespace

void populateStablehloToPrimeIRPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // Field patterns (default benefit = 1)
  patterns.add<ConvertFieldAdd, ConvertFieldSub, ConvertFieldMul,
               ConvertFieldDiv, ConvertFieldNeg, ConvertFieldConstant>(ctx);
  // EC patterns (lower benefit = 0)
  patterns.add<ConvertECAdd, ConvertECSub, ConvertECNeg, ConvertECScalarMul>(
      ctx);
}

} // namespace mlir::stablehlo
