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

using FieldTypeInterface = prime_ir::field::FieldTypeInterface;
using PointTypeInterface = prime_ir::elliptic_curve::PointTypeInterface;

namespace {

bool hasFieldElementType(Value v) {
  return isa<FieldTypeInterface>(getElementTypeOrSelf(v.getType()));
}

bool hasECElementType(Value v) {
  return isa<PointTypeInterface>(getElementTypeOrSelf(v.getType()));
}

//===----------------------------------------------------------------------===//
// Field type conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertFieldAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::AddOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConvertFieldSub : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::SubOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConvertFieldMul : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    // Field*field only; scalar*point goes through ConvertECScalarMul below.
    if (!hasFieldElementType(op.getLhs()) || !hasFieldElementType(op.getRhs()))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::MulOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

// stablehlo.divide(x, y) over a field is x * inverse(y). The field dialect has
// no direct division op — multiplicative inverse is the canonical primitive.
struct ConvertFieldDiv : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getLhs())) return failure();
    auto inv = prime_ir::field::InverseOp::create(
        rewriter, op.getLoc(), op.getRhs().getType(), op.getRhs());
    rewriter.replaceOpWithNewOp<prime_ir::field::MulOp>(op, op.getType(),
                                                        op.getLhs(), inv);
    return success();
  }
};

struct ConvertFieldNeg : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasFieldElementType(op.getOperand())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::NegateOp>(op, op.getType(),
                                                           op.getOperand());
    return success();
  }
};

struct ConvertFieldConstant : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<FieldTypeInterface>(getElementTypeOrSelf(op.getType())))
      return failure();
    rewriter.replaceOpWithNewOp<prime_ir::field::ConstantOp>(op, op.getType(),
                                                             op.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// EC type conversion patterns
//===----------------------------------------------------------------------===//

// Lower benefit (=0) than the field patterns so that for a `multiply` whose
// element type isa<FieldTypeInterface>, the field pattern wins. EC patterns
// match only when at least one operand carries a point type.

struct ConvertECAdd : public OpRewritePattern<AddOp> {
  ConvertECAdd(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getLhs())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::AddOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConvertECSub : public OpRewritePattern<SubtractOp> {
  ConvertECSub(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getLhs())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::SubOp>(
        op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct ConvertECNeg : public OpRewritePattern<NegOp> {
  ConvertECNeg(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getOperand())) return failure();
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::NegateOp>(
        op, op.getType(), op.getOperand());
    return success();
  }
};

// scalar*point or point*scalar lowers to elliptic_curve.scalar_mul(scalar,
// point). Symmetric in user-facing semantics; canonicalized to scalar-first.
struct ConvertECScalarMul : public OpRewritePattern<MulOp> {
  ConvertECScalarMul(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Type lhsElem = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsElem = getElementTypeOrSelf(op.getRhs().getType());
    if (isa<FieldTypeInterface>(lhsElem) && isa<PointTypeInterface>(rhsElem)) {
      rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::ScalarMulOp>(
          op, op.getType(), op.getLhs(), op.getRhs());
      return success();
    }
    if (isa<PointTypeInterface>(lhsElem) && isa<FieldTypeInterface>(rhsElem)) {
      rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::ScalarMulOp>(
          op, op.getType(), op.getRhs(), op.getLhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void populateStablehloToPrimeIRPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertFieldAdd, ConvertFieldSub, ConvertFieldMul,
               ConvertFieldDiv, ConvertFieldNeg, ConvertFieldConstant>(ctx);
  patterns.add<ConvertECAdd, ConvertECSub, ConvertECNeg, ConvertECScalarMul>(
      ctx);
}

}  // namespace mlir::stablehlo
