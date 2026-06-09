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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// stablehlo.convert : tensor<Nx!ec.Repr1> -> tensor<Nx!ec.Repr2> lowers to
// elliptic_curve.convert_point_type. Both element types must be EC point
// types over the same curve; the prime_ir verifier enforces curve match.
//
// Without this pattern, shaped EC stablehlo.convert flows through to
// bufferization and fails with "op was not bufferized" — the elemental
// emitter's per-lane EmitConvert carve-out (xla_fork commit 814a76d8e4)
// only fires on the scalar path, not on shaped tensor-level converts.
struct ConvertECConvert : public OpRewritePattern<ConvertOp> {
  ConvertECConvert(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasECElementType(op.getOperand())) return failure();
    if (!hasECElementType(op.getResult())) return failure();
    // Same-type convert is the identity; fold it away instead of emitting a
    // convert_point_type the prime_ir verifier rejects ("Converting on same
    // types"). Identity converts reach this pass e.g. from jax.export, which
    // wraps arguments of re-imported modules in shape-refinement converts
    // that become element-identity once symbolic shapes are refined.
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    rewriter.replaceOpWithNewOp<prime_ir::elliptic_curve::ConvertPointTypeOp>(
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

// Lowers MSM by unrolling into a chain of N scalar_mul + (N-1) add ops — an
// O(N) expansion that is correct but not asymptotically optimal (a Pippenger
// bucket method would be the fast form). The unroll only applies when:
//   - batch_size <= 1 (single MSM, scalar result);
//   - bases element type is jacobian, so scalar_mul (jacobian -> jacobian) and
//     add (jacobian, jacobian -> jacobian) keep the chain in one coordinate
//     system; affine bases would need a trailing convert_point_type and are
//     rejected so the caller picks the coordinate system explicitly;
//   - N is static, since the chain is built at compile time (dynamic N needs
//     an scf.for loop instead of an unroll);
//   - N <= kMsmUnrollLimit. Each scalar_mul later expands to a
//     double-and-add ladder of ~storage-bit-width group operations, so the
//     post-lowering op count is roughly N * bits — unbounded N exhausts the
//     compiler. Production-size MSMs belong to the runtime lowering, not
//     this unroll.
// Inputs outside these bounds do not match and the op is left for another
// pattern / reported as unlegalized.
constexpr int64_t kMsmUnrollLimit = 32;

struct ConvertMsm : public OpRewritePattern<MsmOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MsmOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBatchSize() > 1) return failure();
    auto scalarsType = cast<RankedTensorType>(op.getScalars().getType());
    auto basesType = cast<RankedTensorType>(op.getBases().getType());
    if (scalarsType.isDynamicDim(0) || basesType.isDynamicDim(0))
      return failure();
    auto jacType = dyn_cast<prime_ir::elliptic_curve::JacobianType>(
        basesType.getElementType());
    if (!jacType) return failure();
    int64_t n = scalarsType.getDimSize(0);
    if (n == 0 || n > kMsmUnrollLimit) return failure();

    Location loc = op.getLoc();
    Value running;
    for (int64_t i = 0; i < n; ++i) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value scalar = tensor::ExtractOp::create(rewriter, loc, op.getScalars(),
                                               ValueRange{idx});
      Value base = tensor::ExtractOp::create(rewriter, loc, op.getBases(),
                                             ValueRange{idx});
      Value product = prime_ir::elliptic_curve::ScalarMulOp::create(
          rewriter, loc, jacType, scalar, base);
      running = i == 0 ? product
                       : prime_ir::elliptic_curve::AddOp::create(
                             rewriter, loc, jacType, running, product)
                             .getResult();
    }
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, op.getType(),
                                                        running);
    return success();
  }
};

// Lowers stablehlo.pairing_check → prime_ir::elliptic_curve::PairingCheckOp.
// prime_ir's op requires affine inputs with G1 over a prime base field and
// G2 over a degree-2 extension field; the verifier additionally checks
// that the curve params alias to a known pairing-friendly family. We
// pre-check the base field shape so this rewrite never produces IR that
// the EC verifier will reject — non-matching inputs leave the op
// untouched (upstream code is expected to insert convert_point_type ops
// or wait for G2 PrimitiveType support).
//
// Result wrapping: prime_ir's op returns scalar i1; stablehlo's result
// is tensor<i1>, so we wrap via tensor.from_elements.
struct ConvertPairingCheck : public OpRewritePattern<PairingCheckOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PairingCheckOp op,
                                PatternRewriter &rewriter) const override {
    auto g1Type = cast<RankedTensorType>(op.getG1Points().getType());
    auto g2Type = cast<RankedTensorType>(op.getG2Points().getType());
    using AffineType = prime_ir::elliptic_curve::AffineType;
    auto g1Aff = dyn_cast<AffineType>(g1Type.getElementType());
    auto g2Aff = dyn_cast<AffineType>(g2Type.getElementType());
    if (!g1Aff || !g2Aff) return failure();
    if (!isa<prime_ir::field::PrimeFieldType>(g1Aff.getBaseFieldType()))
      return failure();
    auto g2Ext =
        dyn_cast<prime_ir::field::ExtensionFieldType>(g2Aff.getBaseFieldType());
    if (!g2Ext || g2Ext.getDegree() != 2) return failure();
    auto i1 = rewriter.getI1Type();
    auto pairing = prime_ir::elliptic_curve::PairingCheckOp::create(
        rewriter, op.getLoc(), i1, op.getG1Points(), op.getG2Points());
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, op.getType(), ValueRange{pairing.getOutput()});
    return success();
  }
};

}  // namespace

void populateStablehloToPrimeIRArithPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertFieldAdd, ConvertFieldSub, ConvertFieldMul,
               ConvertFieldDiv, ConvertFieldNeg, ConvertFieldConstant>(ctx);
  patterns.add<ConvertECAdd, ConvertECSub, ConvertECNeg, ConvertECScalarMul,
               ConvertECConvert>(ctx);
}

void populateStablehloToPrimeIRLoweringPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertPairingCheck, ConvertMsm>(ctx);
}

void populateStablehloToPrimeIRPatterns(RewritePatternSet &patterns) {
  populateStablehloToPrimeIRArithPatterns(patterns);
  populateStablehloToPrimeIRLoweringPatterns(patterns);
}

}  // namespace mlir::stablehlo
