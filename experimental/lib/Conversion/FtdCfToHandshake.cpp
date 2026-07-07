//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the fast token delivery methodology
// https://ieeexplore.ieee.org/abstract/document/10035134, together with the
// straight LSQ allocation https://dl.acm.org/doi/abs/10.1145/3543622.3573050.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>

// [START Boilerplate code for the MLIR pass]
#include "experimental/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

// ---------------------------------------------------------------------------
// Optional debug logging.
//
// When the FTD_LOG_DIR environment variable is set, each function appends to
// "<FTD_LOG_DIR>/<func>_ftd.log" a record of the FTD pass phase boundaries.
// Module-level phase boundaries (everything that happens before/around the
// per-function conversion, e.g. CFG-topology capture, analysis construction,
// and applyFullConversion) are written to "<FTD_LOG_DIR>/__module_ftd.log".
//
// Every line is flushed immediately AND mirrored to stderr (unbuffered), so if
// the pass aborts (e.g. heap corruption) the LAST line that appears identifies
// the phase that was running when the crash surfaced — even if the file
// stream's heap-allocated buffers are unusable at abort time. The stderr mirror
// also disambiguates the "empty log file" case: if FTD_LOG_DIR points somewhere
// the process cannot write, openFtdLog now prints a warning to stderr and
// logging continues on stderr only.
// ---------------------------------------------------------------------------

/// Returns true exactly once-per-process if FTD logging is requested via the
/// FTD_LOG_DIR environment variable. When true, ftdLogStep mirrors every line
/// to stderr regardless of whether the per-function/module file could be
/// opened.
static bool ftdLogEnabled() {
  static const bool enabled = (std::getenv("FTD_LOG_DIR") != nullptr);
  return enabled;
}

static std::unique_ptr<llvm::raw_fd_ostream> openFtdLog(StringRef funcName) {
  const char *logDir = std::getenv("FTD_LOG_DIR");
  if (!logDir)
    return nullptr;
  std::string path = std::string(logDir) + "/" + funcName.str() + "_ftd.log";
  std::error_code ec;
  auto os = std::make_unique<llvm::raw_fd_ostream>(path, ec,
                                                   llvm::sys::fs::OF_Append);
  if (ec) {
    // Surface the path problem instead of silently dropping every log line.
    llvm::errs() << "[FTD] WARNING: could not open log file '" << path
                 << "': " << ec.message() << " (logging to stderr only)\n";
    llvm::errs().flush();
    return nullptr;
  }
  return os;
}

static void ftdLogStep(llvm::raw_ostream *log, const llvm::Twine &msg) {
  if (log) {
    *log << msg << "\n";
    log->flush();
  }
  // Mirror to stderr when logging is enabled. errs() is unbuffered and
  // allocation-light, so it survives a corrupted heap better than the file
  // stream and is visible in the terminal right before an abort.
  if (ftdLogEnabled()) {
    llvm::errs() << "[FTD] " << msg << "\n";
    llvm::errs().flush();
  }
}

struct AllocaOpConversion : public DynOpConversionPattern<memref::AllocaOp> {
  using DynOpConversionPattern<memref::AllocaOp>::DynOpConversionPattern;

  // Construct a dense element attribute with everything zeroes.
  DenseElementsAttr getZeroAttr(ShapedType type) const {
    auto elemType = type.getElementType();
    if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      return DenseElementsAttr::get(type, APInt(intTy.getWidth(), 0));
    }
    if (auto floatTy = dyn_cast<FloatType>(type)) {
      if (floatTy.isF16())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEhalf(), /*negative=*/false));
      if (floatTy.isBF16())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::BFloat(), /*negative=*/false));
      if (floatTy.isF32())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEsingle(), /*negative=*/false));
      if (floatTy.isF64())
        return DenseElementsAttr::get(
            type, APFloat::getZero(APFloat::IEEEdouble(), /*negative=*/false));
      llvm::report_fatal_error("Unhandled float element type!");
    }
    llvm::report_fatal_error("Unknown base element type!");
  }

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    // HACK: By default, we initialize the memory with all zeros. According to
    // the C standard, this only happens for arrays.
    rewriter.replaceOpWithNewOp<handshake::RAMOp>(op, op.getType(),
                                                  getZeroAttr(op.getType()));
    return success();
  }
};

struct GetGlobalOpConversion
    : public DynOpConversionPattern<memref::GetGlobalOp> {
  using DynOpConversionPattern<memref::GetGlobalOp>::DynOpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    // clang-format off
    // Example:
    //  memref.global "external" constant @internal_array : memref<...> = dense<...>
    //  ....
    //  %4 = memref.get_global @internal_array : memref<...>
    //
    // In this case, we remove the global constant and rewrite the addressof
    // node into a RAMOp (and we put an attribute to describe its constant
    // value).
    // clang-format on
    SymbolTableCollection symbolTableCollection;

    auto symNameOfGetGlobal = op.getNameAttr();

    memref::GlobalOp global;
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    moduleOp.walk([&global, symNameOfGetGlobal](memref::GlobalOp gbl) {
      if (gbl.getSymName() == symNameOfGetGlobal.getValue()) {
        global = gbl;
      }
    });

    if (!global) {
      // No corresponding Global (maybe emit pass failure is better)
      return failure();
    }

    /// The initial value doesn't have any type constraints. Therefore we need
    /// to check if it is stored as dense elements.
    mlir::Attribute initValueAttr = global.getInitialValueAttr();
    if (auto denseAttr = initValueAttr.dyn_cast<DenseElementsAttr>()) {
      rewriter.replaceOpWithNewOp<handshake::RAMOp>(op, op.getType(),
                                                    denseAttr);
    } else {
      llvm::report_fatal_error(
          "The initial value must be denoted in DenseElementsAttr.");
    }
    return success();
  }
};

// TODO: Here we simply erase all the global variables and attach the initial
// values to the RAMOps inside the handshake function.
struct GlobalOpConversion : public DynOpConversionPattern<memref::GlobalOp> {
  using DynOpConversionPattern<memref::GlobalOp>::DynOpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Per-block edge information captured from CF-level IR before conversion.
struct BlockEdgeInfo {
  bool isConditional = false;
  bool hasSuccessors = false;
  unsigned trueSuccIdx = 0;
  unsigned falseSuccIdx = 0;
  unsigned uncondSuccIdx = 0;
};

/// Complete CFG topology of one function, captured before conversion.
struct OriginalCFGInfo {
  unsigned numBlocks = 0;
  SmallVector<BlockEdgeInfo> blockEdges;
};

/// Walk every func::FuncOp in the module and capture its CFG topology.
/// Must be called BEFORE applyFullConversion.
static DenseMap<StringRef, OriginalCFGInfo>
captureAllCFGTopologies(ModuleOp moduleOp, llvm::raw_ostream *log) {
  DenseMap<StringRef, OriginalCFGInfo> result;

  ftdLogStep(log, "[capture] start");

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    ftdLogStep(log, "[capture] visiting func: " + funcOp.getSymName());

    if (funcOp.isExternal() || funcOp.getSymName().startswith("__init")) {
      ftdLogStep(log, "[capture]   skipped (external/__init)");
      continue;
    }

    Region &region = funcOp.getBody();
    if (region.empty()) {
      ftdLogStep(log, "[capture]   skipped (empty region)");
      continue;
    }

    OriginalCFGInfo info;

    DenseMap<Block *, unsigned> blockIdx;
    for (auto [idx, block] : llvm::enumerate(region))
      blockIdx[&block] = idx;

    info.numBlocks = blockIdx.size();
    info.blockEdges.resize(info.numBlocks);
    ftdLogStep(log, "[capture]   numBlocks=" + llvm::Twine(info.numBlocks));

    for (auto &[block, idx] : blockIdx) {
      BlockEdgeInfo &edge = info.blockEdges[idx];
      Operation *term = block->getTerminator();
      ftdLogStep(log, "[capture]   block idx=" + llvm::Twine(idx) +
                          " term=" + (term ? term->getName().getStringRef()
                                           : llvm::StringRef("<null>")));

      if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        edge.isConditional = true;
        edge.hasSuccessors = true;
        edge.trueSuccIdx = blockIdx.lookup(condBr.getTrueDest());
        edge.falseSuccIdx = blockIdx.lookup(condBr.getFalseDest());
      } else if (auto br = dyn_cast<cf::BranchOp>(term)) {
        edge.hasSuccessors = true;
        edge.uncondSuccIdx = blockIdx.lookup(br.getDest());
      }
    }

    ftdLogStep(log, "[capture]   recording topology for " + funcOp.getSymName());
    result[funcOp.getSymName()] = std::move(info);
  }

  ftdLogStep(log, "[capture] done (funcs recorded=" +
                      llvm::Twine(result.size()) + ")");

  return result;
}

/// Create a ShadowCFG for a given handshake::FuncOp using the topology
/// captured before conversion.
static ftd::ShadowCFG buildShadowCFG(OpBuilder &builder,
                                     handshake::FuncOp realFuncOp,
                                     const OriginalCFGInfo &info,
                                     llvm::raw_ostream *log) {
  ftd::ShadowCFG shadow;
  Location loc = realFuncOp.getLoc();

  ftdLogStep(log, "[shadow]   buildShadowCFG start (numBlocks=" +
                      llvm::Twine(info.numBlocks) + ")");

  // 1. Create a temporary func::FuncOp with blocks + CF terminators
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(realFuncOp);
    auto funcType = builder.getFunctionType({}, {});
    shadow.shadowFunc =
        builder.create<func::FuncOp>(loc, "__ftd_shadow_cfg__", funcType);
    ftdLogStep(log, "[shadow]   shadow func created");

    Region &R = shadow.shadowFunc.getBody();
    SmallVector<Block *> blocks;
    for (unsigned i = 0; i < info.numBlocks; ++i)
      blocks.push_back(builder.createBlock(&R, R.end()));
    ftdLogStep(log, "[shadow]   created " + llvm::Twine(blocks.size()) +
                        " shadow blocks");

    for (unsigned i = 0; i < info.numBlocks; ++i) {
      const BlockEdgeInfo &edge = info.blockEdges[i];
      builder.setInsertionPointToEnd(blocks[i]);

      if (edge.isConditional) {
        auto dummyCond =
            builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
        builder.create<cf::CondBranchOp>(
            loc, dummyCond, blocks[edge.trueSuccIdx], ValueRange{},
            blocks[edge.falseSuccIdx], ValueRange{});
      } else if (edge.hasSuccessors) {
        builder.create<cf::BranchOp>(loc, blocks[edge.uncondSuccIdx]);
      } else {
        builder.create<func::ReturnOp>(loc);
      }
    }
    ftdLogStep(log, "[shadow]   wired shadow terminators");
  }

  // 2. Scan the real funcOp to map BB index -> real condition Value
  realFuncOp.walk([&](handshake::ConditionalBranchOp brOp) {
    if (brOp->hasAttr("ftd.skip"))
      return;
    auto bbAttr = brOp->getAttrOfType<IntegerAttr>("handshake.bb");
    if (!bbAttr)
      return;
    unsigned bbIdx = bbAttr.getUInt();
    if (!shadow.conditionMap.contains(bbIdx))
      shadow.conditionMap[bbIdx] = brOp.getConditionOperand();
  });
  ftdLogStep(log, "[shadow]   condition scan done (conditions=" +
                      llvm::Twine(shadow.conditionMap.size()) + ")");

  return shadow;
}

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // Module-level log: covers everything that happens before/around the
    // per-function conversion. Opened first so the very first steps (topology
    // capture, analysis construction) are recorded. ftdLogStep also mirrors to
    // stderr, so even if this file cannot be opened the steps are still visible.
    auto mlogOwner = openFtdLog("__module");
    llvm::raw_ostream *mlog = mlogOwner.get();

    {
      SmallString<256> cwd;
      std::error_code cwdEc = llvm::sys::fs::current_path(cwd);
      const char *logDir = std::getenv("FTD_LOG_DIR");
      ftdLogStep(mlog, "[pass] ==== FtdCfToHandshakePass::runDynamaticPass ====");
      ftdLogStep(mlog, llvm::Twine("[pass] cwd=") +
                           (cwdEc ? llvm::StringRef("<unknown>")
                                  : llvm::StringRef(cwd)));
      ftdLogStep(mlog, llvm::Twine("[pass] FTD_LOG_DIR=") +
                           (logDir ? logDir : "<unset>"));
    }

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    // Capture CFG topology before conversion flattens everything.
    ftdLogStep(mlog, "[pass] captureAllCFGTopologies: call");
    auto cfgTopologies = captureAllCFGTopologies(modOp, mlog);
    ftdLogStep(mlog, "[pass] captureAllCFGTopologies: returned");

    // Pre-touch the analyses individually so that, if one of them is the source
    // of the crash, the log pinpoints which. getAnalysis<> caches its result in
    // the AnalysisManager, so the patterns.add<> call below reuses these exact
    // instances; this only fixes the (otherwise unspecified) construction order
    // and moves it a few lines earlier — it does not change what is computed.
    ftdLogStep(mlog, "[pass] getAnalysis<ControlDependenceAnalysis>: start");
    (void)getAnalysis<ControlDependenceAnalysis>();
    ftdLogStep(mlog, "[pass] getAnalysis<ControlDependenceAnalysis>: done");

    ftdLogStep(mlog, "[pass] getAnalysis<gsa::GSAAnalysis>: start");
    (void)getAnalysis<gsa::GSAAnalysis>();
    ftdLogStep(mlog, "[pass] getAnalysis<gsa::GSAAnalysis>: done");

    ftdLogStep(mlog, "[pass] getAnalysis<NameAnalysis>: start");
    (void)getAnalysis<NameAnalysis>();
    ftdLogStep(mlog, "[pass] getAnalysis<NameAnalysis>: done");

    ftdLogStep(mlog, "[pass] building FtdLowerFuncToHandshake pattern");
    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<ControlDependenceAnalysis>(),
        getAnalysis<gsa::GSAAnalysis>(), getAnalysis<NameAnalysis>(), converter,
        ctx);

    ftdLogStep(mlog, "[pass] building one-to-one conversion patterns");
    patterns.add<
        // LowerFuncToHandshake,
        /*ConvertConstants,*/ AllocaOpConversion, ConvertCalls,
        /*ConvertUndefinedValues,*/ GetGlobalOpConversion, GlobalOpConversion,
        ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
        ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
        OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
        OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
        OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
        OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
        OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
        OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
        OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
        OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
        OneToOneConversion<arith::RemSIOp, handshake::RemSIOp>,
        OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
        OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
        OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
        OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
        OneToOneConversion<arith::MaxSIOp, handshake::MaxSIOp>,
        OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
        OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
        OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
        OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
        OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
        OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
        OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
        OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
        OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
        OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
        OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
        OneToOneConversion<arith::TruncFOp, handshake::TruncFOp>,
        OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>,
        OneToOneConversion<arith::SIToFPOp, handshake::SIToFPOp>,
        OneToOneConversion<arith::UIToFPOp, handshake::UIToFPOp>,
        OneToOneConversion<arith::FPToSIOp, handshake::FPToSIOp>,
        OneToOneConversion<arith::ExtFOp, handshake::ExtFOp>,
        OneToOneConversion<math::AbsFOp, handshake::AbsFOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);
    ftdLogStep(mlog, "[pass] patterns built");

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    target.addDynamicallyLegalOp<func::CallOp>([](func::CallOp op) {
      // If the call is to __init, consider it legal for now.
      // This allows the pass to continue so that __placeholder conversion can
      // later erase these __init calls.
      // All other func.CallOp (not calling __init) remain illegal due to the
      // addIllegalDialect rule above and must be converted by a pattern.
      if (auto calledFn = dyn_cast_or_null<func::FuncOp>(
              SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()))) {
        return calledFn.getSymName().startswith("__init");
      }
      // If symbol lookup fails or it's not a func::FuncOp, treat as default
      // (illegal)
      return false;
    });
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return op.getSymName().startswith("__init"); });

    ftdLogStep(mlog, "[pass] applyFullConversion: start");
    if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
      ftdLogStep(mlog, "[pass] applyFullConversion: FAILED (signalPassFailure)");
      return signalPassFailure();
    }
    ftdLogStep(mlog, "[pass] applyFullConversion: done");

    // Clean up: Remove the definition of each __init* function, but only if it
    // has no remaining uses. This is safe because all valid calls to __init*
    // were tracked and deleted earlier.
    ftdLogStep(mlog, "[pass] __init cleanup: start");
    for (auto func : llvm::make_early_inc_range(modOp.getOps<func::FuncOp>())) {
      if (func.getSymName().startswith("__init")) {
        assert(func.use_empty() &&
               "__init function should not have users after transformation");
        func.erase();
      }
    }
    ftdLogStep(mlog, "[pass] __init cleanup: done");

    ftdLogStep(mlog, "[pass] post-conversion FTD loop: start");
    for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
      mlir::OpBuilder builder(funcOp.getContext());

      auto topoIt = cfgTopologies.find(funcOp.getName());
      if (topoIt == cfgTopologies.end()) {
        ftdLogStep(mlog, "[pass]   func " + funcOp.getName() +
                             ": no topology, skipping");
        continue;
      }
      const OriginalCFGInfo &info = topoIt->second;

      if (info.numBlocks <= 1) {
        ftdLogStep(mlog, "[pass]   func " + funcOp.getName() +
                             ": numBlocks<=1, skipping");
        continue;
      }

      ftdLogStep(mlog, "[pass]   func " + funcOp.getName() + ": processing");

      auto logOwner = openFtdLog(funcOp.getName());
      llvm::raw_ostream *log = logOwner.get();
      ftdLogStep(log, "[cf2hs] ==== post-conversion FTD: " + funcOp.getName() +
                          " (blocks=" + llvm::Twine(info.numBlocks) + ") ====");

      // Build the shadow CFG — one struct, everything inside.
      ftd::ShadowCFG shadow = buildShadowCFG(builder, funcOp, info, log);
      ftdLogStep(log, "[cf2hs] buildShadowCFG done");

      // Route the select of every conditional block's terminator branch
      // through that block's condition placeholder.
      DenseMap<unsigned, Value> condPlaceholderByBB;
      for (auto cstOp : funcOp.getOps<handshake::ConstantOp>()) {
        if (!cstOp->hasAttr("ftd.cvar"))
          continue;
        if (auto bbAttr = cstOp->getAttrOfType<IntegerAttr>("handshake.bb"))
          condPlaceholderByBB[bbAttr.getUInt()] = cstOp.getResult();
      }
      ftdLogStep(log, "[cf2hs] collected cond placeholders (count=" +
                          llvm::Twine(condPlaceholderByBB.size()) + ")");
      for (auto brOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
        if (brOp->hasAttr("ftd.skip"))
          continue;
        auto bbAttr = brOp->getAttrOfType<IntegerAttr>("handshake.bb");
        if (!bbAttr)
          continue;
        auto it = condPlaceholderByBB.find(bbAttr.getUInt());
        if (it == condPlaceholderByBB.end())
          continue;
        // operand 0 of a ConditionalBranchOp is the condition (select).
        brOp->setOperand(0, it->second);
      }
      ftdLogStep(log, "[cf2hs] cond placeholder routing done");

      ftd::resolveCondPlaceholders(funcOp, builder, shadow);
      ftdLogStep(log, "[cf2hs] resolveCondPlaceholders done");

      // Populate conditionMap from NotIOp placeholders
      for (auto notOp : funcOp.getOps<handshake::NotIOp>()) {
        if (!notOp->hasAttr("ftd.cvar"))
          continue;
        auto bbAttr = notOp->getAttrOfType<IntegerAttr>("handshake.bb");
        if (!bbAttr)
          continue;
        shadow.conditionMap[bbAttr.getUInt()] = notOp.getResult();
      }
      ftdLogStep(log, "[cf2hs] conditionMap populated");

      ftdLogStep(log, "[cf2hs] addRegen start");
      ftd::addRegen(funcOp, builder, shadow);
      ftdLogStep(log, "[cf2hs] addRegen done");

      ftdLogStep(log, "[cf2hs] addSupp start");
      ftd::addSupp(funcOp, builder, shadow);
      ftdLogStep(log, "[cf2hs] addSupp done");

      ftd::finalizeCondPlaceholders(funcOp);
      ftdLogStep(log, "[cf2hs] finalizeCondPlaceholders done");

      shadow.destroy();
      ftdLogStep(log, "[cf2hs] shadow.destroy done");

      ftdLogStep(mlog, "[pass]   func " + funcOp.getName() + ": done");
    }
    ftdLogStep(mlog, "[pass] post-conversion FTD loop: done");
    ftdLogStep(mlog, "[pass] ==== runDynamaticPass complete ====");
  }
};
} // namespace

/// Converts undefined operations (LLVM::UndefOp) with a default "0"
/// constant triggered by the start signal of the corresponding function.
/// This is usually associated to uninitialized variables in the code
static LogicalResult convertUndefinedValues(ConversionPatternRewriter &rewriter,
                                            handshake::FuncOp &funcOp,
                                            NameAnalysis &namer) {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each undefined value
  auto undefinedValues = funcOp.getBody().getOps<LLVM::UndefOp>();

  for (auto undefOp : undefinedValues) {
    // Create an attribute of the appropriate type for the constant
    auto resType = undefOp.getRes().getType();
    TypedAttr cstAttr;
    if (isa<IndexType>(resType)) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    } else if (isa<IntegerType>(resType)) {
      cstAttr = rewriter.getIntegerAttr(resType, 0);
    } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
      cstAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    }

    // Create a constant with a default value and replace the undefined value
    rewriter.setInsertionPoint(undefOp);
    auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(),
                                                        cstAttr, startValue);
    cstOp->setDialectAttrs(undefOp->getAttrDictionary());
    undefOp.getResult().replaceAllUsesWith(cstOp.getResult());
    namer.replaceOp(cstOp, cstOp);
    rewriter.replaceOp(undefOp, cstOp.getResult());
  }

  return success();
}

/// Determines whether it is possible to transform an arith-level constant into
/// a Handshake-level constant that is triggered by an always-triggering source
/// component without compromising the circuit semantics (e.g., without
/// triggering a memory operation before the circuit "starts"). Returns false if
/// the Handshake-level constant that replaces the input must instead be
/// connected to the control-only network; returns true otherwise. This function
/// assumes that the rest of the std-level operations have already been
/// converted to their Handshake equivalent.
/// NOTE: I doubt this works in half-degenerate cases, but this is the logic
/// that legacy Dynamatic follows.
static bool isCstSourcable(arith::ConstantOp cstOp) {
  std::function<bool(Operation *)> isValidUser = [&](Operation *user) -> bool {
    if (isa<UnrealizedConversionCastOp>(user))
      return llvm::all_of(user->getUsers(), isValidUser);
    return !isa<handshake::BranchOp, handshake::ConditionalBranchOp,
                handshake::LoadOp, handshake::StoreOp>(user);
  };

  return llvm::all_of(cstOp->getUsers(), isValidUser);
}

/// Convers arith-level constants to handshake-level constants. Constants are
/// triggered by the start value of the corresponding function. The FTD
/// algorithm is then in charge of connecting the constants to the rest of the
/// network, in order for them to be re-generated
static LogicalResult convertConstants(ConversionPatternRewriter &rewriter,
                                      handshake::FuncOp &funcOp,
                                      NameAnalysis &namer) {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();
  llvm::DenseMap<Block *, Value> sourcesPerBlock;

  // For each constant
  auto constants = funcOp.getBody().getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : constants) {

    rewriter.setInsertionPoint(cstOp);

    // This variable will work as activation value for the constant. If the
    // constant is considered as sourcable, this will be the output of a source
    // component, otherwise it remains startValue
    Value controlValue;
    if (isCstSourcable(cstOp)) {
      auto sourceOp = rewriter.create<handshake::SourceOp>(cstOp.getLoc());
      inheritBB(cstOp, sourceOp);
      controlValue = sourceOp.getResult();
    } else {
      controlValue = startValue;
    }

    // Continue the conversion by obtaining the size of the constnat
    TypedAttr valueAttr = cstOp.getValue();

    if (isa<IndexType>(valueAttr.getType())) {
      auto intType = rewriter.getIntegerType(32);
      valueAttr = IntegerAttr::get(
          intType, cast<IntegerAttr>(valueAttr).getValue().trunc(32));
    }

    auto newCstOp = rewriter.create<handshake::ConstantOp>(
        cstOp.getLoc(), valueAttr, controlValue);

    newCstOp->setDialectAttrs(cstOp->getDialectAttrs());

    // Replace the constant and the usage of its result
    namer.replaceOp(cstOp, newCstOp);
    cstOp.getResult().replaceAllUsesWith(newCstOp.getResult());
    rewriter.replaceOp(cstOp, newCstOp->getResults());
  }
  return success();
}

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

LogicalResult ftd::FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {
  auto logOwner = openFtdLog(lowerFuncOp.getSymName());
  llvm::raw_ostream *log = logOwner.get();
  ftdLogStep(log, "[cf2hs] >>> matchAndRewrite: " + lowerFuncOp.getSymName());

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }
  ftdLogStep(log, "[cf2hs] memrefToArgIdx built (memrefs=" +
                      llvm::Twine(memrefToArgIdx.size()) + ")");

  ftd::createAllCondPlaceholders(lowerFuncOp.getRegion(), rewriter);
  ftdLogStep(log, "[cf2hs] createAllCondPlaceholders done");

  // Structure used inside addGsaGates to temporarily map a cf value to a
  // backedge until the proper handshake values are created; in which case, the
  // backedge is replaced with the corresponding hanshake values
  static DenseMap<Value, SmallVector<Backedge, 2>> pendingMuxOperands;

  // Add the muxes as obtained by the GSA analysis pass. This requires the
  // start value, as init merges need it as one of their output. However,
  // the start value is not available yet here, so a backedge is adopted
  // instead.
  BackedgeBuilder edgeBuilderStart(rewriter, lowerFuncOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());
  ftdLogStep(log, "[cf2hs] addGsaGates start");
  if (failed(addGsaGates(lowerFuncOp.getRegion(), rewriter, gsaAnalysis,
                         startValueBackedge, &pendingMuxOperands))) {
    ftdLogStep(log, "[cf2hs] addGsaGates FAILED");
    return failure();
  }
  ftdLogStep(log, "[cf2hs] addGsaGates done");

  // First lower the parent function itself, without modifying its body
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure)) {
    ftdLogStep(log, "[cf2hs] lowerSignature FAILED");
    return failure();
  }
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal()) {
    ftdLogStep(log, "[cf2hs] funcOp external, early success");
    return success();
  }
  ftdLogStep(log, "[cf2hs] lowerSignature done");

  // When GSA-MU functions are translated into multiplexers, an `init merge`
  // is created to feed them. This merge requires the start value of the
  // function as one of its data inputs. However, the start value was not
  // present yet when `addGsaGates` is called, thus we need to reconnect
  // it.
  startValueBackedge.setValue((Value)funcOp.getArguments().back());

  for (auto &[originalValue, backedges] : pendingMuxOperands) {
    Value newVal = rewriter.getRemappedValue(originalValue);
    assert(newVal && "Failed to remap GSA mux operand!");

    for (Backedge &be : backedges)
      be.setValue(newVal);
  }
  pendingMuxOperands.clear();
  ftdLogStep(log, "[cf2hs] GSA mux operands resolved");

  // Stores mapping from each value that passes through a merge-like
  // operation to the data result of that merge operation
  ArgReplacements argReplacements;

  // Currently, the following 2 functions do nothing but construct the network
  // of CMerges in complete isolation from the rest of the components
  // implementing the operations
  // In particular, the addMergeOps relies on adding Merges for every block
  // argument but because we removed all "real" arguments, we are only left
  // with the Start value as an argument for every block
  addMergeOps(funcOp, rewriter, argReplacements);
  ftdLogStep(log, "[cf2hs] addMergeOps done");
  addBranchOps(funcOp, rewriter);
  ftdLogStep(log, "[cf2hs] addBranchOps done");

  // addBranchOps only creates handshake::ConditionalBranchOp for live-out
  // values.  If a conditional block has no live-outs, no ConditionalBranchOp
  // is created and the condition value is lost after flattening.
  // Create one using the block's control signal so that buildShadowCFG's
  // walk can always find the condition.
  for (Block &block : funcOp) {
    auto condBr = dyn_cast<cf::CondBranchOp>(block.getTerminator());
    if (!condBr)
      continue;

    bool hasHandshakeCondBr = llvm::any_of(block, [](Operation &op) {
      return isa<handshake::ConditionalBranchOp>(&op);
    });

    if (!hasHandshakeCondBr) {
      Value cond = rewriter.getRemappedValue(condBr.getCondition());
      assert(cond && "Failed to remap condition");
      Value ctrl = block.getArguments().back();
      rewriter.setInsertionPoint(condBr);
      rewriter.create<handshake::ConditionalBranchOp>(condBr.getLoc(), cond,
                                                      ctrl);
    }
  }
  ftdLogStep(log, "[cf2hs] cond-branch fixup done");

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo))) {
    ftdLogStep(log, "[cf2hs] convertMemoryOps FAILED");
    return failure();
  }
  ftdLogStep(log, "[cf2hs] convertMemoryOps done");

  // First round of bb-tagging so that newly inserted Dynamatic memory ports
  // get tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);
  ftdLogStep(log, "[cf2hs] idBasicBlocks #1 done");

  // Create the memory interface according to the algorithm from FPGA'23. This
  // functions introduce new data dependencies that are then passed to FTD for
  // correctly delivering data between them like any real data dependencies
  if (failed(verifyAndCreateMemInterfaces(funcOp, rewriter, memInfo))) {
    ftdLogStep(log, "[cf2hs] verifyAndCreateMemInterfaces FAILED");
    return failure();
  }
  ftdLogStep(log, "[cf2hs] verifyAndCreateMemInterfaces done");

  // Convert the constants and undefined values from the `arith` dialect to
  // the `handshake` dialect, while also using the start value as their
  // control value
  if (failed(convertConstants(rewriter, funcOp, namer)) ||
      failed(convertUndefinedValues(rewriter, funcOp, namer))) {
    ftdLogStep(log, "[cf2hs] constants/undef conversion FAILED");
    return failure();
  }
  ftdLogStep(log, "[cf2hs] constants/undef converted");

  // id basic block
  idBasicBlocks(funcOp, rewriter);
  ftdLogStep(log, "[cf2hs] idBasicBlocks #2 done");

  // Annotate the IR with the CFG information
  cfg::annotateCFG(funcOp, rewriter, namer);
  ftdLogStep(log, "[cf2hs] annotateCFG done");

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements))) {
    ftdLogStep(log, "[cf2hs] flattenAndTerminate FAILED");
    return failure();
  }
  ftdLogStep(log, "[cf2hs] flattenAndTerminate done");

  ftdLogStep(log, "[cf2hs] <<< matchAndRewrite done");
  return success();
}
