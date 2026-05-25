//===- FtdSuppression.cpp - Suppression Infrastructure ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the suppression infrastructure for the Fast Token Delivery (FTD)
// algorithm: local CFG construction, decision graph extraction, cyclic
// analysis, BDD-to-circuit conversion, token distribution, and the main
// insertDirectSuppression entry point.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FtdSuppression.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::ftd;
using namespace dynamatic::experimental::boolean;


// ===--------------------------------------------------------------------=== //
// CyclicGraphManager implementation
// ===--------------------------------------------------------------------=== //

CyclicGraphManager::CyclicGraphManager(LocalCFG &cfg)
    : lcfg(cfg), domInfo(cfg.containerOp),
      loopInfo(domInfo.getDomTree(cfg.region)) {
  analyzeTopology();
}

unsigned CyclicGraphManager::getNestingLevel(Block *bb) const {
  auto it = blockLevelMap.find(bb);
  if (it != blockLevelMap.end())
    return it->second;
  return 0;
}

std::unique_ptr<LoopScope>
CyclicGraphManager::buildScopeRecursive(mlir::CFGLoop *loop, unsigned level) {
  auto scope = std::make_unique<LoopScope>();
  scope->level = level;
  scope->loopInfo = loop;
  scope->header = loop->getHeader();

  SmallVector<Block *> latches;
  loop->getLoopLatches(latches);
  scope->latches = latches;

  for (Block *latch : latches) {
    scope->allBackEdges.insert({latch, scope->header});
  }

  auto loopBlocks = loop->getBlocks();
  for (Block *b : loopBlocks) {
    scope->allBlocksInclusive.insert(b);
    // Outer loops must be processed before inner loops for correct leveling.
    blockLevelMap[b] = level;
  }

  for (auto *subLoop : loop->getSubLoops()) {
    auto subScope = buildScopeRecursive(subLoop, level + 1);
    subScope->parent = scope.get();
    for (auto &edge : subScope->allBackEdges) {
      scope->allBackEdges.insert(edge);
    }
    scope->subLoops.push_back(std::move(subScope));
  }
  return scope;
}

void CyclicGraphManager::analyzeTopology() {
  blockLevelMap.clear();
  topLevelScope = std::make_unique<LoopScope>();
  topLevelScope->level = 0;
  topLevelScope->header = lcfg.newProd;
  topLevelScope->loopInfo = nullptr;

  for (Block &b : lcfg.region->getBlocks()) {
    topLevelScope->allBlocksInclusive.insert(&b);
    blockLevelMap[&b] = 0;
  }

  auto topLoops = loopInfo.getTopLevelLoops();
  for (auto *loop : topLoops) {
    auto subScope = buildScopeRecursive(loop, 1);
    subScope->parent = topLevelScope.get();
    for (auto &edge : subScope->allBackEdges) {
      topLevelScope->allBackEdges.insert(edge);
    }
    topLevelScope->subLoops.push_back(std::move(subScope));
  }
}

std::unique_ptr<LocalCFG>
CyclicGraphManager::extractLayeredCFG(const LoopScope *scope,
                                      OpBuilder &builder) {
  auto newGraph = std::make_unique<LocalCFG>();
  Location loc = builder.getUnknownLoc();

  OpBuilder::InsertionGuard guard(builder);
  auto funcType = builder.getFunctionType({}, {});
  auto dummyFunc =
      builder.create<func::FuncOp>(loc, "__ftd_layered_cfg__", funcType);
  Region &R = dummyFunc.getBody();
  newGraph->region = &R;
  newGraph->containerOp = dummyFunc;

  // Create the sink terminal (False).
  // For level > 0, back-edges to own header are redirected here.
  // All terminal paths eventually converge here.
  Block *falseTerm = new Block();
  R.push_back(falseTerm);
  newGraph->sinkBB = falseTerm;
  newGraph->origMap[falseTerm] = nullptr;

  // Create the consumer terminal (True).
  // For level > 0 this represents the loop exit.
  // For level 0 this maps to the actual consumer in the original CFG.
  Block *trueTerm = new Block();
  R.push_back(trueTerm);
  newGraph->newCons = trueTerm;

  if (scope->level == 0)
    newGraph->origMap[trueTerm] = lcfg.origMap.lookup(lcfg.newCons);
  else
    newGraph->origMap[trueTerm] = nullptr;

  // Clone the header block as the entry (Producer) of this layer.
  Block *clonedHeader = new Block();
  R.push_back(clonedHeader);
  newGraph->newProd = clonedHeader;
  newGraph->origMap[clonedHeader] = lcfg.origMap.lookup(scope->header);

  // Build a mapping from original blocks to their cloned counterparts.
  DenseMap<Block *, Block *> clonedMap;
  clonedMap[scope->header] = clonedHeader;

  // Build topo position map from the input graph for residual back-edge
  // detection (handles irreducible cycles not captured by CFGLoopInfo).
  DenseMap<Block *, unsigned> inputTopoPos;
  for (unsigned i = 0; i < lcfg.topoOrder.size(); ++i)
    inputTopoPos[lcfg.topoOrder[i]] = i;

  // At level 0, map the original consumer and sink to the terminals
  // so that in-scope edges targeting them are resolved correctly.
  if (scope->level == 0) {
    clonedMap[lcfg.newCons] = trueTerm;
    clonedMap[lcfg.sinkBB] = falseTerm;
  }

  // Clone all in-scope blocks except those already handled above.
  for (Block *b : scope->allBlocksInclusive) {
    if (b == scope->header || b == lcfg.newCons || b == lcfg.sinkBB)
      continue;
    // Skip the artificial dummyStart block created by buildDecisionGraph.
    // It has no predecessors (it IS the region entry), unconditionally
    // branches to scope->header (newProd), and carries no decision logic.
    // Cloning it would create an orphan that ends up as region.front(),
    // corrupting downstream CDA which uses region.front() as its entry.
    if (b->hasNoPredecessors() && b != scope->header)
      continue;
    Block *nb = new Block();
    R.push_back(nb);
    clonedMap[b] = nb;
    newGraph->origMap[nb] = lcfg.origMap.lookup(b);
  }

  // Reconstruct edges and terminators for each cloned block.
  for (auto [origBlock, newBlock] : clonedMap) {
    // Skip terminal blocks; they receive their terminators below.
    if (origBlock == lcfg.newCons || origBlock == lcfg.sinkBB)
      continue;

    builder.setInsertionPointToEnd(newBlock);
    Operation *origTerm = origBlock->getTerminator();

    if (!origTerm) {
      llvm::errs() << "Warning: Block without terminator found in LocalCFG "
                      "extraction.\n";
      continue;
    }

    SmallVector<Block *, 2> validSuccessors;
    SmallVector<bool, 2> keepSuccessor;

    for (unsigned i = 0; i < origTerm->getNumSuccessors(); ++i) {
      Block *origSucc = origTerm->getSuccessor(i);

      if (origSucc == scope->header) {
        // Back-edge to this scope's own header.
        // Level 0: header is dummystart; no real back-edge should exist,
        // but cut defensively if encountered.
        // Level > 0: redirect to the sink terminal.
        if (scope->level == 0) {
          validSuccessors.push_back(nullptr);
          keepSuccessor.push_back(false);
        } else {
          validSuccessors.push_back(falseTerm);
          keepSuccessor.push_back(true);
        }
      } else if (scope->allBackEdges.contains({origBlock, origSucc})) {
        // Deep back-edge belonging to an inner loop.
        // Always cut; the inner loop's own layer handles it.
        validSuccessors.push_back(nullptr);
        keepSuccessor.push_back(false);
      } else if (clonedMap.count(origSucc)) {
        // Standard in-scope edge.
        // Check for residual back-edges (irreducible cycles) using the
        // input graph's topological order. These are not natural loop
        // back-edges and thus not in allBackEdges, but still form cycles.
        if (inputTopoPos.count(origSucc) && inputTopoPos.count(origBlock) &&
            inputTopoPos[origSucc] <= inputTopoPos[origBlock]) {
          // Residual back-edge — redirect to sink.
          validSuccessors.push_back(falseTerm);
          keepSuccessor.push_back(true);
        } else {
          validSuccessors.push_back(clonedMap[origSucc]);
          keepSuccessor.push_back(true);
        }
      } else {
        // Edge leaving the current scope.
        if (scope->level == 0) {
          llvm::errs()
              << "[FTD Warning] Block outside Level 0 scope encountered.\n";
        }
        // Redirect to the consumer terminal (represents loop exit).
        validSuccessors.push_back(trueTerm);
        keepSuccessor.push_back(true);
      }
    }

    // Rebuild the terminator based on surviving successors.
    if (auto cbr = dyn_cast<cf::CondBranchOp>(origTerm)) {
      if (keepSuccessor[0] && keepSuccessor[1]) {
        // Both paths survived; keep conditional branch.
        Value cond = builder.create<arith::ConstantIntOp>(loc, 1, 1);
        builder.create<cf::CondBranchOp>(
            loc, cond, validSuccessors[0], validSuccessors[1]);
      } else if (keepSuccessor[0] && !keepSuccessor[1]) {
        // Only the true path survived.
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
      } else if (!keepSuccessor[0] && keepSuccessor[1]) {
        // Only the false path survived.
        builder.create<cf::BranchOp>(loc, validSuccessors[1]);
      }
      // If neither survived the block is left without a terminator (dead).
    } else if (auto br = dyn_cast<cf::BranchOp>(origTerm)) {
      if (keepSuccessor[0]) {
        builder.create<cf::BranchOp>(loc, validSuccessors[0]);
      }
      // If the sole successor was cut, no terminator is created (dead).
    }
  }

  builder.setInsertionPointToEnd(trueTerm);
  builder.create<cf::BranchOp>(loc, falseTerm);
  builder.setInsertionPointToEnd(falseTerm);
  builder.create<func::ReturnOp>(loc);

  // Graph simplification.
  //
  // Level 0: only remove dead blocks (those without a terminator that are
  // not terminals). This eliminates unreachable subgraphs created by
  // cutting deep back-edges. No duplicate-successor merging and no path
  // compression are performed; those are deferred to the caller.
  //
  // Level > 0: full canonicalization. After this pass every non-terminal
  // block should carry a conditional branch, which is what the BDD builder
  // expects.
  bool changed = true;
  while (changed) {
    changed = false;

    for (Block &block : llvm::make_early_inc_range(R)) {
      // Never touch the entry or the two terminal blocks.
      if (&block == newGraph->newProd || &block == newGraph->newCons ||
          &block == newGraph->sinkBB)
        continue;

      Operation *term = block.getTerminator();

      // Dead block removal (applies to all levels).
      // A block without a terminator has no outgoing edges and is dead.
      // Disconnect its predecessors so the removal can propagate.
      if (!term) {
        SmallVector<Operation *, 4> predTerms;
        DenseSet<Operation *> predTermSeen;
        for (auto &use : block.getUses())
          if (predTermSeen.insert(use.getOwner()).second)
            predTerms.push_back(use.getOwner());

        for (Operation *predTerm : predTerms) {
          OpBuilder localBuilder(predTerm->getContext());
          localBuilder.setInsertionPoint(predTerm);

          if (auto br = dyn_cast<cf::BranchOp>(predTerm)) {
            // Predecessor unconditionally jumps here; it becomes dead too.
            predTerm->erase();
          } else if (auto cbr = dyn_cast<cf::CondBranchOp>(predTerm)) {
            Block *trueDest = cbr.getTrueDest();
            Block *falseDest = cbr.getFalseDest();

            if (trueDest == &block && falseDest == &block) {
              // Both legs land on this dead block; predecessor dies.
              predTerm->erase();
            } else if (trueDest == &block) {
              // True leg is dead; degrade to unconditional false branch.
              localBuilder.create<cf::BranchOp>(loc, falseDest);
              predTerm->erase();
            } else if (falseDest == &block) {
              // False leg is dead; degrade to unconditional true branch.
              localBuilder.create<cf::BranchOp>(loc, trueDest);
              predTerm->erase();
            }
          }
        }

        block.erase();
        changed = true;
        continue;
      }

      // The remaining optimizations only apply to level > 0.
      if (scope->level == 0)
      // if (true)
        continue;

      // Merge duplicate successors: CondBranch(A, A) becomes Branch(A).
      if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          OpBuilder localBuilder(term->getContext());
          localBuilder.setInsertionPoint(term);
          localBuilder.create<cf::BranchOp>(loc, condBr.getTrueDest());
          term->erase();
          term = block.getTerminator();
          changed = true;
        }
      }

      // Path compression: if a block unconditionally jumps to a single
      // destination, redirect all predecessors directly to that destination
      // and remove the block.
      if (auto br = dyn_cast<cf::BranchOp>(term)) {
        Block *dest = br.getDest();
        block.replaceAllUsesWith(dest);
        block.erase();
        changed = true;
        continue;
      }
    }
  }

  // Compute topological order starting from the producer.
  DenseSet<Block *> visited;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visited.contains(u))
      return;
    visited.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(); it != term->successor_end(); ++it)
        topo(*it);
    newGraph->topoOrder.push_back(u);
  };

  topo(newGraph->newProd);
  std::reverse(newGraph->topoOrder.begin(), newGraph->topoOrder.end());

  // Erase orphan blocks: any block not reachable from newProd (i.e. not in
  // the visited set) is structural debris.  If left in the region it may
  // end up before newProd in the block list, becoming region.front() and
  // corrupting downstream analyses (CDA, dominance) that assume
  // region.front() is the entry.
  for (Block &block : llvm::make_early_inc_range(R)) {
    if (!visited.contains(&block)) {
      // Erase the terminator first to drop outgoing BlockOperand refs.
      if (Operation *term = block.getTerminator())
        term->erase();
      // The orphan is unreachable, so no other terminator references it.
      block.erase();
    }
  }

  // Physical reordering: ensure region.front() == newProd so that
  // downstream CDA (which starts DFS from region.front()) uses the
  // correct entry block.
  for (Block *b : newGraph->topoOrder) {
    if (b != newGraph->sinkBB) {
      b->moveBefore(newGraph->sinkBB);
    }
  }

  return newGraph;
}

// ===--------------------------------------------------------------------=== //
// SignalRegistry implementation
// ===--------------------------------------------------------------------=== //

void SignalRegistry::registerSignal(StringRef var, const PathContext &path,
                                    Value val) {
  map[var.str()].push_back({path, val});
}

Value SignalRegistry::lookup(StringRef var, const PathContext &queryPath) {
  std::string v = var.str();
  if (map.find(v) == map.end())
    return nullptr;

  Value bestMatch = nullptr;
  size_t bestLen = 0;
  bool foundAny = false;

  for (auto &entry : map[v]) {
    const PathContext &regPath = entry.first;
    if (regPath.size() > queryPath.size())
      continue;

    // Filter: The registered path must be a prefix of the query path
    // to ensure the signal lies on the same control flow path.
    bool isPrefix = true;
    for (size_t i = 0; i < regPath.size(); ++i) {
      if (regPath[i] != queryPath[i]) {
        isPrefix = false;
        break;
      }
    }

    // Selection: Choose the longest matching prefix (closest definition).
    if (isPrefix) {
      if (!foundAny || regPath.size() >= bestLen) {
        bestLen = regPath.size();
        bestMatch = entry.second;
        foundAny = true;
      }
    }
  }
  return bestMatch;
}


// ===--------------------------------------------------------------------=== //
// Static helpers (internal to this TU)
// ===--------------------------------------------------------------------=== //

/// Identify the block that has muxCondition as its terminator condition
/// Note that it is not necessarily the same block defining the muxCondition
static Block *returnMuxConditionBlock(Value muxCondition,
                                      ftd::ShadowCFG &shadow) {
  while (true) {
    Operation *defOp = muxCondition.getDefiningOp();
    if (!defOp)
      return shadow.getBlock(0);

    // Found the NotIOp placeholder — read its BB attribute.
    if (defOp->hasAttr(FTD_COND_VAR)) {
      auto bbAttr = defOp->getAttrOfType<IntegerAttr>("handshake.bb");
      assert(bbAttr && "Condition placeholder missing handshake.bb");
      return shadow.getBlock(bbAttr.getUInt());
    }

    // Trace backward through suppression branches (marked FTD_OP_TO_SKIP).
    if (isa<handshake::ConditionalBranchOp>(defOp) &&
        defOp->hasAttr(FTD_OP_TO_SKIP)) {
      muxCondition = defOp->getOperand(1);
      continue;
    }

    // Fallback: use handshake.bb on the defining op.
    if (auto bbAttr = defOp->getAttrOfType<IntegerAttr>("handshake.bb"))
      return shadow.getBlock(bbAttr.getUInt());

    return shadow.getBlock(0);
  }
}

static IntegerAttr getFirstLoopExitBBAttrIfHeaderConsumer(
    OpBuilder &builder, Region &region, Block *producerBlock, Block *consumerBlock,
    const ftd::BlockIndexing &bi, ftd::ShadowCFG &shadow) {
  if (!producerBlock || !consumerBlock || producerBlock == consumerBlock)
    return {};

  DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  CFGLoop *consumerLoop = loopInfo.getLoopFor(consumerBlock);
  if (!consumerLoop || consumerLoop->getHeader() != consumerBlock ||
      !consumerLoop->contains(producerBlock))
    return {};

  SmallVector<Block *, 4> exitBlocks;
  consumerLoop->getExitBlocks(exitBlocks);
  if (exitBlocks.empty())
    return {};

  llvm::sort(exitBlocks, [&](Block *lhs, Block *rhs) {
    return bi.isLess(lhs, rhs);
  });

  unsigned exitBBIdx = shadow.getBlockIndex(exitBlocks.front());
  return IntegerAttr::get(
      IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned),
      exitBBIdx);
}

/// Helper function to generate expression combining Local Logic (True/False)
/// with Original Variables (c2, c3...).
/// Note: Input is Local Deps
static boolean::BoolExpression *
getHybridPathExpression(const std::vector<Block *> &localPath,
                        const ftd::LocalCFG &lcfg, const ftd::BlockIndexing &bi,
                        const DenseSet<Block *> &localDeps) {

  // Start with 1
  auto *exp = boolean::BoolExpression::boolOne();

  unsigned pathSize = localPath.size();
  for (unsigned i = 0; i < pathSize - 1; i++) {
    // Local Block
    Block *u = localPath[i];
    // Local Block (Target)
    Block *v = localPath[i + 1];

    // 1. Check Dependency using LOCAL sets
    if (localDeps.contains(u)) {
      Operation *term = u->getTerminator();

      if (isa<cf::CondBranchOp>(term)) {
        // 2. Map to Original Block to get the Variable Index/Name
        Block *origU = lcfg.origMap.lookup(u);

        // Special handling for second visit if needed, though usually origMap
        // covers it
        if (!origU && u == lcfg.secondVisitBB) {
          origU = lcfg.origMap.lookup(lcfg.newProd);
        }

        if (origU) {
          // 3. Get Name from ORIGINAL CFG (e.g., "c2")
          auto blockIndexOptional = bi.getIndexFromBlock(origU);
          if (blockIndexOptional.has_value()) {
            std::string blockCondition = bi.getBlockCondition(origU);
            boolean::BoolExpression *cond =
                boolean::BoolExpression::parseSop(blockCondition);

            // 4. Get Polarity from LOCAL CFG (Structure)
            // Does the path go to 'v' via the False branch in the Decision
            // Graph?
            auto condOp = dyn_cast<cf::CondBranchOp>(term);
            if (condOp.getFalseDest() == v) {
              // Add '~' if Local Graph says False
              cond->boolNegate();
            }

            // Combine
            exp = boolean::BoolExpression::boolAnd(exp, cond);
          }
        }
      }
    }
  }
  return exp;
}

// ===--------------------------------------------------------------------=== //
// Path enumeration on LocalCFG
// ===--------------------------------------------------------------------=== //

BoolExpression *ftd::enumeratePaths(const ftd::LocalCFG &lcfg,
                                      const ftd::BlockIndexing &bi,
                                      const DenseSet<Block *> &controlDeps) {

  // 1. Path Finding using Iterative DFS (on Local CFG)
  std::vector<std::vector<Block *>> allPaths;

  struct StackFrame {
    Block *u;
    unsigned currIdx;
    unsigned numSuccs;
  };

  std::vector<StackFrame> dfsStack;
  std::vector<Block *> currentLocalPath;

  if (lcfg.newProd && lcfg.newCons) {
    Block *root = lcfg.newProd;
    auto *term = root->getTerminator();
    unsigned n = term ? term->getNumSuccessors() : 0;

    dfsStack.push_back({root, 0, n});
    currentLocalPath.push_back(root);
  } else {
    return BoolExpression::boolZero();
  }

  while (!dfsStack.empty()) {
    StackFrame &frame = dfsStack.back();

    // Case A: Reached Consumer
    if (frame.u == lcfg.newCons) {
      // [CRITICAL] Store the LOCAL path exactly as traversed.
      // Do NOT map to original blocks here.
      allPaths.push_back(currentLocalPath);

      currentLocalPath.pop_back();
      dfsStack.pop_back();
      continue;
    }

    // Case B: Traverse Successors
    if (frame.currIdx < frame.numSuccs) {
      auto *term = frame.u->getTerminator();
      Block *succ = term->getSuccessor(frame.currIdx);
      frame.currIdx++;

      bool isCycle = std::find(currentLocalPath.begin(), currentLocalPath.end(),
                               succ) != currentLocalPath.end();

      if (succ != lcfg.sinkBB && !isCycle) {
        auto *succTerm = succ->getTerminator();
        unsigned succN = succTerm ? succTerm->getNumSuccessors() : 0;

        dfsStack.push_back({succ, 0, succN});
        currentLocalPath.push_back(succ);
      }
    }
    // Case C: Backtrack
    else {
      currentLocalPath.pop_back();
      dfsStack.pop_back();
    }
  }

  if (allPaths.empty())
    return BoolExpression::boolZero();

  // 2. Expression Generation
  BoolExpression *sop = BoolExpression::boolZero();

  for (const std::vector<Block *> &path : allPaths) {
    // Use the hybrid helper to look up logic locally and names globally
    BoolExpression *minterm =
        getHybridPathExpression(path, lcfg, bi, controlDeps);

    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}


// ===--------------------------------------------------------------------=== //
// BDD → Circuit conversion
// ===--------------------------------------------------------------------=== //

/// Retrieves the initial value from BlockIndexing.
static Value getOriginalValue(mlir::OpBuilder &builder, StringRef varName,
                              const ftd::BlockIndexing &bi,
                              DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
                              ftd::ShadowCFG *shadow = nullptr) {
  StringRef lookupName = varName;
  if (lookupName.startswith("~")) {
    llvm::errs() << "[FTD Error] Negated variable '" << varName << "'.\n";
    lookupName = lookupName.drop_front();
  }

  auto conditionOpt = bi.getBlockFromCondition(lookupName.str());
  if (!conditionOpt.has_value())
    return nullptr;

  Value condition;
  if (shadow) {
    // conditionOpt is the shadow Block* looked up by BlockIndexing.
    // Convert it to enumeration index for conditionMap lookup.
    Block *shadowBlock = conditionOpt.value();
    unsigned enumIdx = shadow->getBlockIndex(shadowBlock);
    condition = shadow->getCondition(enumIdx);
  } else {
    // Pre-flatten CfToHandshake path: use a SourceOp placeholder instead of
    // the real terminator condition. This avoids creating backedges for
    // condition values (which caused null-Value crashes in bddToCircuit).
    condition = getOrCreateCondPlaceholder(conditionOpt.value(), builder);
  }

  return condition;
}

/// Converts a boolean expression node to a circuit signal.
static Value boolExpressionToCircuit(
    mlir::OpBuilder &builder, experimental::boolean::BoolExpression *expr,
    Block *block, SignalRegistry &registry, const PathContext &currentPath,
    const ftd::BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ftd::ShadowCFG *shadow = nullptr, IntegerAttr forcedBBAttr = {}) {

  // Case 1: Variable
  if (expr->type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(expr);
    std::string varName = singleCond->id;

    // 1. Registry Lookup
    Value val = registry.lookup(varName, currentPath);

    // 2. Fallback
    if (!val) {
      val = getOriginalValue(builder, varName, bi, pendingMuxOperands, shadow);

      if (!val) {
        llvm::errs() << "[FTD Error] Variable '" << varName
                     << "' not found in Registry or BlockIndexing.\n";
        assert(val && "Signal missing from IR");
      }

      if (!val.getType().isa<handshake::ChannelType>()) {
        val.setType(ftd::channelifyType(val.getType()));
      }
    }

    // 3. Handle Negation
    if (singleCond->isNegated) {
      builder.setInsertionPointToStart(block);
      Location loc = block->getOperations().front().getLoc();
      Type chanI1 = ftd::channelifyType(builder.getI1Type());

      auto notOp = builder.create<handshake::NotIOp>(loc, chanI1, val);
      notOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      setBBAttrWithFallback(notOp, forcedBBAttr, block, builder);
      return notOp->getResult(0);
    }

    return val;
  }

  // Case 2: Constant
  auto sourceOp = builder.create<handshake::SourceOp>(block->front().getLoc());
  setBBAttr(sourceOp, block, builder);
  auto intType = builder.getIntegerType(1);
  int constVal = (expr->type == ExpressionType::One ? 1 : 0);
  auto cstAttr = builder.getIntegerAttr(intType, constVal);
  auto constOp = builder.create<handshake::ConstantOp>(
      block->front().getLoc(), cstAttr, sourceOp.getResult());
  constOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
  setBBAttrWithFallback(sourceOp, forcedBBAttr, block, builder);
  setBBAttrWithFallback(constOp, forcedBBAttr, block, builder);

  return constOp.getResult();
}

/// Recursively converts a BDD to a Mux Tree.
Value ftd::bddToCircuit(mlir::OpBuilder &builder, BDD *bdd, Block *block,
                          SignalRegistry &registry, PathContext currentPath,
                          const ftd::BlockIndexing &bi,
                          DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
                          ftd::ShadowCFG *shadow,
                          IntegerAttr forcedBBAttr) {
  using namespace experimental::boolean;

  // 1. Leaf Node
  if (!bdd->successors.has_value()) {
    return boolExpressionToCircuit(builder, bdd->boolVariable, block, registry,
                                   currentPath, bi, pendingMuxOperands, shadow,
                                   forcedBBAttr);
  }

  // 2. Mux Node
  std::string varName = bdd->boolVariable->toString();

  Value muxCond = registry.lookup(varName, currentPath);
  if (!muxCond) {
    muxCond = getOriginalValue(builder, varName, bi, pendingMuxOperands, shadow);
    assert(muxCond && "Mux condition not found");
    if (!muxCond.getType().isa<handshake::ChannelType>())
      muxCond.setType(ftd::channelifyType(muxCond.getType()));
  }

  SmallVector<Value> muxOperands;

  // Recursion: Update PathContext so downstream lookups find distributed
  // signals
  PathContext falsePath = currentPath;
  falsePath.push_back({varName, false});
  muxOperands.push_back(bddToCircuit(builder, bdd->successors.value().first,
                                     block, registry, falsePath, bi,
                                     pendingMuxOperands, shadow,
                                     forcedBBAttr));

  PathContext truePath = currentPath;
  truePath.push_back({varName, true});
  muxOperands.push_back(bddToCircuit(builder, bdd->successors.value().second,
                                     block, registry, truePath, bi,
                                     pendingMuxOperands, shadow,
                                     forcedBBAttr));

  auto muxOp = builder.create<handshake::MuxOp>(
      muxCond.getLoc(), muxOperands[0].getType(), muxCond, muxOperands);
  muxOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
  setBBAttrWithFallback(muxOp, forcedBBAttr, block, builder);

  return muxOp.getResult();
}


// ===--------------------------------------------------------------------=== //
// Token distribution
// ===--------------------------------------------------------------------=== //

/// Generates the Suppression Logic (Mux Tree) for a branch's select signal.
/// It constructs the logic for the "UNREACHABLE" condition.
/// F_suppress = NOT( OR( All Valid Paths ) )
static Value
generateReachabilityLogic(mlir::OpBuilder &builder, Block *block,
                          const std::vector<VariableRequirement> &requirements,
                          const PathContext &currentPath,
                          SignalRegistry &registry,
                          const ftd::BlockIndexing &bi, size_t startIndex,
                          DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
                          ftd::ShadowCFG *shadow = nullptr) {

  using namespace experimental::boolean;

  // 1. Construct Boolean Expression for Valid Paths
  BoolExpression *fValid = BoolExpression::boolZero();

  for (const auto &req : requirements) {
    BoolExpression *pathExpr = BoolExpression::boolOne();

    // Iterate through the path suffix starting from the current split point
    for (size_t i = startIndex; i < req.path.size(); ++i) {
      PathStep step = req.path[i];
      // Construct: SingleCond(Type, Name, Negated)
      BoolExpression *stepExpr =
          new SingleCond(ExpressionType::Variable, step.var, !step.value);
      pathExpr = BoolExpression::boolAnd(pathExpr, stepExpr);
    }
    fValid = BoolExpression::boolOr(fValid, pathExpr);
  }

  // 2. Compute Suppression Condition: F_suppress = NOT( F_valid )
  // We want the circuit to output TRUE when the path is INVALID.
  BoolExpression *fSuppress = fValid->boolNegate();
  fSuppress = fSuppress->boolMinimize();

  // 4. Build BDD and Circuit for Suppression Condition
  std::set<std::string> vars = fSuppress->getVariables();
  std::vector<std::string> cofactorList(vars.begin(), vars.end());

  // Sort cofactor list to match topological order (e.g., c0, then c1)
  // This ensures the Mux Tree structure matches the dependency order.
  std::sort(cofactorList.begin(), cofactorList.end(),
            [&](const std::string &a, const std::string &b) {
              auto idA = bi.getBlockFromCondition(a);
              auto idB = bi.getBlockFromCondition(b);
              if (!idA || !idB)
                return a < b;
              return bi.isLess(idA.value(), idB.value());
            });

  BDD *bdd = buildBDD(fSuppress, cofactorList);

  // Note: bddToCircuit uses registry.lookup. If the suppression logic involves
  // variables distributed earlier (like c3a), it will correctly find them.
  return bddToCircuit(builder, bdd, block, registry, currentPath, bi, pendingMuxOperands, shadow);
}

/// Recursively builds the Branch Tree.
static void
buildBranchTreeRecursive(mlir::OpBuilder &builder, StringRef currentVar,
                         std::vector<VariableRequirement> &requirements,
                         PathContext currentPath, SignalRegistry &registry,
                         const ftd::BlockIndexing &bi,
                         DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
                         ftd::ShadowCFG *shadow = nullptr) {

  // 1. Retrieve Data Signal
  // Look up the current data signal to be distributed from the registry using
  // the current path context.
  Value sourceVal = registry.lookup(currentVar, currentPath);
  assert(sourceVal && "Source value for distribution not found");

  // 2. Identify Split Variable
  // Key: {Variable Name, Value} -> Value: List of requirements.
  std::map<std::pair<std::string, bool>, std::vector<VariableRequirement>>
      groups;
  std::string splitVar = "";
  bool splitFound = false;

  size_t maxDepth = 0;
  for (const auto &req : requirements)
    maxDepth = std::max(maxDepth, req.path.size());

  // Scan forward to find the first point where requirements disagree on a
  // variable value.
  size_t scanDepth = currentPath.size();
  for (; scanDepth < maxDepth; ++scanDepth) {
    groups.clear();
    splitVar = "";

    // Simply collect the step at this depth.
    for (auto &req : requirements) {
      PathStep step = req.path[scanDepth];
      if (splitVar == "")
        splitVar = step.var;

      groups[{step.var, step.value}].push_back(req);
    }

    // Divergence found.
    if (groups.size() > 1) {
      splitFound = true;
      llvm::errs() << "[FTD] Split Variable Found: " << splitVar
                   << " at Depth: " << scanDepth << "\n";
      break;
    }
  }

  if (!splitFound)
    return;

  // 3. Retrieve Raw Select Signal
  // We need the physical control signal corresponding to the 'splitVar' found
  // above.
  Value conditionVal = registry.lookup(splitVar, currentPath);
  if (!conditionVal) {
    // Fallback: If not in registry, get the original value from the IR
    // (BlockIndexing).
    conditionVal = getOriginalValue(builder, splitVar, bi, pendingMuxOperands, shadow);
  }
  assert(conditionVal && "Splitter condition value not found");

  // Ensure Types are compatible with Handshake channels.
  if (!conditionVal.getType().isa<handshake::ChannelType>())
    conditionVal.setType(ftd::channelifyType(conditionVal.getType()));
  if (!sourceVal.getType().isa<handshake::ChannelType>())
    sourceVal.setType(ftd::channelifyType(sourceVal.getType()));

  // 4. Register Outputs and Recurse
  // [Context Backfilling]
  // Since we might have skipped several variables (Common Prefix) to reach
  // 'splitVar', we must fill these skipped steps back into the PathContext.
  // This ensures that the recursive call has a continuous path history, keeping
  // index alignment correct.
  PathContext baseNextPath = currentPath;
  if (!groups.empty()) {
    // Take the first requirement as a template to retrieve the skipped steps.
    const auto &repReq = groups.begin()->second.front();
    for (size_t k = currentPath.size(); k < scanDepth; ++k)
      baseNextPath.push_back({repReq.path[k].var, repReq.path[k].value});
  }

  // 5. [Suppression Logic]
  // Generate the logic to identify "Unreachable" or "Invalid" paths.
  // We pass 'scanDepth' (the index of splitVar) to the generator.
  // This tells the generator to check validity starting from the current split
  // variable, effectively ignoring the "Common Prefix" variables skipped in
  // Step 2 (which are implicitly valid). The logic checks the entire future
  // path to ensure reachability.
  Value suppressCondition = generateReachabilityLogic(
      builder, sourceVal.getParentBlock(), requirements, baseNextPath,
      registry, bi, scanDepth, pendingMuxOperands, shadow);

  if (!suppressCondition.getType().isa<handshake::ChannelType>())
    suppressCondition.setType(ftd::channelifyType(suppressCondition.getType()));

  // [Suppression Branch]
  // Acts as a filter:
  // If suppressCondition is TRUE (Invalid Path) -> Output to Sink (Discard
  // Token). If suppressCondition is FALSE (Valid Path)  -> Output to
  // 'activeSelectSignal' (Pass Token).
  SmallVector<Type> suppResultTypes = {conditionVal.getType(),
                                       conditionVal.getType()};
  auto suppBranch = builder.create<handshake::ConditionalBranchOp>(
      conditionVal.getLoc(), suppResultTypes, suppressCondition, conditionVal);
  suppBranch->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
  setBBAttr(suppBranch, conditionVal.getParentBlock(), builder);

  // False Output -> Active Select (Pass to Main Branch)
  Value activeSelectSignal = suppBranch.getFalseResult();

  // 6. [Distribution Logic] Main Branch
  SmallVector<Type> resultTypes = {sourceVal.getType(), sourceVal.getType()};

  // Create the branch that splits the 'sourceVal' based on the (possibly
  // filtered) 'activeSelectSignal'.
  auto branchOp = builder.create<handshake::ConditionalBranchOp>(
      sourceVal.getLoc(), resultTypes, activeSelectSignal, sourceVal);
  branchOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
  setBBAttr(branchOp, sourceVal.getParentBlock(), builder);

  Value trueResult = branchOp.getTrueResult();
  Value falseResult = branchOp.getFalseResult();

  // Handle the True branch recursion
  if (!groups[{splitVar, true}].empty()) {
    PathContext truePath = baseNextPath;
    truePath.push_back({splitVar, true});
    registry.registerSignal(currentVar, truePath, trueResult);
    buildBranchTreeRecursive(builder, currentVar, groups[{splitVar, true}],
                             truePath, registry, bi, pendingMuxOperands, shadow);
  }

  // Handle the False branch recursion
  if (!groups[{splitVar, false}].empty()) {
    PathContext falsePath = baseNextPath;
    falsePath.push_back({splitVar, false});
    registry.registerSignal(currentVar, falsePath, falseResult);
    buildBranchTreeRecursive(builder, currentVar, groups[{splitVar, false}],
                             falsePath, registry, bi, pendingMuxOperands, shadow);
  }
}

/// Main entry point of distribution logic.
void ftd::buildDistributionNetwork(mlir::OpBuilder &builder,
                                     const ftd::LocalCFG &lcfg,
                                     const ftd::BlockIndexing &bi,
                                     SignalRegistry &registry,
                                     DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
                                     ftd::ShadowCFG *shadow) {
  using namespace experimental::boolean;

  // 1. Collect Variable Requirements
  std::map<std::string, std::vector<VariableRequirement>> varNeeds;
  DenseSet<Block *> collectOnStack;
  std::function<void(Block *, PathContext)> collect = [&](Block *curr,
                                                          PathContext path) {
    // Stop recursion if reaching the consumer or the sink block
    if (curr == lcfg.newCons || curr == lcfg.sinkBB)
      return;

    // Cycle detection: stop if this block is already on the current DFS path
    if (!collectOnStack.insert(curr).second)
      return;

    // Find the condition variable
    Block *origBlock = lcfg.origMap.lookup(curr);
    std::string var = "";
    if (origBlock)
      var = bi.getBlockCondition(origBlock);

    // Record the variable requirement
    if (!var.empty()) {
      varNeeds[var].push_back({var, path});
    }

    auto *term = curr->getTerminator();
    if (!term)
      return;

    // Handle conditional branches.
    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      if (!var.empty()) {
        PathContext truePath = path;
        truePath.push_back({var, true});
        collect(condBr.getTrueDest(), truePath);

        PathContext falsePath = path;
        falsePath.push_back({var, false});
        collect(condBr.getFalseDest(), falsePath);
      } else {
        llvm::errs() << "[FTD ERROR] CondBranchOp encountered with empty "
                        "condition variable at block "
                     << origBlock << ". Successors will not be traversed.\n";
      }
    } else {
      // Handle unconditional branches
      for (Block *succ : term->getSuccessors()) {
        collect(succ, path);
      }
    }

    collectOnStack.erase(curr);
  };

  // Start collection from the producer block
  if (lcfg.newProd) {
    collect(lcfg.newProd, {});
  }

  // 2. Topological Sort
  std::vector<std::string> sortedVars;
  for (auto &kv : varNeeds)
    sortedVars.push_back(kv.first);

  std::sort(sortedVars.begin(), sortedVars.end(),
            [&](const std::string &a, const std::string &b) {
              auto idA = bi.getBlockFromCondition(a);
              auto idB = bi.getBlockFromCondition(b);
              if (!idA || !idB) {
                llvm::errs()
                    << "[FTD Warning] Variable missing from BlockIndexing: '"
                    << (idA ? b : a) << "'\n";
                return a < b;
              }
              return bi.isLess(idA.value(), idB.value());
            });

  // 3. Initial Registration and Construct Branch Trees
  for (const auto &var : sortedVars) {
    // Use pre-registered value if available (e.g. demoted high-level variable)
    Value rawVal = registry.lookup(var, {});
    if (!rawVal) {
      rawVal = getOriginalValue(builder, var, bi, pendingMuxOperands, shadow);
      if (!rawVal) {
        llvm::errs() << "[FTD Error] Variable '" << var
                     << "' not found in BlockIndexing during registration.\n";
        assert(rawVal && "Signal missing from IR");
      }
      if (!rawVal.getType().isa<handshake::ChannelType>())
        rawVal.setType(ftd::channelifyType(rawVal.getType()));
      registry.registerSignal(var, {}, rawVal);
    }
    if (varNeeds[var].size() > 1) {
      buildBranchTreeRecursive(builder, var, varNeeds[var], {}, registry,
                               bi, pendingMuxOperands, shadow);
    }
  }
}

// ===--------------------------------------------------------------------=== //
// expressionToCircuit — unified BDD pipeline
// ===--------------------------------------------------------------------=== //

DenseMap<Block *, unsigned> ftd::computeTopoRank(const LocalCFG &lcfg) {
  DenseMap<Block *, unsigned> rank;
  unsigned i = 0;
  for (Block *b : lcfg.topoOrder)
    if (auto *ob = lcfg.origMap.lookup(b))
      rank[ob] = i++;
  return rank;
}

Value ftd::expressionToCircuit(
    OpBuilder &builder, BoolExpression *expr,
    const DenseMap<Block *, unsigned> &varRank,
    Block *insertBlock, SignalRegistry &registry, const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow, IntegerAttr forcedBBAttr) {

  expr = expr->boolMinimize();
  std::set<std::string> vars = expr->getVariables();

  // Sort cofactors by topological rank
  std::vector<std::pair<unsigned, std::string>> tmp;
  tmp.reserve(vars.size());
  for (auto &var : vars)
    if (auto blkOpt = bi.getBlockFromCondition(var))
      if (varRank.count(*blkOpt))
        tmp.emplace_back(varRank.lookup(*blkOpt), var);
  llvm::sort(tmp, [](auto &a, auto &b) { return a.first < b.first; });

  std::vector<std::string> cofactorList;
  cofactorList.reserve(tmp.size());
  for (auto &p : tmp)
    cofactorList.push_back(p.second);

  BDD *bdd = buildBDD(expr, cofactorList);
  return bddToCircuit(builder, bdd, insertBlock, registry, {}, bi,
                      pendingMuxOperands, shadow, forcedBBAttr);
}

Value ftd::expressionToCircuit(
    OpBuilder &builder, BoolExpression *expr,
    const LocalCFG &rankSource,
    Block *insertBlock, SignalRegistry &registry, const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow, IntegerAttr forcedBBAttr) {
  DenseMap<Block *, unsigned> rank = computeTopoRank(rankSource);
  return expressionToCircuit(builder, expr, rank, insertBlock, registry, bi,
                             pendingMuxOperands, shadow, forcedBBAttr);
}

// ===--------------------------------------------------------------------=== //
// computeLoopBackedgeCondition — shared MU/regen pipeline
// ===--------------------------------------------------------------------=== //

Value ftd::computeLoopBackedgeCondition(
    OpBuilder &builder, Block *loopHeader, Block *insertBlock,
    const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow) {

  // 1. Build Local CFG with Prod = Cons = Loop Header (self-loop graph)
  OpBuilder tmpBuilder(builder.getContext());
  auto locGraph = buildLocalCFGRegion(tmpBuilder, loopHeader, loopHeader, bi);

  // 2. CDA on raw graph to get full control deps
  ControlDependenceAnalysis locCDATmp(*locGraph->region);
  DenseSet<Block *> locConsControlDepsFull =
      locCDATmp.getAllBlockDeps()[locGraph->newCons].allControlDeps;
  DenseSet<Block *> locConsAcyclicDeps =
      locCDATmp.getAllBlockDeps()[locGraph->newCons].allControlDeps;

  // 3. Build full decision graph for CyclicGraphManager
  auto fullDecisionGraph =
      buildDecisionGraph(*locGraph, locConsControlDepsFull);
  CyclicGraphManager cyclicMgr(*fullDecisionGraph);

  // 4. Build origToFullDG map
  DenseMap<Block *, Block *> origToFullDG;
  for (auto [dgBlock, origBlock] : fullDecisionGraph->origMap)
    if (origBlock)
      origToFullDG[origBlock] = dgBlock;

  // 5. Create cyclic demotion helper
  CyclicDemotionHelper demotionHelper(
      builder, builder.getContext(), bi, cyclicMgr, origToFullDG,
      builder.getUnknownLoc(), insertBlock, pendingMuxOperands, shadow);

  // 6. Build acyclic decision graph
  auto acyclicDG = buildDecisionGraph(*locGraph, locConsAcyclicDeps);

  // 7. Pre-register demoted values for high-level variables
  SignalRegistry registry;
  demotionHelper.preRegisterDemotedValues(acyclicDG, registry);

  // 8. Construct distribution circuit
  buildDistributionNetwork(builder, *acyclicDG, bi, registry,
                           pendingMuxOperands, shadow);

  // 9. CDA on acyclic DG
  ControlDependenceAnalysis locCDA(*acyclicDG->region);
  DenseSet<Block *> locConsControlDeps =
      locCDA.getAllBlockDeps()[acyclicDG->newCons].allControlDeps;

  // 10. Enumerate paths → expression → circuit
  BoolExpression *fBackedge = enumeratePaths(*acyclicDG, bi, locConsControlDeps);
  Value conditionValue =
      expressionToCircuit(builder, fBackedge, *acyclicDG, insertBlock,
                          registry, bi, pendingMuxOperands, shadow);

  // 11. Clean up temporary graphs
  acyclicDG->containerOp->erase();
  fullDecisionGraph->containerOp->erase();
  locGraph->containerOp->erase();

  return conditionValue;
}


// ===--------------------------------------------------------------------=== //
// LocalCFG and Decision Graph construction
// ===--------------------------------------------------------------------=== //

/// Build a local control-flow subgraph (LocalCFG) between a producer and
/// consumer. The subgraph is reconstructed as a region with unique entry
/// (producer) and exit (sink).
std::unique_ptr<LocalCFG>
ftd::buildLocalCFGRegion(OpBuilder &builder, Block *origProd, Block *origCons,
                    const ftd::BlockIndexing &bi) {
  auto L = std::make_unique<ftd::LocalCFG>();
  Location loc = builder.getUnknownLoc();

  // Setup Region Container
  OpBuilder::InsertionGuard guard(builder);
  auto funcType = builder.getFunctionType({}, {});
  auto dummyFunc =
      builder.create<func::FuncOp>(loc, "__ftd_local_cfg__", funcType);
  Region &R = dummyFunc.getBody();
  L->region = &R;
  L->containerOp = dummyFunc;

  // Sink Block: The unified exit for all paths (valid or suppressed).
  L->sinkBB = new Block();
  R.push_back(L->sinkBB);
  L->origMap[L->sinkBB] = nullptr;

  // Producer Block: The entry point of the local CFG.
  Block *entry = new Block();
  R.push_back(entry);
  L->newProd = entry;
  L->origMap[entry] = origProd;

  DenseMap<Block *, Block *> cloned;
  DenseSet<Block *> visited;
  // Avoid scheduling the same orig block twice
  DenseSet<Block *> enqueued;
  cloned[origProd] = entry;

  // DFS Function
  std::function<void(Block *, Block *)> dfs = [&](Block *currOrig,
                                                  Block *currNew) {
    visited.insert(currOrig);

    auto *term = currOrig->getTerminator();

    // Dead End: Implicit flow to Sink.
    if (!term || term->getNumSuccessors() == 0) {
      builder.setInsertionPointToEnd(currNew);
      builder.create<cf::BranchOp>(loc, L->sinkBB);
      return;
    }

    // LIST 1: The distinct successors in the NEW Local CFG for the current
    // block. Used to construct the BranchOp/CondBranchOp.
    SmallVector<Block *, 2> localSuccessors;

    // LIST 2: The successors that are valid and new, requiring further DFS
    // traversal. Stored as pairs: {Original Successor, New Local Block}.
    SmallVector<std::pair<Block *, Block *>, 2> successorsToVisit;

    for (auto it = term->successor_begin(), e = term->successor_end(); it != e;
         ++it) {
      Block *succOrig = *it;
      Block *nextBlockInLocalCFG =
          nullptr; // Where the edge points to in the new graph

      // Determine the edge destination based on rules

      // Case 1: Consumer Reached (Valid Delivery)
      if (succOrig == origCons) {
        if (succOrig == origProd) {
          // Self-loop delivery
          if (!L->secondVisitBB) {
            L->secondVisitBB = new Block();
            R.push_back(L->secondVisitBB);
            L->origMap[L->secondVisitBB] = nullptr;
            // Terminate SecondVisit immediately to Sink
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToEnd(L->secondVisitBB);
            builder.create<cf::BranchOp>(loc, L->sinkBB);
          }
          nextBlockInLocalCFG = L->secondVisitBB;
          L->newCons = L->secondVisitBB;
        } else {
          // Standard delivery
          if (!L->newCons || L->newCons == L->secondVisitBB) {
            Block *proxy = new Block();
            R.push_back(proxy);
            L->origMap[proxy] = succOrig;
            L->newCons = proxy;
            // Terminate Proxy immediately to Sink
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToEnd(proxy);
            builder.create<cf::BranchOp>(loc, L->sinkBB);
          }
          nextBlockInLocalCFG = L->newCons;
        }
      }
      // Case 2: Producer revisited, consumer not reached (Invalid)
      else if (succOrig == origProd) {
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Already cloned (discovered) but may not be visited yet.
      // Reuse the existing clone to avoid duplicating blocks for the same orig.
      else if (cloned.count(succOrig)) {
        nextBlockInLocalCFG = cloned[succOrig];
        // If this node hasn't been visited yet, ensure it will be traversed
        // once.
        if (!visited.count(succOrig) && !enqueued.count(succOrig)) {
          enqueued.insert(succOrig);
          successorsToVisit.push_back({succOrig, nextBlockInLocalCFG});
        }
      }
      // Case 3: Visited
      else if (visited.count(succOrig)) {
        // Normally unreachable now because cloned.count(succOrig) should hold
        // if it was ever visited through this builder. Keep as safety.
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Case 4: Invalid Back-edge
      else if (bi.isLess(succOrig, currOrig)) {
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Case 5: Valid Forward Edge (Continue Traversal)
      else {
        Block *newSucc = new Block();
        R.push_back(newSucc);
        cloned[succOrig] = newSucc;
        L->origMap[newSucc] = succOrig;

        nextBlockInLocalCFG = newSucc;
        // Schedule this node for DFS visitation (once)
        if (!enqueued.count(succOrig)) {
          enqueued.insert(succOrig);
          successorsToVisit.push_back({succOrig, newSucc});
        }
      }

      // Add the determined destination to the list of local successors
      localSuccessors.push_back(nextBlockInLocalCFG);
    }

    // Create the branch instruction
    builder.setInsertionPointToEnd(currNew);
    if (localSuccessors.size() == 1) {
      builder.create<cf::BranchOp>(loc, localSuccessors[0]);
    } else if (localSuccessors.size() == 2) {
      // Placeholder condition for 2-way branches
      Value cond = builder.create<arith::ConstantIntOp>(loc, 1, 1);
      builder.create<cf::CondBranchOp>(loc, cond, localSuccessors[0],
                                       localSuccessors[1]);
    } else {
      // Default fall-through for complex control flow
      builder.create<cf::BranchOp>(loc, L->sinkBB);
    }

    // Continue DFS
    for (auto &pair : successorsToVisit) {
      dfs(pair.first, pair.second);
    }
  };

  // Start DFS
  dfs(origProd, L->newProd);

  // Finalize Sink
  builder.setInsertionPointToEnd(L->sinkBB);
  builder.create<func::ReturnOp>(loc);

  if (!L->newCons)
    L->newCons = L->sinkBB;

  // Compute Topological Order
  DenseSet<Block *> visitedTopo;
  SmallVector<Block *> order;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visitedTopo.contains(u))
      return;
    visitedTopo.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(), e = term->successor_end();
           it != e; ++it)
        topo(*it);
    order.push_back(u);
  };

  topo(L->newProd);
  std::reverse(order.begin(), order.end());
  L->topoOrder = std::move(order);

  // Physical Reordering
  // Reorder blocks in the region list to match the topological order.
  // This does not change the graph structure (pointers), only the memory
  // layout/print order.
  for (Block *b : L->topoOrder) {
    if (b != L->sinkBB) {
      b->moveBefore(L->sinkBB);
    }
  }

  return L;
}

/// Constructs a NEW LocalCFG that represents the Decision Graph.
/// \param rawGraph The source LocalCFG.
/// \param dependencies The set of blocks (from rawGraph) that are relevant
/// decision nodes.
/// \param muxConstraints A map {Block* -> bool} enforcing specific values for
/// blocks. If a block is in this map, the branch corresponding to !value is
/// wired to Sink.
std::unique_ptr<LocalCFG> ftd::buildDecisionGraph(
    const ftd::LocalCFG &rawGraph, const DenseSet<Block *> &dependencies,
    const DenseMap<Block *, bool> &muxConstraints) {

  if (!rawGraph.newCons)
    return nullptr;

  // NodeSet: Consumer + Sink + Dependencies
  DenseSet<Block *> nodeSet;
  nodeSet.insert(rawGraph.newCons);
  nodeSet.insert(rawGraph.sinkBB);
  for (Block *b : dependencies) {
    nodeSet.insert(b);
  }

  // 2. Setup New Container
  auto newL = std::make_unique<ftd::LocalCFG>();
  OpBuilder builder(rawGraph.containerOp->getContext());
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto newContainer =
      builder.create<func::FuncOp>(loc, "__ftd_decision_graph__", funcType);
  Region &newRegion = newContainer.getBody();
  newL->region = &newRegion;
  newL->containerOp = newContainer;

  // 3. Create Blocks & Map
  // Create a dummy start block that ensures the entry block never has
  // back-edges.
  Block *dummyStart = new Block();
  newRegion.push_back(dummyStart);
  newL->origMap[dummyStart] = nullptr;

  DenseMap<Block *, Block *> oldToNew;

  for (Block *oldBlock : rawGraph.topoOrder) {
    if (nodeSet.contains(oldBlock)) {
      Block *newBlock = new Block();
      newRegion.push_back(newBlock);
      oldToNew[oldBlock] = newBlock;

      if (Block *origIR = rawGraph.origMap.lookup(oldBlock)) {
        newL->origMap[newBlock] = origIR;
      } else {
        newL->origMap[newBlock] = nullptr;
      }

      // Set newProd to the first valid block (TopoOrder)
      if (newL->newProd == nullptr) {
        newL->newProd = newBlock;
      }

      if (oldBlock == rawGraph.newCons)
        newL->newCons = newBlock;
      if (oldBlock == rawGraph.sinkBB)
        newL->sinkBB = newBlock;
      if (oldBlock == rawGraph.secondVisitBB)
        newL->secondVisitBB = newBlock;
    }
  }

  // Fallback for newProd (unlikely, as newCons/sinkBB are in nodeSet)
  if (newL->newProd == nullptr && newRegion.getBlocks().size() > 1) {
    newL->newProd = &newRegion.back();
  }

  // 4. Helper: Find Nearest using DFS with Visited Set
  auto findNearest = [&](Block *start) -> Block * {
    if (!start)
      return nullptr;
    DenseSet<Block *> visited;
    std::function<Block *(Block *)> dfs = [&](Block *curr) -> Block * {
      if (!curr)
        return nullptr;
      if (nodeSet.contains(curr))
        return curr;
      if (!visited.insert(curr).second)
        return nullptr; // Cycle
      for (Block *succ : curr->getSuccessors()) {
        if (Block *res = dfs(succ))
          return res;
      }
      return nullptr;
    };
    return dfs(start);
  };

  // 5. Wire the Graph

  // Wire the dummy start unconditionally to the true logic entry
  builder.setInsertionPointToEnd(dummyStart);
  if (newL->newProd) {
    builder.create<cf::BranchOp>(loc, newL->newProd);
  } else {
    builder.create<func::ReturnOp>(loc);
  }

  for (auto [oldBlock, newBlock] : oldToNew) {
    // Sink Logic: Terminate
    if (oldBlock == rawGraph.sinkBB) {
      builder.setInsertionPointToEnd(newBlock);
      builder.create<func::ReturnOp>(loc);
      continue;
    }

    // Consumer Logic: MUST Branch to Sink
    if (oldBlock == rawGraph.newCons) {
      builder.setInsertionPointToEnd(newBlock);
      // In LocalCFG, Consumer always branches to Sink.
      // We replicate this connection in the new graph.
      Block *newSink = newL->sinkBB;
      if (newSink) {
        builder.create<cf::BranchOp>(loc, newSink);
      } else {
        // Should not happen if Sink is in nodeSet
        builder.create<func::ReturnOp>(loc);
      }
      continue;
    }

    // Decision Node Logic
    Operation *term = oldBlock->getTerminator();
    if (!term)
      continue;

    builder.setInsertionPointToEnd(newBlock);

    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      Block *oldTrue = findNearest(condBr.getTrueDest());
      Block *oldFalse = findNearest(condBr.getFalseDest());

      Block *newTrue = oldToNew.lookup(oldTrue);
      Block *newFalse = oldToNew.lookup(oldFalse);

      // [Safety Wiring] If a path is dead/looping, wire to Sink
      if (!newTrue)
        newTrue = newL->sinkBB;
      if (!newFalse)
        newFalse = newL->sinkBB;

      // [Constraint Application]
      // If this block has a constraint, wire the invalid path to Sink.
      if (muxConstraints.count(oldBlock)) {
        bool requiredVal = muxConstraints.lookup(oldBlock);
        if (requiredVal) {
          // Require True -> Wire False to Sink
          newFalse = newL->sinkBB;
        } else {
          // Require False -> Wire True to Sink
          newTrue = newL->sinkBB;
        }
      }

      Value cond = builder.create<arith::ConstantIntOp>(loc, 1, 1);
      builder.create<cf::CondBranchOp>(loc, cond, newTrue, newFalse);
    } else {
      // Non-CondBranch nodes in the decision set (rare)
      // Wire to nearest valid successor or Sink
      Block *oldTarget = findNearest(term->getSuccessor(0));
      Block *newTarget = oldToNew.lookup(oldTarget);
      if (!newTarget)
        newTarget = newL->sinkBB;
      builder.create<cf::BranchOp>(loc, newTarget);
    }
  }

  // 6. Compute TopoOrder
  DenseSet<Block *> visited;
  SmallVector<Block *, 8> order;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visited.contains(u))
      return;
    visited.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(); it != term->successor_end(); ++it)
        topo(*it);
    order.push_back(u);
  };

  if (!newRegion.empty()) {
    topo(&newRegion.front());
    std::reverse(order.begin(), order.end());
    newL->topoOrder = std::move(order);
  }

  return newL;
}


// ===--------------------------------------------------------------------=== //
// CyclicDemotionHelper implementation
// ===--------------------------------------------------------------------=== //

CyclicDemotionHelper::CyclicDemotionHelper(
    mlir::OpBuilder &builder, MLIRContext *ctx,
    const BlockIndexing &bi, CyclicGraphManager &cyclicMgr,
    DenseMap<Block *, Block *> &origToFullDG,
    Location loc, Block *insertBlock,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow)
    : builder(builder), ctx(ctx), bi(bi),
      cyclicMgr(cyclicMgr), origToFullDG(origToFullDG),
      loc(loc), insertBlock(insertBlock),
      pendingMuxOperands(pendingMuxOperands),
      shadow(shadow) {}

unsigned CyclicDemotionHelper::getVarNativeLevel(const std::string &var) {
    auto opt = bi.getBlockFromCondition(var);
    if (!opt)
      return 0;
    auto it = origToFullDG.find(opt.value());
    if (it == origToFullDG.end())
      return 0;
    return cyclicMgr.getNestingLevel(it->second);
}

LoopScope *CyclicDemotionHelper::findScopeForBlock(Block *origIRBlock,
                                                    unsigned level) {
    auto it = origToFullDG.find(origIRBlock);
    if (it == origToFullDG.end())
      return nullptr;
    Block *dgBlock = it->second;
    std::function<ftd::LoopScope *(ftd::LoopScope *)> search;
    search = [&](ftd::LoopScope *s) -> ftd::LoopScope * {
      if (s->level == level && s->allBlocksInclusive.contains(dgBlock))
        return s;
      for (auto &sub : s->subLoops)
        if (auto *f = search(sub.get()))
          return f;
      return nullptr;
    };
    return search(cyclicMgr.getTopLevelScope());
  }

Value CyclicDemotionHelper::demoteOneLevel(Value currentValue, Block *origBlock,
                                            unsigned fromLevel) {
    ftd::LoopScope *scope = findScopeForBlock(origBlock, fromLevel);
    if (!scope) {
      llvm::errs() << "[FTD Warning] No LoopScope found at level " << fromLevel
                   << " for demotion.\n";
      return currentValue;
    }

    // 1. Extract acyclic layered CFG for this loop scope
    OpBuilder layerBuilder(ctx);
    auto levelCFG = cyclicMgr.extractLayeredCFG(scope, layerBuilder);

    // 2. CDA on levelCFG (main suppression: header -> loop exit)
    ControlDependenceAnalysis levelCDA(*levelCFG->region);
    DenseSet<Block *> levelDeps =
        levelCDA.getAllBlockDeps()[levelCFG->newCons].allControlDeps;

    // 3. Create level-local registry and pre-register all dep variables
    //    at their correct level (demoting higher-level ones recursively)
    SignalRegistry levelRegistry;
    for (Block *depBlock : levelDeps) {
      Block *depOrigBlock = levelCFG->origMap.lookup(depBlock);
      if (!depOrigBlock)
        continue;
      std::string depVar = bi.getBlockCondition(depOrigBlock);
      if (depVar.empty())
        continue;

      unsigned depNative = getVarNativeLevel(depVar);
      Value depValue;
      if (depNative <= fromLevel) {
        // Variable is at this level or outer — use original value
        depValue = getOriginalValue(builder, depVar, bi, pendingMuxOperands, shadow);
      } else {
        // Variable is from a deeper loop — demote to this level
        depValue = getValueAtLevel(depVar, fromLevel);
      }
      if (depValue) {
        if (!depValue.getType().isa<handshake::ChannelType>())
          depValue.setType(ftd::channelifyType(depValue.getType()));
        levelRegistry.registerSignal(depVar, {}, depValue);
      }
    }

    // 4. Distribution on levelCFG
    buildDistributionNetwork(builder, *levelCFG, bi, levelRegistry, pendingMuxOperands, shadow);

    // 5. Main suppression expression: header -> loop exit
    BoolExpression *fCons = enumeratePaths(*levelCFG, bi, levelDeps);
    fCons = fCons->boolMinimize();
    BoolExpression *fSup = fCons->boolNegate()->boolMinimize();

    // 6. DP suppression: header -> value's block
    BoolExpression *fSupDP = BoolExpression::boolZero();
    DenseMap<Block *, unsigned> rankDP;  // pre-computed rank for DP graph
    Block *origHeader = levelCFG->origMap.lookup(levelCFG->newProd);

    if (origHeader && origHeader != origBlock) {
      OpBuilder dpBuilder(ctx);
      auto dpLocGraph =
          buildLocalCFGRegion(dpBuilder, origHeader, origBlock, bi);
      if (dpLocGraph->newCons) {
        ControlDependenceAnalysis dpCDA(*dpLocGraph->region);
        auto dpDepsTmp =
            dpCDA.getAllBlockDeps()[dpLocGraph->newCons].allControlDeps;
        auto dpDG = buildDecisionGraph(*dpLocGraph, dpDepsTmp);
        ControlDependenceAnalysis dpDGCDA(*dpDG->region);
        auto dpDeps =
            dpDGCDA.getAllBlockDeps()[dpDG->newCons].allControlDeps;
        BoolExpression *fConsDP = enumeratePaths(*dpDG, bi, dpDeps);
        fSupDP = fConsDP->boolMinimize()->boolNegate()->boolMinimize();

        // Ensure all DP expression variables are in the level registry
        std::set<std::string> dpVars = fSupDP->getVariables();
        for (const auto &v : dpVars) {
          if (!levelRegistry.lookup(v, {})) {
            unsigned vNative = getVarNativeLevel(v);
            Value vVal;
            if (vNative <= fromLevel)
              vVal = getOriginalValue(builder, v, bi, pendingMuxOperands, shadow);
            else
              vVal = getValueAtLevel(v, fromLevel);
            if (vVal) {
              if (!vVal.getType().isa<handshake::ChannelType>())
                vVal.setType(ftd::channelifyType(vVal.getType()));
              levelRegistry.registerSignal(v, {}, vVal);
            }
          }
        }

        // Pre-compute DP rank before erasing
        rankDP = computeTopoRank(*dpDG);

        dpDG->containerOp->erase();
      }
      dpLocGraph->containerOp->erase();
    }

    // 7. Build suppression circuit
    Value result = currentValue;
    if (!currentValue.getType().isa<handshake::ChannelType>())
      currentValue.setType(ftd::channelifyType(currentValue.getType()));

    if (fSup->type != experimental::boolean::ExpressionType::Zero) {
      DenseMap<Block *, unsigned> rank = computeTopoRank(*levelCFG);
      Value branchCond = expressionToCircuit(builder, fSup, rank, insertBlock,
                                             levelRegistry, bi,
                                             pendingMuxOperands, shadow);

      // Cascaded DP filter
      if (fSupDP->type != experimental::boolean::ExpressionType::Zero) {
        Value dpCond = expressionToCircuit(builder, fSupDP, rankDP, insertBlock,
                                           levelRegistry, bi,
                                           pendingMuxOperands, shadow);
        auto dpBranch = builder.create<handshake::ConditionalBranchOp>(
            loc, ftd::getListTypes(branchCond.getType()),
            dpCond, branchCond);
        dpBranch->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
        setBBAttr(dpBranch, insertBlock, builder);
        branchCond = dpBranch.getFalseResult();
      }

      auto branchOp = builder.create<handshake::ConditionalBranchOp>(
          loc, ftd::getListTypes(currentValue.getType()),
          branchCond, currentValue);
      branchOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      setBBAttr(branchOp, insertBlock, builder);
      result = branchOp.getFalseResult();
    }

    levelCFG->containerOp->erase();
    return result;
  }

Value CyclicDemotionHelper::getValueAtLevel(const std::string &varName,
                                             unsigned targetLevel) {
    auto key = std::make_pair(varName, targetLevel);
    auto it = demotionCache.find(key);
    if (it != demotionCache.end())
      return it->second;

    unsigned native = getVarNativeLevel(varName);
    Value val;
    if (native <= targetLevel) {
      // Variable is at or below target level — use original value
      val = getOriginalValue(builder, varName, bi, pendingMuxOperands, shadow);
      if (val && !val.getType().isa<handshake::ChannelType>())
        val.setType(ftd::channelifyType(val.getType()));
    } else {
      // Recursively get value one level above, then demote
      Value higher = getValueAtLevel(varName, targetLevel + 1);
      auto blockOpt = bi.getBlockFromCondition(varName);
      assert(blockOpt.has_value() && "Variable block not found");
      val = demoteOneLevel(higher, blockOpt.value(), targetLevel + 1);
    }

    demotionCache[key] = val;
    return val;
  }


// ===--------------------------------------------------------------------=== //
// insertDirectSuppression — main suppression entry point
// ===--------------------------------------------------------------------=== //

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
void ftd::insertDirectSuppression(
    mlir::OpBuilder &builder, handshake::FuncOp &funcOp, Operation *consumer,
    Value connection, ftd::ShadowCFG &shadow) {
  Region &shadowRegion = shadow.getRegion();
  ftd::BlockIndexing bi(shadowRegion);

  // Map producer and consumer to shadow blocks via handshake.bb
  unsigned prodBBIdx = 0;
  if (auto *defOp = connection.getDefiningOp())
    if (auto attr = defOp->getAttrOfType<IntegerAttr>("handshake.bb"))
      prodBBIdx = attr.getUInt();
  unsigned consBBIdx = 0;
  if (auto attr = consumer->getAttrOfType<IntegerAttr>("handshake.bb"))
    consBBIdx = attr.getUInt();
  IntegerAttr prodBBAttr = IntegerAttr::get(
      IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned),
      prodBBIdx);
  IntegerAttr targetBBAttr = prodBBAttr;
  Block *producerIRBlock =
      connection.getDefiningOp() ? connection.getDefiningOp()->getBlock()
                                 : consumer->getBlock();

  Block *entryBlock = shadow.getBlock(0);
  Block *producerBlock = shadow.getBlock(prodBBIdx);
  Block *consumerBlock = shadow.getBlock(consBBIdx);
  if (IntegerAttr loopExitBBAttr = getFirstLoopExitBBAttrIfHeaderConsumer(
          builder, shadowRegion, producerBlock, consumerBlock, bi, shadow))
    targetBBAttr = loopExitBBAttr;
  Block *dominatorBlock = producerBlock;

  bool debuglog = false;
  bool debugGraphDump = false;
  // std::string funcName = funcOp.getName().str();
  // std::string dir = "/home/username/dynamatic-scripts/Outputs/";
  // std::string logFile = dir + funcName + "_debuglog.txt";
  // std::error_code EC_log;
  // llvm::raw_fd_ostream log(logFile, EC_log,
  //                          static_cast<llvm::sys::fs::OpenFlags>(0x0004));
  // llvm::raw_ostream &out = EC_log ? llvm::errs() : log;
  llvm::raw_ostream &out = llvm::errs();


  // Helper: translate a local block to a readable orig-block name
  auto blockName = [](const ftd::LocalCFG &cfg, Block *b) -> std::string {
    if (!b)
      return "(null)";
    if (b == cfg.sinkBB)
      return "sink";
    if (b == cfg.secondVisitBB)
      return "secondVisit";
    auto it = cfg.origMap.find(b);
    if (it != cfg.origMap.end() && it->second) {
      std::string name;
      llvm::raw_string_ostream ss(name);
      it->second->printAsOperand(ss);
      return ss.str();
    }
    std::string name;
    llvm::raw_string_ostream ss(name);
    b->printAsOperand(ss);
    return "local(" + ss.str() + ")";
  };

  // Concise topology dump: roles, topo order, and edges in orig-block names
  auto dumpLocalCFG = [&](const ftd::LocalCFG &cfg, llvm::StringRef name) {
    if (!debugGraphDump)
      return;
    out << "===== " << name << " =====\n";
    out << "  prod=" << blockName(cfg, cfg.newProd)
        << "  cons=" << blockName(cfg, cfg.newCons)
        << "  sink=" << blockName(cfg, cfg.sinkBB);
    if (cfg.secondVisitBB)
      out << "  secondVisit=" << blockName(cfg, cfg.secondVisitBB);
    out << "\n  topo: ";
    for (unsigned i = 0; i < cfg.topoOrder.size(); ++i) {
      if (i > 0)
        out << " -> ";
      out << blockName(cfg, cfg.topoOrder[i]);
    }
    out << "\n  edges:\n";
    if (cfg.region) {
      for (Block *b : cfg.topoOrder) {
        Operation *term = b->getTerminator();
        if (!term || term->getNumSuccessors() == 0)
          continue;
        std::string src = blockName(cfg, b);
        if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
          out << "    " << src
              << " -T-> " << blockName(cfg, condBr.getTrueDest())
              << ", -F-> " << blockName(cfg, condBr.getFalseDest()) << "\n";
        } else {
          for (unsigned i = 0; i < term->getNumSuccessors(); ++i) {
            out << "    " << src
                << " -> " << blockName(cfg, term->getSuccessor(i)) << "\n";
          }
        }
      }
    }
    out << "\n";
  };

  // Account for the condition of a Mux only if it corresponds to a GAMMA GSA
  // gate and the producer is one of its data inputs
  bool deliverToGamma = llvm::isa<handshake::MuxOp>(consumer) &&
                        consumer->hasAttr(FTD_EXPLICIT_GAMMA) &&
                        (producerBlock != consumerBlock ||
                         connection.getDefiningOp()->hasAttr(FTD_EXPLICIT_MU));

  // Activate graph dumps when deliverToGamma is true (user-adjustable)
  // debugGraphDump = deliverToGamma;

  if (debuglog) {
    out << "[FTD] Producer block: ";
    if (producerBlock)
      producerBlock->printAsOperand(out);
    else
      out << "(null)";
    out << ", Consumer block: ";
    if (consumerBlock)
      consumerBlock->printAsOperand(out);
    else
      out << "(null)";
    out << "\n";
  }

  // If producer is unreachable, the suppression is not needed.
  if (!isReachable(entryBlock, producerBlock)) {
    return;
  }

  auto getBB = [](Operation *op) -> unsigned {
    auto attr = op->getAttrOfType<IntegerAttr>("handshake.bb");
    return attr ? attr.getUInt() : 0;
  };

  // If deliverToGamma is true, we need to trace down the mux chain to find the
  // condition block that effectively controls the delivery.
  if (deliverToGamma) {
    Operation *currentMuxOp = consumer;
    Value currentConnection = connection;
    Block *lastValidDominator = producerBlock;
    // Acyclic reachability check: only follows forward edges (skips back-edges)
    // using the block ordering from BlockIndexing.
    auto isReachableAcyclic = [&](Block *start, Block *end) -> bool {
      if (start == end)
        return true;
      DenseSet<Block *> visited;
      SmallVector<Block *, 8> stack;
      stack.push_back(start);
      visited.insert(start);
      while (!stack.empty()) {
        Block *curr = stack.pop_back_val();
        for (Block *succ : curr->getSuccessors()) {
          if (bi.isLess(succ, curr) || succ == curr)
            continue;
          if (succ == end)
            return true;
          if (visited.insert(succ).second)
            stack.push_back(succ);
        }
      }
      return false;
    };
    while (true) {
      // If the connection enters as operand(0) (condition input),
      // skip the reachability check and just trace to the next Mux.
      if (currentMuxOp->getOperand(0) != currentConnection) {
        // Connection is a data input — check this Mux's condition block
        Value condition = currentMuxOp->getOperand(0);
        Block *condBlock = returnMuxConditionBlock(condition, shadow);

        bool bothReach = condBlock &&
                         condBlock->getNumSuccessors() >= 2 &&
                         isReachableAcyclic(condBlock->getSuccessor(0), producerBlock) &&
                         isReachableAcyclic(condBlock->getSuccessor(1), producerBlock);

        // Among all qualifying condBlocks, pick the one earliest in
        // topological order (smallest by bi). This gives the outermost
        // dominator that still encloses the producer on both branches.
        if (bothReach && bi.isLess(condBlock, lastValidDominator)) {
          lastValidDominator = condBlock;
        }
      }

      // Trace down to the next Gamma Mux in the same block
      Operation *nextMuxOp = nullptr;
      Value currentResult = currentMuxOp->getResult(0);
      for (auto *user : currentResult.getUsers()) {
        if (llvm::isa<handshake::MuxOp>(user) &&
            user->hasAttr(FTD_EXPLICIT_GAMMA) &&
            getBB(user) == getBB(currentMuxOp)) {
          unsigned cnt = 0;
          if (user->getOperand(1) == currentResult) cnt++;
          if (user->getOperand(2) == currentResult) cnt++;
          if (cnt != 2) {
            nextMuxOp = user;
            break;
          }
        }
      }

      if (!nextMuxOp)
        break;
      currentConnection = currentMuxOp->getResult(0);
      currentMuxOp = nextMuxOp;
    }
    if (lastValidDominator && lastValidDominator != producerBlock) {
      llvm::errs() << "[FTD] Last valid dominator block in Mux chain: ";
      lastValidDominator->printAsOperand(llvm::errs());
      llvm::errs() << "\n";
    }
    dominatorBlock = lastValidDominator;
  }

  if (debuglog && deliverToGamma) {
    out << "[FTD] Another Dominator block: ";
    if (dominatorBlock)
      dominatorBlock->printAsOperand(out);
    out << "\n";
  }

  // Create a temporary builder to isolate the LocalCFG creation from the
  // main OpBuilder. This prevents the OpBuilder from tracking the
  // temporary operations which are later erased manually.
  OpBuilder tmpBuilder(funcOp.getContext());
  auto locGraph =
      buildLocalCFGRegion(tmpBuilder, dominatorBlock, consumerBlock, bi);

  dumpLocalCFG(*locGraph, "locGraph (Dominator -> Consumer)");

  ControlDependenceAnalysis locCDA(*locGraph->region);
  // Full Set: contains consumer dependences, Mux Conditions and their
  // dependences.
  DenseSet<Block *> locConsControlDepsFull =
      locCDA.getAllBlockDeps()[locGraph->newCons].allControlDeps;
  // The set containing Mux Conditions
  DenseSet<Block *> muxConditionSet;

  // Map to store specific constraints for Mux Conditions (LocalBlock ->
  // RequiredValue)
  DenseMap<Block *, bool> muxConstraints;

  // Logic specific to Gamma delivery to identify Mux dependencies
  if (deliverToGamma) {
    Operation *currentMuxOp = consumer;
    Value currentConnection = connection;
    bool isChainActive = true;

    while (isChainActive) {
      bool isDataInput = false;
      bool requiredVal = true;

      // Check how the connection enters the current Mux
      // If operand(0) == currentConnection, it is the condition input.
      // In that case, isDataInput remains false, and we simply traverse
      // to the output to find the next mux in the chain.
      if (!(currentMuxOp->getOperand(0) == currentConnection)) {
        if (currentMuxOp->getOperand(1) == currentConnection) {
          // Input 1 is the FALSE input
          isDataInput = true;
          requiredVal = false;
        } else if (currentMuxOp->getOperand(2) == currentConnection) {
          // Input 2 is the TRUE input
          isDataInput = true;
          requiredVal = true;
        }
      }

      if (isDataInput) {
        if (debuglog) {
          out << "[MUX] Mux Condition Block: ";
          Block *muxConditionBlock =
              returnMuxConditionBlock(currentMuxOp->getOperand(0), shadow);
          if (muxConditionBlock)
            muxConditionBlock->printAsOperand(out);
          else
            out << "(null)";
          out << "\n";

          if (!requiredVal) {
            out << "      Negated : (false input Selected)\n";
          } else {
            out << "      Original: (true input Selected)\n";
          }
        }
        // 1. Get the condition value driving this Mux
        Value muxCondition = currentMuxOp->getOperand(0);

        // Trace back through suppression branches
        // to find the original source of the condition signal.
        while (Operation *defOp = muxCondition.getDefiningOp()) {
          if (llvm::isa<handshake::ConditionalBranchOp>(defOp) &&
              defOp->hasAttr(FTD_OP_TO_SKIP)) {
            // Move up to the data input of the branch
            muxCondition = defOp->getOperand(1);
          } else {
            break;
          }
        }

        // 2. Identify the Original Block defining this condition variable
        // (Using the provided helper function)
        Block *muxConditionBlock = returnMuxConditionBlock(muxCondition, shadow);
        muxConditionSet.insert(muxConditionBlock);

        // 3. Find the corresponding Block in the Local CFG
        // Since locGraph->origMap maps Local->Original, we iterate to reverse
        // lookup.
        Block *condBlockLocal = nullptr;
        for (auto it : locGraph->origMap) {
          if (it.second == muxConditionBlock) {
            condBlockLocal = it.first;
            break;
          }
        }

        // 4. Add to dependencies and record requirement
        if (condBlockLocal) {
          if (bi.isLess(muxConditionBlock, dominatorBlock))
            continue;
          // Add this block and its all-dependency blocks to the dependency set
          // so path enumeration observes it.
          locConsControlDepsFull.insert(condBlockLocal);
          for (Block *dep :
               locCDA.getAllBlockDeps()[condBlockLocal].allControlDeps) {
            locConsControlDepsFull.insert(dep);
          }

          // Record the specific value required (True/False) to pass this Mux
          muxConstraints[condBlockLocal] = requiredVal;
        } else {
          // llvm::errs() << "[FTD Warning] Mux condition block not found in "
          //                 "LocalCFG for condition: ";
          // muxConditionBlock->printAsOperand(llvm::errs());
          // llvm::errs() << " \n";
        }
      }

      // 5. Traverse Downstream (Search for cascaded Gamma Muxes)
      Operation *nextMuxOp = nullptr;
      Value currentResult = currentMuxOp->getResult(0);

      for (auto *user : currentResult.getUsers()) {
        // Check if user is a Mux in the same block marked as Gamma
        if (llvm::isa<handshake::MuxOp>(user) &&
            user->hasAttr(FTD_EXPLICIT_GAMMA) &&
            getBB(user) == getBB(currentMuxOp)) {
          // Count how many data inputs use currentResult
          unsigned connectionCount = 0;
          if (user->getOperand(1) == currentResult)
            connectionCount++;
          if (user->getOperand(2) == currentResult)
            connectionCount++;

          // We only proceed if at most ONE data input comes from the previous
          // mux. Otherwise, the user is a temporary MUX which we don't care
          // about.
          if (connectionCount != 2) {
            nextMuxOp = user;
            // Update connection for next iteration
            currentConnection = currentResult;
            break;
          }
        }
      }

      if (nextMuxOp) {
        if (debuglog) {
          out << "    -> Found Cascaded Gamma Mux:\n";
        }
        currentMuxOp = nextMuxOp;
      } else {
        isChainActive = false;
        if (debuglog)
          out << "    -> End of Gamma Mux Chain.\n";
      }
    }
  }

  if (locConsControlDepsFull.empty()) {
    if (locGraph->newCons == locGraph->sinkBB) {
      builder.setInsertionPointToStart(consumer->getBlock());
      auto src = builder.create<handshake::SourceOp>(consumer->getLoc());
      src->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      src->setAttr("handshake.bb", targetBBAttr);
      auto cstAttr = builder.getIntegerAttr(builder.getIntegerType(1), 1);
      auto constOp = builder.create<handshake::ConstantOp>(
          consumer->getLoc(), cstAttr, src.getResult());
      constOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      constOp->setAttr("handshake.bb", targetBBAttr);

      Value supData = connection;
      auto branchOp = builder.create<handshake::ConditionalBranchOp>(
          consumer->getLoc(), ftd::getListTypes(supData.getType()),
          constOp.getResult(), supData);
      branchOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      branchOp->setAttr("handshake.bb", targetBBAttr);

      for (auto &use : llvm::make_early_inc_range(connection.getUses())) {
        if (use.getOwner() != consumer)
          continue;
        use.set(branchOp.getFalseResult());
      }
      locGraph->containerOp->erase();
      return;
    }
    locGraph->containerOp->erase();
    return;
  }

  // Common Logic for Building Suppression.
  // If deliverToGamma is true, we use the empty constraints to build the
  // distribution network (so it covers all paths), but we use the
  // muxConstraints to calculate the specific suppression condition for this
  // path. If deliverToGamma is false, muxConstraints will be empty, so both
  // graphs are identical.
  SignalRegistry registry;
  builder.setInsertionPointToStart(consumer->getBlock());
  auto fullDecisionGraph =
      buildDecisionGraph(*locGraph, locConsControlDepsFull);

  dumpLocalCFG(*fullDecisionGraph, "fullDecisionGraph (locGraph + FullDeps)");

  // Cycle Analysis Integration
  ftd::CyclicGraphManager cyclicMgr(*fullDecisionGraph);

  // Reverse map: original IR block -> fullDecisionGraph block
  DenseMap<Block *, Block *> origToFullDG;
  for (auto [dgBlock, origBlock] : fullDecisionGraph->origMap)
    if (origBlock)
      origToFullDG[origBlock] = dgBlock;

  // Create the cyclic demotion helper
  CyclicDemotionHelper demotionHelper(builder, funcOp.getContext(), bi, cyclicMgr,
                                       origToFullDG, consumer->getLoc(),
                                       consumer->getBlock(), nullptr, &shadow);

  // Level 0: Extract acyclic layered CFG from the full decision graph.
  // This cuts all back-edges, giving a DAG where CDA produces correct
  // forward dependencies without the findNearest-through-backedge problem.
  OpBuilder level0Builder(funcOp.getContext());
  auto level0CFG = cyclicMgr.extractLayeredCFG(
      cyclicMgr.getTopLevelScope(), level0Builder);

  dumpLocalCFG(*level0CFG, "level0CFG (Acyclic Layered from fullDG)");

  // CDA on the acyclic level 0 CFG — allControlDeps == forwardControlDeps
  // because the graph is a DAG.
  ControlDependenceAnalysis level0CDA(*level0CFG->region);
  DenseSet<Block *> level0Deps =
      level0CDA.getAllBlockDeps()[level0CFG->newCons].allControlDeps;

  // Translate mux condition blocks (original IR) into level0CFG and insert
  // them into the dependency set — mirrors locConsAcyclicDeps.insert(condBlockLocal).
  for (Block *muxCondOrigIR : muxConditionSet) {
    for (auto &[l0Block, l0Orig] : level0CFG->origMap) {
      if (l0Orig == muxCondOrigIR) {
        level0Deps.insert(l0Block);
        for (Block *dep :
             level0CDA.getAllBlockDeps()[l0Block].allControlDeps)
          level0Deps.insert(dep);
        break;
      }
    }
  }

  // Translate mux constraints from locGraph blocks to level0CFG blocks.
  DenseMap<Block *, bool> level0MuxConstraints;
  for (auto &[locBlock, reqVal] : muxConstraints) {
    Block *origIR = locGraph->origMap.lookup(locBlock);
    if (!origIR)
      continue;
    for (auto &[l0Block, l0Orig] : level0CFG->origMap) {
      if (l0Orig == origIR) {
        level0MuxConstraints[l0Block] = reqVal;
        break;
      }
    }
  }

  // Level 0: distribution graph (unconstrained, covers all paths)
  auto level0FullDG = buildDecisionGraph(*level0CFG, level0Deps);

  dumpLocalCFG(*level0FullDG, "level0FullDG (level0CFG + Deps)");

  // Pre-register demoted values for high-level variables in level 0
  demotionHelper.preRegisterDemotedValues(level0FullDG, registry);

  // Build distribution on level 0
  buildDistributionNetwork(builder, *level0FullDG, bi, registry, nullptr, &shadow);

  // Level 0: constrained graph (for suppression expression)
  auto level0ConstrainedDG =
      buildDecisionGraph(*level0CFG, level0Deps, level0MuxConstraints);

  dumpLocalCFG(*level0ConstrainedDG,
               "level0ConstrainedDG (level0CFG + muxConstraints)");

  if (debugGraphDump && !muxConstraints.empty()) {
    out << "  muxConstraints (" << muxConstraints.size() << "):\n";
    for (auto &entry : muxConstraints) {
      out << "    ";
      if (entry.first)
        entry.first->printAsOperand(out);
      out << " requires " << (entry.second ? "TRUE" : "FALSE") << "\n";
    }
    out << "\n";
  }

  ControlDependenceAnalysis decCDA(*level0ConstrainedDG->region);
  DenseSet<Block *> constrainedDeps =
      decCDA.getAllBlockDeps()[level0ConstrainedDG->newCons].allControlDeps;

  BoolExpression *fCons =
      enumeratePaths(*level0ConstrainedDG, bi, constrainedDeps);

  fCons = fCons->boolMinimize();
  if (debuglog) {
    out << "fCons  = " << fCons->toString() << "\n";
  }
  // f_supp = f_prod and not f_cons
  BoolExpression *fSup = fCons->boolNegate();
  fSup = fSup->boolMinimize();
  if (debuglog) {
    out << "fSupmin  = " << fSup->toString() << "\n\n\n";
  }

  // Suppression Logic between Dominator and Producer (locGraphDP)
  BoolExpression *fSupDP = BoolExpression::boolZero();
  DenseMap<Block *, unsigned> rankDP;  // pre-computed rank for DP graph

  if (dominatorBlock != producerBlock) {
    OpBuilder tmpBuilder2(funcOp.getContext());
    auto locGraphDP = buildLocalCFGRegion(tmpBuilder2, dominatorBlock,
                                          producerBlock, bi);

    dumpLocalCFG(*locGraphDP, "locGraphDP (Dominator -> Producer)");

    if (locGraphDP->newCons) {
      // Build fullDecisionGraph for DP path
      ControlDependenceAnalysis dpLocCDA(*locGraphDP->region);
      DenseSet<Block *> dpDepsTmp =
          dpLocCDA.getAllBlockDeps()[locGraphDP->newCons].allControlDeps;
      auto dpFullDG = buildDecisionGraph(*locGraphDP, dpDepsTmp);

      dumpLocalCFG(*dpFullDG, "dpFullDG (locGraphDP + dpDeps)");

      ControlDependenceAnalysis finalDpCDA(*dpFullDG->region);
      DenseSet<Block *> dpDeps =
          finalDpCDA.getAllBlockDeps()[dpFullDG->newCons].allControlDeps;

      BoolExpression *fConsDP = enumeratePaths(*dpFullDG, bi, dpDeps);
      fSupDP = fConsDP->boolMinimize()->boolNegate()->boolMinimize();

      if (debuglog) {
        out << "fSupDP   = " << fSupDP->toString() << "\n\n\n";
      }

      // Pre-compute DP rank before erasing
      rankDP = computeTopoRank(*dpFullDG);

      dpFullDG->containerOp->erase();
    }

    // Pre-compute DP rank from locGraphDP as fallback if dpFullDG was not built
    if (rankDP.empty())
      rankDP = computeTopoRank(*locGraphDP);

    locGraphDP->containerOp->erase();
  }

  Value supData = connection;

  level0CFG->containerOp->erase();
  level0FullDG->containerOp->erase();
  level0ConstrainedDG->containerOp->erase();
  fullDecisionGraph->containerOp->erase();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    IntegerAttr connectionBBAttr = targetBBAttr;
    DenseMap<Block *, unsigned> rank = computeTopoRank(*locGraph);
    Value branchCond =
        expressionToCircuit(builder, fSup, rank, producerIRBlock,
                            registry, bi, nullptr, &shadow, connectionBBAttr);

    // Cascaded Upstream Filter
    if (fSupDP->type != experimental::boolean::ExpressionType::Zero) {
      Value dpBranchCond =
          expressionToCircuit(builder, fSupDP, rankDP, producerIRBlock,
                              registry, bi, nullptr, &shadow, connectionBBAttr);

      // Upstream logic filters the SUPPRESSION SIGNAL.
      // Create an Intermediate Branch on the 'branchCond' wire.
      // Data Input: branchCond (The Downstream suppression signal)
      // Condition: dpBranchCond (from Upstream logic)
      auto dpBranchOp = builder.create<handshake::ConditionalBranchOp>(
          consumer->getLoc(), ftd::getListTypes(branchCond.getType()),
          dpBranchCond, branchCond);
      dpBranchOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
      dpBranchOp->setAttr("handshake.bb", targetBBAttr);
      branchCond = dpBranchOp.getFalseResult();
    }

    auto branchOp = builder.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), ftd::getListTypes(supData.getType()), branchCond,
        supData);
    branchOp->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
    branchOp->setAttr("handshake.bb", targetBBAttr);
    supData = branchOp.getFalseResult();

    // Take into account the possibility of a mux to get the condition input
    // also as data input. In this case, the data input can be optimized to
    // a constant value, since it is always selected only when its own value
    // is true or false.
    for (auto &use : llvm::make_early_inc_range(connection.getUses())) {
      if (use.getOwner() != consumer)
        continue;
      if (llvm::isa<handshake::MuxOp>(consumer) &&
          consumer->getOperand(0) == connection &&
          use.getOperandNumber() != 0) {
        auto src = builder.create<handshake::SourceOp>(consumer->getLoc());
        setBBAttrWithFallback(src, targetBBAttr, producerIRBlock, builder);
        auto innerType =
            connection.getType().cast<handshake::ChannelType>().getDataType();
        auto attr =
            builder.getIntegerAttr(innerType, (use.getOperandNumber() == 2));
        auto cst = builder.create<handshake::ConstantOp>(
            consumer->getLoc(), connection.getType(), attr, src.getResult());
        cst->setAttr(FTD_OP_TO_SKIP, builder.getUnitAttr());
        setBBAttrWithFallback(cst, targetBBAttr, producerIRBlock, builder);
        use.set(cst.getResult());
        continue;
      }
      use.set(supData);
    }
  }
  locGraph->containerOp->erase();
}
