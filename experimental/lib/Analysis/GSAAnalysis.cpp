//===- GSAAnalysis.cpp - GSA analyis utilities ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assignment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vector"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "gsa"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::ftd;
using namespace dynamatic::experimental::boolean;

// ---------------------------------------------------------------------------
// Crash-hunting debug logging.
//
// Writes each line straight to stderr (unbuffered) and flushes, so the LAST
// line printed before an abort identifies exactly where execution died. This
// is deliberately independent of any file stream because heap corruption can
// make buffered/heap-backed streams unusable at abort time. Prefix [GSA-DBG]
// distinguishes it from the [FTD] logs in FtdCfToHandshake.cpp.
//
// These are pure instrumentation lines; no analysis logic is changed.
// ---------------------------------------------------------------------------
static void gsaLog(const llvm::Twine &msg) {
  llvm::errs() << "[GSA-DBG] " << msg << "\n";
  llvm::errs().flush();
}

// ---------------------------------------------------------------------------
// Strict-weak-ordering block comparator.
//
// `DominanceInfo::dominates(a, b)` is only a PARTIAL order: blocks on distinct
// CFG branches are mutually incomparable, so both dominates(a,b) and
// dominates(b,a) are false. Passing such a predicate to llvm::sort/std::sort is
// undefined behavior and yields a non-deterministic ordering. Sorting by the
// dominator tree's DFS in-number is a TOTAL order that still guarantees "a
// dominator precedes the blocks it dominates", which is what the callers rely
// on. nullptr blocks (empty GateInputs) are ordered first, deterministically.
//
// The DomTree's DFS numbers must be refreshed before use (they are lazy).
static bool blockDfsLess(Block *a, Block *b,
                         llvm::DominatorTreeBase<Block, /*IsPostDom=*/false>
                             &domTree) {
  if (a == b)
    return false;
  if (!a)
    return true;  // nullptr sorts first
  if (!b)
    return false;
  auto *na = domTree.getNode(a);
  auto *nb = domTree.getNode(b);
  // Unreachable blocks have no DomTree node; order them last but consistently
  // (by pointer) so the comparator stays a strict weak ordering.
  if (!na || !nb)
    return na && !nb ? true : (!na && nb ? false : std::less<Block *>()(a, b));
  return na->getDFSNumIn() < nb->getDFSNumIn();
}

experimental::gsa::GSAAnalysis::GSAAnalysis(handshake::MergeOp &merge,
                                            Region &region) {
  inputOp = &region;
  convertSSAToGSAMerges(merge, region);
}

void experimental::gsa::GSAAnalysis::convertSSAToGSAMerges(
    handshake::MergeOp &mergeOp, Region &region) {

  // Associate an index to each basic block in "funcOp" so that if Bi
  // dominates Bj than i < j
  BlockIndexing bi(region);

  // Initialize the index of the class
  uniqueGateIndex = 0;

  Block *block = mergeOp.getResult().getParentBlock();

  // Create an empty list for the phi functions corresponding to the block
  gatesPerBlock.insert({block, llvm::SmallVector<Gate *>()});

  // Create a set for the operands of the corresponding phi function
  SmallVector<GateInput *> operands;

  auto isAlreadyPresent = [&](Value c) -> bool {
    return std::any_of(operands.begin(), operands.end(), [c](GateInput *in) {
      return in->isTypeValue() && in->getValue() == c;
    });
  };

  // Add to the list of operands of the new gate all the values which were not
  // already used.
  // Since in STQ one block can't be producer of more than one phi operand,
  // sender logic is not needed.

  // Handle self-dependent merges: if the merge uses its own result as an input,
  // record that operand so it can be reconnected to the generated phi later.
  SmallVector<GateInput *> selfInputs;

  for (Value v : mergeOp.getOperands()) {
    if (!isAlreadyPresent(v)) {
      GateInput *gateInput = new GateInput(v);
      if (v == mergeOp.getResult())
        selfInputs.push_back(gateInput);
      gateInputList.push_back(gateInput);
      operands.push_back(gateInput);
    }
  }

  // If the list of operands is not empty (i.e. the phi has at least
  // one input), add it to the phis associated to that block
  Gate *newPhi = nullptr;
  if (!operands.empty()) {
    newPhi = new Gate(mergeOp.getResult(), operands, GateType::PhiGate,
                      ++uniqueGateIndex);
    gatesPerBlock[block].push_back(newPhi);
  }

  // Reconnect self-dependent operands to the newly created Phi
  for (GateInput *gi : selfInputs)
    gi->input = newPhi;

  convertPhiToMu(region, bi);
  convertPhiToGamma(region, bi);

  // After the conversion is done, `gatesPerBlock` will contain some phis, some
  // MUs and some GAMMAs. Since we are only interested in the last two, we can
  // remove all the phis.
  removePhiGates();
  printAllGates();
}

experimental::gsa::GSAAnalysis::GSAAnalysis(Operation *operation) {

  gsaLog("GSAAnalysis(Operation*): enter");

  // Only one function should be present in the module, excluding external
  // functions
  unsigned functionsCovered = 0;

  // TODO: Extend to support multiple functions

  // The analysis can be instantiated either over a module containing one
  // function only or over a function
  if (ModuleOp modOp = dyn_cast<ModuleOp>(operation); modOp) {
    gsaLog("GSAAnalysis(Operation*): operation is a ModuleOp");
    for (func::FuncOp funcOp : modOp.getOps<func::FuncOp>()) {

      // Skip if external
      if (funcOp.isExternal()) {
        gsaLog("GSAAnalysis(Operation*): skip external func " +
               funcOp.getSymName());
        continue;
      }

      // Analyze the function
      if (!functionsCovered) {
        gsaLog("GSAAnalysis(Operation*): analyzing func " +
               funcOp.getSymName());
        inputOp = &funcOp.getRegion();
        convertSSAToGSA(*inputOp);
        functionsCovered++;
        gsaLog("GSAAnalysis(Operation*): convertSSAToGSA returned for func " +
               funcOp.getSymName());
      } else {
        llvm::errs() << "[GSA] Too many functions to handle in the module";
      }
    }
  } else if (func::FuncOp fOp = dyn_cast<func::FuncOp>(operation); fOp) {
    gsaLog("GSAAnalysis(Operation*): operation is a FuncOp " + fOp.getSymName());
    convertSSAToGSA(fOp.getRegion());
    functionsCovered = 1;
    gsaLog("GSAAnalysis(Operation*): convertSSAToGSA returned (func path)");
  }

  // report an error indicating that the analysis is instantiated over
  // an inappropriate operation
  if (functionsCovered != 1)
    llvm::errs() << "[GSA] GSAAnalysis failed due to a wrong input type\n";

  gsaLog("GSAAnalysis(Operation*): exit");
}

experimental::gsa::Gate *experimental::gsa::GSAAnalysis::expandGammaTree(
    ListExpressionsPerGate &expressions, std::queue<unsigned> conditions,
    Gate *originalPhi, const BlockIndexing &bi) {

  gsaLog("  expandGammaTree: enter (expressions=" +
         llvm::Twine(expressions.size()) + ", conditions=" +
         llvm::Twine(conditions.size()) + ")");

  // At each iteration, we want to use a cofactor that is present in all the
  // expressions in `expressions`. Since the cofactors are ordered according to
  // the basic block number, we can say that if a cofactor is present in one
  // expression, then it must be present in all the others, since they all have
  // the blocks associated to that cofactor as common dominator.
  unsigned indexToUse;
  std::string conditionToUse;
  while (true) {
    if (conditions.empty()) {
      gsaLog("  expandGammaTree: WARNING conditions queue empty before finding "
             "a usable cofactor (about to call conditions.front() on empty)");
    }
    indexToUse = conditions.front();
    conditionToUse = "c" + std::to_string(indexToUse);
    conditions.pop();
    unsigned cofactorUsage =
        std::count_if(expressions.begin(), expressions.end(),
                      [&](std::pair<BoolExpression *, GateInput *> g) {
                        return g.first->containsMintern(conditionToUse);
                      });

    gsaLog("  expandGammaTree: trying cofactor " + conditionToUse +
           " usage=" + llvm::Twine(cofactorUsage) + "/" +
           llvm::Twine(expressions.size()));

    // None is using that cofactor, go to the next one
    if (cofactorUsage == 0)
      continue;

    // All are using the cofactor, expand the expressions over it
    if (cofactorUsage == expressions.size())
      break;

    llvm_unreachable("A cofactor is used only by some expressions");
  }

  gsaLog("  expandGammaTree: chosen cofactor " + conditionToUse);

  ListExpressionsPerGate conditionsFalseExpressions;
  ListExpressionsPerGate conditionsTrueExpressions;

  // For each expression
  for (const auto &expression : expressions) {

    // Restrict an expression according to a condition ans an input boolean
    // value and return it
    auto getRestrictedExpression = [&](bool value) -> BoolExpression * {
      BoolExpression *expr = expression.first->deepCopy();
      restrict(expr, conditionToUse, value);
      return expr->boolMinimize();
    };

    // Substitute a condition in the expression both with value true and false
    BoolExpression *exprTrue = getRestrictedExpression(true);
    BoolExpression *exprFalse = getRestrictedExpression(false);

    // One of the two expressions above will be zero: add the input to the
    // corresponding list
    if (exprTrue->type != ExpressionType::Zero)
      conditionsTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->type != ExpressionType::Zero)
      conditionsFalseExpressions.emplace_back(exprFalse, expression.second);
  }

  gsaLog("  expandGammaTree: split done (true=" +
         llvm::Twine(conditionsTrueExpressions.size()) + ", false=" +
         llvm::Twine(conditionsFalseExpressions.size()) + ")");

  SmallVector<GateInput *> operandsGamma(2);

  // If the number of non-null expressions obtained with cofactor = X is greater
  // than 1, then the expansions must be done again over those inputs with a new
  // cofactor. The resulting gamma is going to be used as X input of this
  // gamma.
  //
  // If only one element is present, that same element is used as input of the
  // gamma.
  //
  // If no elements are present, then that input will never be used, and it is
  // considered as empty.
  auto setGammaOperand = [&](int input, ListExpressionsPerGate &list) -> void {
    gsaLog("  expandGammaTree: setGammaOperand input=" + llvm::Twine(input) +
           " listSize=" + llvm::Twine(list.size()));
    if (list.size() > 1) {
      Gate *gamma = expandGammaTree(list, conditions, originalPhi, bi);
      operandsGamma[input] = new GateInput(gamma);
      gateInputList.push_back(operandsGamma[input]);
    } else if (list.size() == 1) {
      operandsGamma[input] = list[0].second;
    } else {
      operandsGamma[input] = new GateInput();
      gateInputList.push_back(operandsGamma[input]);
    }
  };

  setGammaOperand(1, conditionsTrueExpressions);
  setGammaOperand(0, conditionsFalseExpressions);

  // Create a new gamma and add it to the list of phis for the original basic
  // block.

  // Get the index of the condition (it is associated to a basic block in
  // "indexPerBlock" mapping)
  gsaLog("  expandGammaTree: getBlockFromIndex(" + llvm::Twine(indexToUse) +
         ")");
  auto blockFromIndex = bi.getBlockFromIndex(indexToUse);
  if (!blockFromIndex.has_value())
    gsaLog("  expandGammaTree: WARNING getBlockFromIndex returned nullopt "
           "(about to call .value())");
  Gate *newGate = new Gate(
      originalPhi->result, operandsGamma, GateType::GammaGate,
      ++uniqueGateIndex, blockFromIndex.value(),
      BoolExpression::boolVar(conditionToUse),
      {conditionToUse}); // since condition is one block boolvar is enough
  gatesPerBlock[originalPhi->getBlock()].push_back(newGate);

  gsaLog("  expandGammaTree: created gamma idx=" + llvm::Twine(uniqueGateIndex) +
         ", return");

  return newGate;
}

void experimental::gsa::GSAAnalysis::convertSSAToGSA(Region &region) {

  gsaLog("convertSSAToGSA: enter (blocks=" +
         llvm::Twine(region.getBlocks().size()) + ")");

  if (region.getBlocks().size() == 1) {
    gsaLog("convertSSAToGSA: single-block region, returning early");
    return;
  }

  // Associate an index to each basic block in "funcOp" so that if Bi
  // dominates Bj than i < j
  BlockIndexing bi(region);
  gsaLog("convertSSAToGSA: BlockIndexing built");

  // This function works in two steps. First, all the block arguments in the
  // IR are converted into PHIs, taking care or properly extracting the
  // information about the producers of the operands. Then, the phis are
  // converted either to GAMMAs or MUs.

  // The input of a phi might be another gate. This is the case when the
  // input of a phi is a block argument from a block with index different
  // from 0. In this situation, we mark the phi input as "missing", and we
  // store the necessary information (block argument) to later reconstruct
  // the relationship.
  struct MissingPhi {

    // Which input is missing
    GateInput *pi;

    // Related block argument
    BlockArgument blockArg;

    MissingPhi(GateInput *pi, BlockArgument blockArg)
        : pi(pi), blockArg(blockArg) {}
  };

  // Initialize the index of the class
  uniqueGateIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phisToConnect;

  unsigned blockCounter = 0;

  // For each block in the function
  for (Block &block : region.getBlocks()) {

    gsaLog("convertSSAToGSA: [phase1] block #" + llvm::Twine(blockCounter++) +
           " numArgs=" + llvm::Twine(block.getNumArguments()) + " numPreds=" +
           llvm::Twine(std::distance(block.getPredecessors().begin(),
                                     block.getPredecessors().end())));

    // Create an empty list for the phi functions corresponding to the block
    gatesPerBlock.insert({&block, llvm::SmallVector<Gate *>()});

    // For each block argument
    for (BlockArgument &arg : block.getArguments()) {
      unsigned argNumber = arg.getArgNumber();
      gsaLog("convertSSAToGSA: [phase1]   arg #" + llvm::Twine(argNumber));
      // Create a set for the operands of the corresponding phi function
      SmallVector<GateInput *> operands;
      // Track block-argument operands to avoid recording duplicates
      SmallVector<MissingPhi> operandsMissPhi;
      DenseSet<Block *> coveredPredecessors;
      // For each predecessor of the block, which is in charge of
      // providing the inputs of the phi functions
      for (Block *pred : block.getPredecessors()) {

        // Make sure that a predecessor is covered only once
        if (coveredPredecessors.contains(pred))
          continue;
        coveredPredecessors.insert(pred);

        // Get the branch terminator
        auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
        assert(branchOp && "Expected terminator operation in a predecessor "
                           "block feeding a block argument!");

        // Check if a block-argument operand is already present among missing
        // phis. Two missing-phi operands are considered equal if they
        // originated from and target the same argument of the same block.
        // The value is not compared, which might be problematic.
        auto isBlockArgAlreadyPresent = [&](BlockArgument blockArg) -> bool {
          for (MissingPhi &mPhi : operandsMissPhi) {
            if (mPhi.blockArg.getParentBlock() == blockArg.getParentBlock() &&
                mPhi.blockArg.getArgNumber() == blockArg.getArgNumber()) {
              mPhi.pi->senders.insert(pred);
              return true;
            }
          }
          return false;
        };

        // Check if value is already among the operands of the phi.
        // If found, record preds as a sender of that operand.
        auto isValueAlreadyPresent = [&](Value v) -> bool {
          for (GateInput *in : operands) {
            if (in->isTypeValue() && in->getValue() == v) {
              in->senders.insert(pred);
              return true;
            }
          }
          return false;
        };

        // For each alternative in the branch terminator
        for (auto [successorId, successorBlock] :
             llvm::enumerate(branchOp->getSuccessors())) {

          // Skip the successor if it is not the block under analysis
          if (successorBlock != &block)
            continue;

          // Get the values used for that branch
          auto successorOperands = branchOp.getSuccessorOperands(successorId);
          // Get the value used as input of the gate
          Value producer = successorOperands[argNumber];
          GateInput *gateInput = nullptr;

          /// If it is a block argument whose parent block has some predecessor,
          /// then the value is the output of a phi, and we add it to the list
          /// of missing phis. Otherwise, the input is a value, and it can be
          /// safely added directly.
          if (BlockArgument blockArg = dyn_cast<BlockArgument>(producer);
              blockArg && !producer.getParentBlock()->hasNoPredecessors()) {
            if (!isBlockArgAlreadyPresent(blockArg)) {
              gateInput = new GateInput((Gate *)nullptr);
              MissingPhi missingPhi = MissingPhi(gateInput, blockArg);
              missingPhi.pi->senders.insert(pred);
              phisToConnect.push_back(missingPhi);
              gateInputList.push_back(gateInput);
              operandsMissPhi.push_back(missingPhi);
            }
          } else {
            if (!isValueAlreadyPresent(dyn_cast<Value>(producer))) {
              gateInput = new GateInput(producer);
              gateInput->senders.insert(pred);
              gateInputList.push_back(gateInput);
            }
          }

          // Insert the value among the inputs of the phi
          if (gateInput)
            operands.push_back(gateInput);

          break;
        }
      }

      // If the list of operands is not empty (i.e. the phi has at least
      // one input), add it to the phis associated to that block
      if (!operands.empty()) {
        Gate *newPhi =
            new Gate(arg, operands, GateType::PhiGate, ++uniqueGateIndex);
        gatesPerBlock[&block].push_back(newPhi);
        gsaLog("convertSSAToGSA: [phase1]   created PHI idx=" +
               llvm::Twine(uniqueGateIndex) + " operands=" +
               llvm::Twine(operands.size()));
      }
    }
  }

  gsaLog("convertSSAToGSA: [phase1] done, phisToConnect=" +
         llvm::Twine(phisToConnect.size()));

  // Find the missing phi and correct the pointers
  unsigned missingCounter = 0;
  for (MissingPhi &missing : phisToConnect) {
    gsaLog("convertSSAToGSA: [reconnect] #" + llvm::Twine(missingCounter++) +
           " blockArg argNumber=" +
           llvm::Twine(missing.blockArg.getArgNumber()));
    auto list = gatesPerBlock[missing.blockArg.getParentBlock()];
    gsaLog("convertSSAToGSA: [reconnect]   candidate gates in parent block=" +
           llvm::Twine(list.size()));
    Gate **foundGate = std::find_if(list.begin(), list.end(),
                                    [&](Gate *&t) {
                                      return t->getArgumentNumber() ==
                                             missing.blockArg.getArgNumber();
                                    }

    );
    if (foundGate == list.end())
      gsaLog("convertSSAToGSA: [reconnect]   WARNING no matching phi found "
             "(assert will fire)");
    assert(foundGate != list.end() && "[GSA] Not found phi to reconnect");
    missing.pi->input = *foundGate;
  }

  gsaLog("convertSSAToGSA: [reconnect] done");

  gsaLog("convertSSAToGSA: calling convertPhiToMu");
  convertPhiToMu(region, bi);
  gsaLog("convertSSAToGSA: convertPhiToMu returned");

  gsaLog("convertSSAToGSA: calling convertPhiToGamma");
  convertPhiToGamma(region, bi);
  gsaLog("convertSSAToGSA: convertPhiToGamma returned");

  // After the conversion is done, `gatesPerBlock` will contain some phis, some
  // MUs and some GAMMAs. Since we are only interested in the last two, we can
  // remove all the phis.
  gsaLog("convertSSAToGSA: calling removePhiGates");
  removePhiGates();
  gsaLog("convertSSAToGSA: removePhiGates returned");

  printAllGates();
  gsaLog("convertSSAToGSA: exit");
}

void experimental::gsa::GSAAnalysis::convertPhiToGamma(
    Region &region, const BlockIndexing &bi) {

  gsaLog("convertPhiToGamma: enter");

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  // DFS numbers of the dominator tree are used to order the phi operands with a
  // strict weak ordering (see blockDfsLess). Refresh them once here; the tree
  // is not mutated during this function.
  llvm::DominatorTreeBase<Block, /*IsPostDom=*/false> &domTree =
      domInfo.getDomTree(&region);
  domTree.updateDFSNumbers();

  auto gatesSnapshot = gatesPerBlock;
  gsaLog("convertPhiToGamma: snapshot taken (blocks=" +
         llvm::Twine(gatesSnapshot.size()) + ")");

  // Iterate the snapshot in a DETERMINISTIC order.
  //
  // gatesPerBlock is a DenseMap keyed by Block*, whose native iteration order
  // follows pointer values and therefore varies run-to-run (ASLR). Because each
  // processed phi mutates gatesPerBlock (expandGammaTree appends gammas), the
  // *order* in which phis are processed changes the graph state seen by later
  // phis, which in turn changes their path enumeration. That made the whole
  // pass non-deterministic (same input -> different path counts each run).
  //
  // Fix: collect the blocks and sort them by their stable BlockIndexing index
  // (a total order, dominator-first), then walk them in that fixed order.
  SmallVector<Block *> orderedBlocks;
  orderedBlocks.reserve(gatesSnapshot.size());
  for (auto const &kv : gatesSnapshot)
    orderedBlocks.push_back(kv.first);
  llvm::sort(orderedBlocks.begin(), orderedBlocks.end(),
             [&](Block *a, Block *b) {
               auto ia = bi.getIndexFromBlock(a);
               auto ib = bi.getIndexFromBlock(b);
               // Blocks always have an index here; fall back to pointer order
               // only to keep a strict weak ordering if one were missing.
               if (!ia.has_value() || !ib.has_value())
                 return std::less<Block *>()(a, b);
               return ia.value() < ib.value();
             });

  // [DIAG] Dump the deterministic iteration order as stable block indices.
  // With the sort above this must now be identical across runs on the same
  // input. Pure diagnostics.
  {
    std::string order;
    for (Block *b : orderedBlocks) {
      auto i = bi.getIndexFromBlock(b);
      order += (i.has_value() ? std::to_string(i.value()) : std::string("?")) +
               " ";
    }
    gsaLog("convertPhiToGamma: [DIAG] snapshot iteration order (block idx) = [ " +
           order + "]");
  }

  unsigned blockCounter = 0;
  // For each block (deterministic order)
  for (Block *phiBlock : orderedBlocks) {
    const SmallVector<Gate *> &phis = gatesSnapshot[phiBlock];

    gsaLog("convertPhiToGamma: block #" + llvm::Twine(blockCounter++) +
           " phis=" + llvm::Twine(phis.size()));

    unsigned phiCounter = 0;
    // For each phi
    for (Gate *phi : phis) {

      gsaLog("convertPhiToGamma:   phi #" + llvm::Twine(phiCounter++) +
             " gateFn=" + llvm::Twine((int)phi->gsaGateFunction));

      // Skip if the phi is not of type `Phi`
      if (phi->gsaGateFunction != PhiGate)
        continue;

      // Sort the operands of the phi so that Bi comes before Bj if Bi dominates
      // Bj.
      //
      // NOTE: the comparator MUST be a strict weak ordering. The previous
      // `domInfo.dominates(a, b)` predicate is only a partial order (undefined
      // behavior for std::sort) and produced an inconsistent operand order.
      // That inconsistency broke the `isGreater`-based pruning inside
      // findAllPaths (which assumes later operands have strictly larger block
      // indices), causing the path enumeration to explode. Sort by dominator
      // tree DFS in-number instead: a total order that preserves the
      // "dominator first" property.
      SmallVector<GateInput *> phiOperands = phi->operands;
      gsaLog("convertPhiToGamma:     sorting " +
             llvm::Twine(phiOperands.size()) + " operands");
      llvm::sort(phiOperands.begin(), phiOperands.end(),
                 [&](GateInput *a, GateInput *b) {
                   return blockDfsLess(a->getBlock(), b->getBlock(), domTree);
                 });

      // Find the nearest common dominator among all the blocks involved
      // in the phi inputs. Also add all the blocks to the list of 'blocks to
      // avoid'
      std::vector<Block *> blocksToAvoid = {phiOperands[0]->getBlock()};
      Block *commonDominator = phiOperands[0]->getBlock();
      for (GateInput *operand : llvm::drop_begin(phiOperands)) {
        Block *bb = operand->getBlock();
        commonDominator =
            domInfo.findNearestCommonDominator(bb, commonDominator);
        blocksToAvoid.push_back(bb);
      }
      gsaLog("convertPhiToGamma:     commonDominator computed, blocksToAvoid=" +
             llvm::Twine(blocksToAvoid.size()));

      // When traversing a path, some blocks have a condition which steers the
      // control flow execution. This set stores the list of the blocks
      // accordingly. They will become boolean conditions afterwards
      DenseSet<unsigned> blocksWithConditionInPath;

      // Vector associating each input of the Phi to a boolean expression
      std::vector<std::pair<BoolExpression *, GateInput *>> expressionsList;

      unsigned operandCounter = 0;
      // For each input of the phi, compute the boolean expression which defines
      // its usage
      for (GateInput *operand : phiOperands) {

        gsaLog("convertPhiToGamma:     operand #" +
               llvm::Twine(operandCounter++));

        // Remove the current operand from the list of blocks to avoid
        blocksToAvoid.erase(blocksToAvoid.begin());

        // [DIAG] Dump the STABLE block indices (from BlockIndexing, not
        // pointers) of everything that feeds findAllPaths. If these differ
        // between two runs on the same input, the non-determinism is in how
        // commonDominator / blocksToAvoid are built (sorting / map order). If
        // they are identical but the path count differs, the non-determinism is
        // inside findAllPaths / dfsAllPaths itself. Pure diagnostics.
        {
          auto idxOf = [&](Block *b) -> int {
            if (!b)
              return -2;
            auto i = bi.getIndexFromBlock(b);
            return i.has_value() ? (int)i.value() : -1;
          };
          gsaLog("    [DIAG] commonDominator idx=" +
                 llvm::Twine(idxOf(commonDominator)) + " phiBlock idx=" +
                 llvm::Twine(idxOf(phiBlock)) + " operandBlock idx=" +
                 llvm::Twine(idxOf(operand->getBlock())));
          std::string avoid;
          for (Block *b : blocksToAvoid)
            avoid += std::to_string(idxOf(b)) + " ";
          gsaLog("    [DIAG] blocksToAvoid idx = [ " + avoid + "]");
        }

        // Find all the paths from "commonDominator" to "phiBlock" which pass
        // through operand's block but not through any of the "blocksToAvoid"
        auto allPaths = findAllPaths(commonDominator, phiBlock, bi,
                                     operand->getBlock(), blocksToAvoid);
        gsaLog("convertPhiToGamma:       findAllPaths -> " +
               llvm::Twine(allPaths.size()) + " paths");

        // Keep paths where the block before the phi matches a sender or no
        // senders are recorded.
        std::vector<std::vector<Block *>> paths;
        for (auto path : allPaths) {
          Block *prev = path[path.size() - 2];
          if (operand->senders.empty() ||
              llvm::is_contained(operand->senders, prev))
            paths.push_back(path);
        }
        gsaLog("convertPhiToGamma:       kept " + llvm::Twine(paths.size()) +
               " paths after sender filter");

        BoolExpression *phiInputCondition = BoolExpression::boolZero();

        // Sum all the conditions for each path.
        //
        // PERFORMANCE: boolMinimize() runs the full Espresso minimizer (which
        // builds a 2^(numVars) truth table internally). Calling it once per
        // path meant O(#paths) Espresso runs on an ever-growing expression
        // (e.g. 81 runs for a single operand here), which is the dominant cost.
        // The intermediate minimizations are not needed for correctness: the
        // result is logically equivalent whether we minimize after every OR or
        // once at the end. So we OR all path conditions together first and
        // minimize a single time afterwards. This also does not affect
        // `blocksWithConditionInPath`, which is populated by getPathExpression
        // independently of minimization.
        for (std::vector<Block *> &path : paths) {
          boolean::BoolExpression *condition =
              getPathExpression(path, blocksWithConditionInPath, bi);
          phiInputCondition =
              BoolExpression::boolOr(condition, phiInputCondition);
        }
        // Minimize once, after all paths have been OR-ed together.
        phiInputCondition = phiInputCondition->boolMinimize();

        // Associate the expression to the phi operand
        expressionsList.emplace_back(phiInputCondition, operand);
      }

      // Get a queue with all the necessary conditions, ordered according to
      // their basic block index
      std::vector<unsigned> conditionsToOrder;
      for (unsigned &index : blocksWithConditionInPath)
        conditionsToOrder.push_back(index);
      std::sort(conditionsToOrder.begin(), conditionsToOrder.end());
      std::queue<unsigned> conditionsOrdered;
      for (unsigned &index : conditionsToOrder)
        conditionsOrdered.push(index);

      gsaLog("convertPhiToGamma:     conditions=" +
             llvm::Twine(conditionsOrdered.size()) + ", calling "
             "expandGammaTree");

      // Expand the expressions to get the tree of gammas
      Gate *gammaRoot =
          expandGammaTree(expressionsList, conditionsOrdered, phi, bi);
      gammaRoot->isRoot = true;

      gsaLog("convertPhiToGamma:     expandGammaTree returned, rerouting users");

      // Once that a phi has been converted into a tree of gammas, all the
      // gates which used the original phi as input must be connected to the
      // root of the gamma tree
      for (auto &[bb, phis] : gatesPerBlock) {
        for (Gate *phii : phis) {
          for (GateInput *op : phii->operands) {
            if (op->isTypeGate() && op->getGate() == phi)
              op->input = gammaRoot;
          }
        }
      }
    }
  }

  gsaLog("convertPhiToGamma: exit");
}

static bool IsBlockInLoop(Block *block, CFGLoop *loop, mlir::CFGLoopInfo &li) {
  for (CFGLoop *blockLoop = li.getLoopFor(block); blockLoop;
       blockLoop = blockLoop->getParentLoop()) {
    if (blockLoop == loop)
      return true;
  }
  return false;
}

// TODO: reuse functions from FtdImplementation
BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                          CFGLoopInfo &li,
                                          const BlockIndexing &bi) {

  // Get the boolean expression associated to the block exit
  BoolExpression *blockCond =
      BoolExpression::parseSop(bi.getBlockCondition(loopExit));

  // Since we are in a loop, the terminator is a conditional branch.
  auto *terminatorOperation = loopExit->getTerminator();
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);
  assert(condBranch && "Terminator of a loop must be `cf::CondBranchOp`");

  // If the destination of the false outcome is not the block, then the
  // condition must be negated
  if (li.getLoopFor(condBranch.getFalseDest()) != loop)
    blockCond->boolNegate();

  return blockCond;
}

static BoolExpression *
getLoopExitCondition(CFGLoop *loop, std::vector<std::string> *cofactorList,
                     mlir::CFGLoopInfo &li, const BlockIndexing &bi) {

  SmallVector<Block *> exitBlocks;
  loop->getExitingBlocks(exitBlocks);

  BoolExpression *fLoopExit = BoolExpression::boolZero();

  // Get the list of all the cofactors related to possible exit conditions
  for (Block *exitBlock : exitBlocks) {
    BoolExpression *blockCond =
        getBlockLoopExitCondition(exitBlock, loop, li, bi);
    fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
    cofactorList->push_back(bi.getBlockCondition(exitBlock));
    fLoopExit = fLoopExit->boolMinimize();
  }
  // Sort the cofactors alphabetically
  std::sort(cofactorList->begin(), cofactorList->end());
  return fLoopExit;
}

void experimental::gsa::GSAAnalysis::convertPhiToMu(Region &region,
                                                    const BlockIndexing &bi) {
  gsaLog("convertPhiToMu: enter");

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  unsigned entryCounter = 0;
  // For each phi
  for (const std::pair<Block *, SmallVector<Gate *>> &entry : gatesPerBlock) {
    Block *phiBlock = entry.first;
    SmallVector<Gate *> phis = entry.second;

    gsaLog("convertPhiToMu: entry #" + llvm::Twine(entryCounter++) + " phis=" +
           llvm::Twine(phis.size()));

    unsigned phiCounter = 0;
    for (Gate *phi : phis) {

      gsaLog("convertPhiToMu:   phi #" + llvm::Twine(phiCounter++));

      // A phi can be a MU only if it is inside a loop and has at least two
      // operands
      if (!loopInfo.getLoopFor(phiBlock) || phi->operands.size() < 2)
        continue;

      // Checks whether the block of the merge is a loop header
      if (loopInfo.getLoopFor(phiBlock)->getHeader() != phiBlock)
        continue;

      gsaLog("convertPhiToMu:   phi is MU candidate (loop header)");

      // MU gate has two groups of operands: from inside and from outside the
      // loop
      SmallVector<GateInput *> initialInputs, loopInputs;

      // Separate inputs from outside the loop (initialInputs) and inside the
      // loop (loopInputs)
      for (GateInput *input : phi->operands) {
        Block *inputBlock = input->getBlock();
        if (IsBlockInLoop(inputBlock, loopInfo.getLoopFor(phiBlock), loopInfo))
          loopInputs.push_back(input);
        else
          initialInputs.push_back(input);
      }

      gsaLog("convertPhiToMu:   initialInputs=" +
             llvm::Twine(initialInputs.size()) + " loopInputs=" +
             llvm::Twine(loopInputs.size()));

      // If both initialInputs and loopInputs have at least one member, we have
      // a MU gate
      if (initialInputs.size() < 1 || loopInputs.size() < 1)
        continue;

      // MU gate has exactly one operand from inside and one from outside the
      // loop. If more than one exists, a phi gate is added to select the output
      // Note: gates created for loop inputs are flagged as MU-generated,
      // so they will later be placed in the condition block. This flagging is
      // not done for gates created from initial inputs.
      GateInput *operandInit = nullptr, *operandLoop = nullptr;

      // Handle initail input
      if (initialInputs.size() == 1)
        operandInit = initialInputs[0];
      else {
        Gate *initialPhi = new Gate(phi->result, initialInputs,
                                    GateType::PhiGate, ++uniqueGateIndex);
        gsaLog("convertPhiToMu:   inserting initialPhi into gatesPerBlock "
               "(DURING iteration over gatesPerBlock)");
        gatesPerBlock[phiBlock].push_back(initialPhi);
        operandInit = new GateInput(initialPhi);
        gateInputList.push_back(operandInit);
      }

      // Handle loop input
      if (loopInputs.size() == 1)
        operandLoop = loopInputs[0];
      else {
        // The new Phi gate has a flag muGenerated so later in the convert phi
        // to gamma it effects the place that gaama is added
        Gate *loopPhi = new Gate(phi->result, loopInputs, GateType::PhiGate,
                                 ++uniqueGateIndex, nullptr,
                                 BoolExpression::boolZero(), {}, true);
        gsaLog("convertPhiToMu:   inserting loopPhi into gatesPerBlock "
               "(DURING iteration over gatesPerBlock)");
        gatesPerBlock[phiBlock].push_back(loopPhi);
        operandLoop = new GateInput(loopPhi);
        gateInputList.push_back(operandLoop);
      }

      phi->gsaGateFunction = GateType::MuGate;
      phi->operands = {operandInit, operandLoop};

      // The block determining the MU condition is the exiting block of the
      // innermost loop the MU is in
      phi->conditionBlock =
          loopInfo.getLoopFor(phi->getBlock())->getExitingBlock();

      // Mu condition is the negation of loop exit-> if loop exit == false ? use
      // loop input : use initial input
      phi->condition = getLoopExitCondition(loopInfo.getLoopFor(phiBlock),
                                            &phi->cofactorList, loopInfo, bi)
                           ->boolNegate();
      phi->isRoot = true;

      gsaLog("convertPhiToMu:   MU gate finalized");
    }
  }

  gsaLog("convertPhiToMu: exit");
}

void experimental::gsa::GSAAnalysis::removePhiGates() {
  gsaLog("removePhiGates: enter");
  for (auto const &[block, gates] : gatesPerBlock) {

    // New vector of gates, which will contain only gamms and mus
    SmallVector<Gate *> gatesWithoutPhis;
    for (Gate *g : gates) {
      // Delete the gate in case of a phi, otherwise keep it and insert it in
      // the new list.
      if (g->gsaGateFunction == GateType::PhiGate)
        delete g;
      else
        gatesWithoutPhis.push_back(g);
    }
    // Modify the list
    gatesPerBlock[block] = gatesWithoutPhis;
  }
  gsaLog("removePhiGates: exit");
}

ArrayRef<experimental::gsa::Gate *>
experimental::gsa::GSAAnalysis::getGatesPerBlock(Block *bb) const {
  auto it = gatesPerBlock.find(bb);
  return it == gatesPerBlock.end() ? ArrayRef<Gate *>() : it->getSecond();
}

void experimental::gsa::GSAAnalysis::printAllGates() {
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      g->print();
  }
}

experimental::gsa::GSAAnalysis::~GSAAnalysis() {
  for (GateInput *gi : gateInputList)
    delete gi;
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      delete g;
  }
}

void experimental::gsa::Gate::print() {

  LLVM_DEBUG(
      auto getPhiName = [](Gate *p) -> std::string {
        switch (p->gsaGateFunction) {
        case GammaGate:
          return "GAMMA";
        case MuGate:
          return "MU";
        default:
          return "PHI";
        }
      };

      llvm::dbgs() << "[GSA] Block "; getBlock()->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " arg " << getArgumentNumber() << " type "
                   << getPhiName(this) << "_" << index;

      if (gsaGateFunction == GammaGate || gsaGateFunction == MuGate) {
        llvm::dbgs() << " condition ";
        conditionBlock->printAsOperand(llvm::dbgs());
      }

      llvm::dbgs()
      << "\n";

      for (GateInput *&op : operands) {
        if (op->isTypeValue()) {
          llvm::dbgs() << "[GSA]\t VALUE\t: ";
          op->getValue().print(llvm::dbgs());
        } else if (op->isTypeEmpty()) {
          llvm::dbgs() << "[GSA]\t EMPTY";
        } else {
          llvm::dbgs() << "[GSA]\t GATE\t: " << getPhiName(op->getGate()) << "_"
                       << op->getGate()->index;
        }

        if (!op->isTypeEmpty()) {
          llvm::dbgs() << "\t(";
          op->getBlock()->printAsOperand(llvm::dbgs());
          llvm::dbgs() << ")";
        }

        if (!op->senders.empty()) {
          llvm::dbgs() << "\t[senders: ";
          bool first = true;
          for (auto *sender : op->senders) {
            if (!first)
              llvm::dbgs() << ", ";
            sender->printAsOperand(llvm::dbgs());
            first = false;
          }
          llvm::dbgs() << "]";
        }
        llvm::dbgs() << "\n";
      });
}

Block *experimental::gsa::GateInput::getBlock() {
  if (isTypeEmpty())
    return nullptr;
  return isTypeGate() ? getGate()->getBlock() : getValue().getParentBlock();
}
