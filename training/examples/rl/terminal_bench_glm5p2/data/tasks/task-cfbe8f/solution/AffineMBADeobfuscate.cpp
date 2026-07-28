//==============================================================================
// AffineMBADeobfuscate.cpp
//
// Reverses two Affine-MBA obfuscation variants on i8 arithmetic:
//
// Addition pattern:
//   (((a ^ b) + 2*(a & b)) * 173 + 91) * 37 + 217  ==  a + b  (mod 256)
//   where 173*37 = 1 (mod 256) and 91*37+217 = 0 (mod 256)
//   Constants in signed i8: 173=-83, 217=-39
//
// Subtraction pattern:
//   ((2*(a & ~b) - (a ^ b)) * 59 + 67) * 243 + 103  ==  a - b  (mod 256)
//   where 59*243 = 1 (mod 256) and 67*243+103 = 0 (mod 256)
//   Constants in signed i8: 243=-13
//==============================================================================
#include "AffineMBADeobfuscate.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Helper: given a BinaryOperator and an expected i8 constant (as unsigned),
// check whether one operand is that constant and return the other operand.
// ---------------------------------------------------------------------------
static Value *matchConstOperand(BinaryOperator *BO, uint8_t Expected) {
    for (unsigned i = 0; i < 2; ++i) {
        if (auto *CI = dyn_cast<ConstantInt>(BO->getOperand(i))) {
            if (static_cast<uint8_t>(CI->getZExtValue()) == Expected)
                return BO->getOperand(1 - i);
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Match the addition Affine-MBA pattern rooted at instruction I.
//
// Pattern (bottom-up):
//   %r    = add  i8   217,  %m37         (217 = -39 signed)
//   %m37  = mul  i8    37,  %a91
//   %a91  = add  i8    91,  %m173
//   %m173 = mul  i8   173,  %sum         (173 = -83 signed)
//   %sum  = add  i8  %xor,  %m2
//   %m2   = mul  i8     2,  %and
//   %and  = and  i8    %A,   %B
//   %xor  = xor  i8    %A,   %B          (same operand pair as and)
// ---------------------------------------------------------------------------
static bool matchMBAAdd(Instruction *I, Value *&A, Value *&B) {
    auto *Top = dyn_cast<BinaryOperator>(I);
    if (!Top || Top->getOpcode() != Instruction::Add ||
        !Top->getType()->isIntegerTy(8))
        return false;

    // Step 8: add 217 (=-39), %mul37
    Value *V = matchConstOperand(Top, 217);
    if (!V) return false;

    // Step 7: mul 37, %add91
    auto *Mul37 = dyn_cast<BinaryOperator>(V);
    if (!Mul37 || Mul37->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul37, 37);
    if (!V) return false;

    // Step 6: add 91, %mul173
    auto *Add91 = dyn_cast<BinaryOperator>(V);
    if (!Add91 || Add91->getOpcode() != Instruction::Add)
        return false;
    V = matchConstOperand(Add91, 91);
    if (!V) return false;

    // Step 5: mul 173 (=-83), %sum
    auto *Mul173 = dyn_cast<BinaryOperator>(V);
    if (!Mul173 || Mul173->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul173, 173);
    if (!V) return false;

    // Step 4: add %xor, %mul2
    auto *Sum = dyn_cast<BinaryOperator>(V);
    if (!Sum || Sum->getOpcode() != Instruction::Add)
        return false;

    BinaryOperator *XorOp = nullptr, *MulOp = nullptr;
    auto *S0 = dyn_cast<BinaryOperator>(Sum->getOperand(0));
    auto *S1 = dyn_cast<BinaryOperator>(Sum->getOperand(1));
    if (S0 && S1) {
        if (S0->getOpcode() == Instruction::Xor &&
            S1->getOpcode() == Instruction::Mul) {
            XorOp = S0; MulOp = S1;
        } else if (S0->getOpcode() == Instruction::Mul &&
                   S1->getOpcode() == Instruction::Xor) {
            XorOp = S1; MulOp = S0;
        }
    }
    if (!XorOp || !MulOp) return false;

    // Step 3: mul 2, %and
    Value *AndVal = matchConstOperand(MulOp, 2);
    if (!AndVal) return false;

    auto *AndOp = dyn_cast<BinaryOperator>(AndVal);
    if (!AndOp || AndOp->getOpcode() != Instruction::And)
        return false;

    // Verify xor and and share the same pair of operands
    Value *XA = XorOp->getOperand(0), *XB = XorOp->getOperand(1);
    Value *AA = AndOp->getOperand(0), *AB = AndOp->getOperand(1);

    if ((XA == AA && XB == AB) || (XA == AB && XB == AA)) {
        A = XA;
        B = XB;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Match the subtraction Affine-MBA pattern rooted at instruction I.
//
// Pattern (bottom-up):
//   %r    = add  i8   103,  %m243
//   %m243 = mul  i8   243,  %a67         (243 = -13 signed)
//   %a67  = add  i8    67,  %m59
//   %m59  = mul  i8    59,  %diff
//   %diff = sub  i8  %dbl,  %xor         (NOT commutative)
//   %dbl  = mul  i8     2,  %masked
//   %masked = and i8  %A,   %notB
//   %notB = xor  i8    %B,   -1
//   %xor  = xor  i8    %A,   %B          (same A, B as above)
// ---------------------------------------------------------------------------
static bool matchMBASub(Instruction *I, Value *&A, Value *&B) {
    auto *Top = dyn_cast<BinaryOperator>(I);
    if (!Top || Top->getOpcode() != Instruction::Add ||
        !Top->getType()->isIntegerTy(8))
        return false;

    // Step 9: add 103, %mul243
    Value *V = matchConstOperand(Top, 103);
    if (!V) return false;

    // Step 8: mul 243 (=-13), %add67
    auto *Mul243 = dyn_cast<BinaryOperator>(V);
    if (!Mul243 || Mul243->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul243, 243);
    if (!V) return false;

    // Step 7: add 67, %mul59
    auto *Add67 = dyn_cast<BinaryOperator>(V);
    if (!Add67 || Add67->getOpcode() != Instruction::Add)
        return false;
    V = matchConstOperand(Add67, 67);
    if (!V) return false;

    // Step 6: mul 59, %diff
    auto *Mul59 = dyn_cast<BinaryOperator>(V);
    if (!Mul59 || Mul59->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul59, 59);
    if (!V) return false;

    // Step 5: sub %dbl, %xorAB  (sub is NOT commutative)
    auto *Diff = dyn_cast<BinaryOperator>(V);
    if (!Diff || Diff->getOpcode() != Instruction::Sub)
        return false;

    auto *DblOp = dyn_cast<BinaryOperator>(Diff->getOperand(0));
    auto *XorOp = dyn_cast<BinaryOperator>(Diff->getOperand(1));
    if (!DblOp || !XorOp) return false;

    // Step 3: mul 2, %masked
    if (DblOp->getOpcode() != Instruction::Mul) return false;
    Value *MaskedVal = matchConstOperand(DblOp, 2);
    if (!MaskedVal) return false;

    // Step 2: and %A, %notB
    auto *AndOp = dyn_cast<BinaryOperator>(MaskedVal);
    if (!AndOp || AndOp->getOpcode() != Instruction::And)
        return false;

    // Step 4: xor %A, %B
    if (XorOp->getOpcode() != Instruction::Xor) return false;

    // Step 1: identify which operand of And is ~B (xor %B, -1)
    // and which is A. Then verify XorOp uses {A, B}.
    Value *OrigA = nullptr, *OrigB = nullptr;

    for (unsigned i = 0; i < 2; ++i) {
        Value *NotBCandidate = AndOp->getOperand(i);
        Value *ACandidate = AndOp->getOperand(1 - i);

        auto *NotBOp = dyn_cast<BinaryOperator>(NotBCandidate);
        if (!NotBOp || NotBOp->getOpcode() != Instruction::Xor)
            continue;

        // Check xor %B, -1  (unsigned 255)
        Value *BCandidate = matchConstOperand(NotBOp, 255);
        if (!BCandidate) continue;

        // Verify XorOp has operands {ACandidate, BCandidate}
        Value *X0 = XorOp->getOperand(0), *X1 = XorOp->getOperand(1);
        if ((X0 == ACandidate && X1 == BCandidate) ||
            (X0 == BCandidate && X1 == ACandidate)) {
            OrigA = ACandidate;
            OrigB = BCandidate;
            break;
        }
    }

    if (!OrigA || !OrigB) return false;

    A = OrigA;
    B = OrigB;
    return true;
}

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------
PreservedAnalyses AffineMBADeobfuscate::run(Function &F,
                                             FunctionAnalysisManager &) {
    bool Changed = false;

    for (auto &BB : F) {
        bool Replaced;
        do {
            Replaced = false;
            for (auto &I : BB) {
                Value *A, *B;
                bool IsAdd = matchMBAAdd(&I, A, B);
                bool IsSub = !IsAdd && matchMBASub(&I, A, B);

                if (IsAdd || IsSub) {
                    IRBuilder<> Builder(&I);
                    Value *Replacement;
                    if (IsAdd)
                        Replacement = Builder.CreateAdd(A, B);
                    else
                        Replacement = Builder.CreateSub(A, B);

                    I.replaceAllUsesWith(Replacement);
                    RecursivelyDeleteTriviallyDeadInstructions(&I);
                    Replaced = true;
                    Changed  = true;
                    break;
                }
            }
        } while (Replaced);
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// ---------------------------------------------------------------------------
// New Pass Manager registration
// ---------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getAffineMBADeobfuscatePluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "affine-mba-deobfuscate",
            LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "affine-mba-deobfuscate") {
                            FPM.addPass(AffineMBADeobfuscate());
                            return true;
                        }
                        return false;
                    });
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getAffineMBADeobfuscatePluginInfo();
}
