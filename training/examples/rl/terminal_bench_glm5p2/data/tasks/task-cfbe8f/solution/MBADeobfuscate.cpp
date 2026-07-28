//==============================================================================
// MBADeobfuscate.cpp
//
// Reverses 8-bit integer add obfuscation using Mixed Boolean Arithmetic.
// Recognizes the pattern:
//   (((a ^ b) + 2 * (a & b)) * 39 + 23) * 151 + 111  ==  a + b  (mod 256)
// and replaces it with a single  add i8  instruction.
//==============================================================================
#include "MBADeobfuscate.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Helper: given a BinaryOperator and an expected i8 constant (as unsigned),
// check whether one operand is that constant and return the other operand.
// Returns nullptr if no match.
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
// Try to match the full MBA-add pattern rooted at instruction I.
//
// Pattern (bottom-up from the result):
//   %r  = add  i8  111,  %m151         ; or commuted
//   %m151 = mul i8  151,  %a23         ; 151 == -105 (unsigned i8)
//   %a23  = add i8   23,  %m39
//   %m39  = mul i8   39,  %sum
//   %sum  = add i8  %xor, %m2
//   %m2   = mul i8    2,  %and
//   %and  = and i8  %A,   %B
//   %xor  = xor i8  %A,   %B           ; same operands as and
//
// On success, sets A and B to the original addition operands and returns true.
// ---------------------------------------------------------------------------
static bool matchMBAAdd(Instruction *I, Value *&A, Value *&B) {
    // Must be  add i8
    auto *Top = dyn_cast<BinaryOperator>(I);
    if (!Top || Top->getOpcode() != Instruction::Add ||
        !Top->getType()->isIntegerTy(8))
        return false;

    // 1. add 111, %mul151
    Value *V = matchConstOperand(Top, 111);
    if (!V) return false;

    // 2. mul 151, %add23
    auto *Mul151 = dyn_cast<BinaryOperator>(V);
    if (!Mul151 || Mul151->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul151, 151);          // 151 unsigned == -105 signed
    if (!V) return false;

    // 3. add 23, %mul39
    auto *Add23 = dyn_cast<BinaryOperator>(V);
    if (!Add23 || Add23->getOpcode() != Instruction::Add)
        return false;
    V = matchConstOperand(Add23, 23);
    if (!V) return false;

    // 4. mul 39, %sum
    auto *Mul39 = dyn_cast<BinaryOperator>(V);
    if (!Mul39 || Mul39->getOpcode() != Instruction::Mul)
        return false;
    V = matchConstOperand(Mul39, 39);
    if (!V) return false;

    // 5. add %xorVal, %mul2Val   (operand order may vary)
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

    // 6. mul 2, %andVal
    Value *AndVal = matchConstOperand(MulOp, 2);
    if (!AndVal) return false;

    auto *AndOp = dyn_cast<BinaryOperator>(AndVal);
    if (!AndOp || AndOp->getOpcode() != Instruction::And)
        return false;

    // 7. Verify xor and and share the same pair of operands.
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
// Pass entry point
// ---------------------------------------------------------------------------
PreservedAnalyses MBADeobfuscate::run(Function &F,
                                       FunctionAnalysisManager &) {
    bool Changed = false;

    for (auto &BB : F) {
        // After each replacement the iterator is invalidated, so restart
        // the scan from the beginning of the block.
        bool Replaced;
        do {
            Replaced = false;
            for (auto &I : BB) {
                Value *A, *B;
                if (matchMBAAdd(&I, A, B)) {
                    IRBuilder<> Builder(&I);
                    Value *NewAdd = Builder.CreateAdd(A, B);
                    I.replaceAllUsesWith(NewAdd);
                    // Recursively remove the now-dead instruction chain.
                    RecursivelyDeleteTriviallyDeadInstructions(&I);
                    Replaced = true;
                    Changed  = true;
                    break;           // restart BB iteration
                }
            }
        } while (Replaced);
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// ---------------------------------------------------------------------------
// New Pass Manager registration
// ---------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getMBADeobfuscatePluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "mba-deobfuscate", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "mba-deobfuscate") {
                            FPM.addPass(MBADeobfuscate());
                            return true;
                        }
                        return false;
                    });
            }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getMBADeobfuscatePluginInfo();
}
