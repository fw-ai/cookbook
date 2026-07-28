#ifndef AFFINE_MBA_DEOBFUSCATE_H
#define AFFINE_MBA_DEOBFUSCATE_H

#include "llvm/IR/PassManager.h"

struct AffineMBADeobfuscate : public llvm::PassInfoMixin<AffineMBADeobfuscate> {
    llvm::PreservedAnalyses run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &);
    static bool isRequired() { return true; }
};

#endif
