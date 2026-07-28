#ifndef MBA_DEOBFUSCATE_H
#define MBA_DEOBFUSCATE_H

#include "llvm/IR/PassManager.h"

struct MBADeobfuscate : public llvm::PassInfoMixin<MBADeobfuscate> {
    llvm::PreservedAnalyses run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &);
    static bool isRequired() { return true; }
};

#endif
