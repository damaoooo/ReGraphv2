#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
// #include "llvm/IR/MDNode.h"  <-- 删除这一行

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// Pass logic (保持不变)
struct StripAllMetadataPass : public PassInfoMixin<StripAllMetadataPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        for (Function &F : M) {
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    if (I.hasMetadata()) {
                        SmallVector<std::pair<unsigned, MDNode *>, 4> AllMDs;
                        I.getAllMetadata(AllMDs);
                        for (const auto &MD : AllMDs) {
                            I.setMetadata(MD.first, nullptr);
                        }
                    }
                }
            }
        }
        return PreservedAnalyses::all();
    }
};

// Plugin entry point (保持不变)
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "StripAllMetadata", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "strip-all-metadata") {
                        MPM.addPass(StripAllMetadataPass());
                        return true;
                    }
                    return false;
                }
            );
        }
    };
}