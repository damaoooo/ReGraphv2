// DDGPrinter.cpp (最终版 - 增加边去重功能)

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <algorithm>
#include <iomanip>
#include <set>      // <-- 新增头文件，用于去重
#include <utility>  // <-- 新增头文件，用于std::pair

using namespace llvm;

namespace {

// 获取指令的标签，包含指令文本和我们注入的ID
std::string getInstructionLabel(const Instruction *I) {
    std::string Label;
    raw_string_ostream OS(Label);
    I->print(OS);
    
    std::replace(Label.begin(), Label.end(), '"', '\'');
    
    if (MDNode *MD = I->getMetadata("my_id")) {
        if (MD->getNumOperands() > 0) {
            if (auto *Val = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))) {
                if (auto *IntVal = dyn_cast<ConstantInt>(Val->getValue())) {
                    uint64_t ID = IntVal->getZExtValue();
                    Label += "\n(ID: " + std::to_string(ID) + ")";
                }
            }
        }
    }
    
    return Label;
}

struct StatementDGPass : public PassInfoMixin<StatementDGPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        
        LLVMContext &Ctx = F.getContext();
        unsigned InstID = 0;
        for (Instruction &I : instructions(F)) {
            Constant *IDVal = ConstantInt::get(Type::getInt32Ty(Ctx), InstID++);
            MDNode *N = MDNode::get(Ctx, {ConstantAsMetadata::get(IDVal)});
            I.setMetadata("my_id", N);
        }

        auto &DI = FAM.getResult<DependenceAnalysis>(F);

        std::string Filename = "statement_ddg." + F.getName().str() + ".dot";
        errs() << "Writing Statement-level DDG to '" << Filename << "'...\n";

        std::error_code EC;
        raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);

        if (EC) {
            errs() << "Error opening file: " << EC.message() << "\n";
            return PreservedAnalyses::none();
        }

        File << "digraph \"Statement-level DDG for " << F.getName().str() << "\" {\n";
        File << "  node [shape=box, style=rounded];\n";

        for (const Instruction &I : instructions(F)) {
             File << "  \"" << (void*)&I << "\" [label=\"" << getInstructionLabel(&I) << "\"];\n";
        }

        // --- 新增的去重逻辑 ---
        // 1. 创建一个集合来存储已经画过的边
        std::set<std::pair<const Instruction*, const Instruction*>> PrintedEdges;

        for (Instruction &Src : instructions(F)) {
            for (Instruction &Dst : instructions(F)) {
                if (&Src == &Dst) continue;
                
                if (auto D = DI.depends(&Src, &Dst, true)) {
                    if (D->isFlow()) {
                        // 2. 在画边之前，检查是否已经画过
                        if (PrintedEdges.find({&Src, &Dst}) == PrintedEdges.end()) {
                            // 3. 如果没画过，就画出来，并记录下来
                            File << "  \"" << (void*)&Src << "\" -> \"" << (void*)&Dst 
                                 << "\" [color=blue, label=\"RAW\"];\n";
                            PrintedEdges.insert({&Src, &Dst});
                        }
                    }
                }
            }
        }

        File << "}\n";

        return PreservedAnalyses::none();
    }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "StatementDDGPrinter", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dot-statement-ddg") {
                        FPM.addPass(StatementDGPass());
                        return true;
                    }
                    return false;
                });
        }};
}
}