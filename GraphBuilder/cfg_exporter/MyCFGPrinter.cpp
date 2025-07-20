// MyCFGPrinter.cpp (基于你成功的DDGPrinter.cpp适配而成)

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h" // 确保新的PassManager头文件存在
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/BranchProbabilityInfo.h" // 包含分支概率分析

#include <string>
#include <algorithm>
#include <iomanip>
#include <openssl/sha.h>
#include <sstream> // 包含 sstream 以使用 stringstream

using namespace llvm;

// 命令行选项，让Python传入输出目录
static cl::opt<std::string> CfgDotOutputDir(
    "cfg-dot-output-dir",
    cl::desc("Specify the output directory for CFG DOT files"),
    cl::value_desc("directory"),
    cl::init("."));

namespace {

// 复用你成功的SHA256哈希函数
std::string getSha256HashedName(const std::string& long_name) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(long_name.c_str()), long_name.length(), digest);
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return ss.str();
}

struct MyCFGPrinterPass : public PassInfoMixin<MyCFGPrinterPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        // 从分析管理器中获取分支概率信息
        BranchProbabilityInfo &BPI = FAM.getResult<BranchProbabilityAnalysis>(F);
        
        // --- 生成安全的文件名 (逻辑与你的DDG Pass一致) ---
        std::string funcName = F.getName().str();
        if (funcName.empty()) funcName = "anonymous_function";
        std::string hashedName = getSha256HashedName(funcName);
        std::string fullPath = CfgDotOutputDir.getValue() + "/cfg_" + hashedName + ".dot";

        errs() << "Write CFG to '" << fullPath << "'\n";
        
        std::error_code EC;
        raw_fd_ostream File(fullPath, EC, sys::fs::OF_Text);
        if (EC) {
            errs() << "Error opening file '" << fullPath << "': " << EC.message() << "\n";
            return PreservedAnalyses::all();
        }

        // --- 核心逻辑：生成CFG的DOT描述 ---
        File << "digraph \"CFG for '" << funcName << "' function\" {\n";
        File << "    node [shape=box, style=rounded];\n";

        // 1. 遍历所有基本块，生成节点
        for (const BasicBlock &BB : F) {
            std::string NodeLabelStr;
            raw_string_ostream OS(NodeLabelStr);
            BB.print(OS);
            
            // 使用反斜杠转义双引号以避免DOT语法错误
            std::string labelEscaped;
            labelEscaped.reserve(OS.str().size());
            for (char c : OS.str()) {
                if (c == '"') labelEscaped += "\\\"";
                // 处理换行符 
                else if (c == '\n') labelEscaped += "\\n";
                else labelEscaped += c;
            }

            File << "    Node" << &BB << " [label=\"" << labelEscaped << "\"];\n";
        }

        // 2. 再次遍历，生成边和边的标签
        for (const BasicBlock &BB : F) {
            const Instruction *TInst = BB.getTerminator();
            for (unsigned i = 0, NSucc = TInst->getNumSuccessors(); i < NSucc; ++i) {
                BasicBlock *Succ = TInst->getSuccessor(i);
                
                File << "    Node" << &BB << " -> Node" << Succ;

                if (TInst->getNumSuccessors() > 1) {
                    BranchProbability Prob = BPI.getEdgeProbability(&BB, Succ);
                    
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(1) 
                       << (Prob.getNumerator() / (double)Prob.getDenominator() * 100.0) << "%";
                    
                    File << " [label=\"" << ss.str() << "\"]";
                }
                
                File << ";\n";
            }
        }

        File << "}\n";

        return PreservedAnalyses::all();
    }
};

// 插件入口点 (逻辑与你的DDG Pass一致)
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "MyCFGPrinter", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dot-my-cfg") {
                        FPM.addPass(MyCFGPrinterPass());
                        return true;
                    }
                    return false;
                });
        }};
}

} // end namespace