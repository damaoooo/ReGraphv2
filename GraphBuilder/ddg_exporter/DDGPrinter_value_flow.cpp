// DDGPrinter.cpp (值流图 Value-Flow Graph 版本)

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <algorithm>

using namespace llvm;

namespace {

// 获取一个 Value 的简短描述作为标签
std::string getValueLabel(const Value *V) {
    if (V->hasName()) {
        return std::string(V->getName());
    }
    if (isa<Instruction>(V)) {
        std::string Str;
        raw_string_ostream OS(Str);
        V->print(OS);
        std::replace(Str.begin(), Str.end(), '"', '\'');
        return Str;
    }
    if (isa<ConstantInt>(V)) {
        std::string Str;
        raw_string_ostream OS(Str);
        V->print(OS);
        return OS.str();
    }
    return ""; // 其他情况暂时不处理
}

struct ValueFlowGraphPass : public PassInfoMixin<ValueFlowGraphPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        std::string Filename = "value_flow." + F.getName().str() + ".dot";
        errs() << "Writing Value-Flow Graph to '" << Filename << "'...\n";

        std::error_code EC;
        raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);

        if (EC) {
            errs() << "Error opening file: " << EC.message() << "\n";
            return PreservedAnalyses::all();
        }

        File << "digraph \"Value Flow Graph for " << F.getName().str() << "\" {\n";
        File << "  node [shape=box, style=rounded];\n";

        // 1. 遍历所有指令和参数，创建节点
        // 使用一个 DenseSet 来避免重复创建节点
        DenseSet<const Value*> Nodes;

        // 添加函数参数作为节点
        for (const Argument &Arg : F.args()) {
            Nodes.insert(&Arg);
        }
        // 添加指令作为节点
        for (const Instruction &I : instructions(F)) {
            Nodes.insert(&I);
        }

        // 写入所有节点到 .dot 文件
        for (const Value* V : Nodes) {
             File << "  \"" << (void*)V << "\" [label=\"" << getValueLabel(V) << "\"];\n";
        }

        // 2. 遍历所有指令，创建边 (从操作数指向指令)
        for (const Instruction &I : instructions(F)) {
            // I 是一个 User，它的操作数是它使用的 Value
            for (const Use &U : I.operands()) {
                const Value *Operand = U.get();

                // 只画我们关心的节点之间的边
                if (Nodes.count(Operand)) {
                    File << "  \"" << (void*)Operand << "\" -> \"" << (void*)&I 
                         << "\" [color=blue];\n";
                }
            }
        }

        File << "}\n";

        return PreservedAnalyses::all();
    }
};

// 注册 Pass
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "ValueFlowGraphPrinter", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dot-value-flow") { // 我们给它起个新名字
                        FPM.addPass(ValueFlowGraphPass());
                        return true;
                    }
                    return false;
                });
        }};
}

} // 匿名命名空间结束