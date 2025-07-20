// DDGPrinter.cpp (最终版 - 注入并显示唯一ID)

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <algorithm>
#include <iomanip>
#include <openssl/sha.h>

using namespace llvm;

namespace {


// 辅助函数：将长字符串转换为SHA256哈希值
std::string getSha256HashedName(const std::string& long_name) {
    unsigned char digest[SHA256_DIGEST_LENGTH]; // SHA256摘要长度是32字节
    
    SHA256(reinterpret_cast<const unsigned char*>(long_name.c_str()), long_name.length(), digest);
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    
    return ss.str(); // 返回一个64个字符的十六进制字符串
}

// 获取一个 Value 的标签，现在加入了我们自己注入的ID
std::string getValueLabel(const Value *V) {
    std::string Label;
    if (V->hasName()) {
        Label = std::string(V->getName());
    } else if (isa<Instruction>(V)) {
        raw_string_ostream OS(Label);
        V->print(OS);
    } else if (isa<ConstantInt>(V)) {
        raw_string_ostream OS(Label);
        V->print(OS);
    }
    
    std::replace(Label.begin(), Label.end(), '"', '\'');
    
    // --- 读取我们自己注入的ID ---
    if (const Instruction *I = dyn_cast<Instruction>(V)) {
        // 尝试获取名为 "my_id" 的元数据
        if (MDNode *MD = I->getMetadata("my_id")) {
            if (MD->getNumOperands() > 0) {
                if (auto *Val = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))) {
                    if (auto *IntVal = dyn_cast<ConstantInt>(Val->getValue())) {
                        uint64_t ID = IntVal->getZExtValue();
                        // 将ID附加到标签上
                        Label += "\n(ID: " + std::to_string(ID) + ")";
                    }
                }
            }
        }
    }
    
    return Label;
}

struct IDInjectorAndGraphPass : public PassInfoMixin<IDInjectorAndGraphPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        
        // --- 第一步：遍历所有指令并注入唯一ID ---
        LLVMContext &Ctx = F.getContext();
        unsigned InstID = 0;
        for (Instruction &I : instructions(F)) {
            // 创建一个包含整数ID的元数据
            Constant *IDVal = ConstantInt::get(Type::getInt32Ty(Ctx), InstID++);
            Metadata *MDVal = ConstantAsMetadata::get(IDVal);
            MDNode *N = MDNode::get(Ctx, MDVal);
            
            // 将元数据附加到指令上，并给它一个名字 "my_id"
            I.setMetadata("my_id", N);
        }

        // --- 第二步：生成值流图（和之前一样） ---
        std::string originalFuncName = F.getName().str();
        if (originalFuncName.empty()) {
            originalFuncName = "anonymous_function";
        }
        std::string hashedFuncName = getSha256HashedName(originalFuncName); // <-- 调用SHA256哈希函数
        std::string dotFileName = "ddg_" + hashedFuncName + ".dot";
        errs() << "Writing ID-tagged graph to '" << dotFileName << "'...\n";

        std::error_code EC;
        raw_fd_ostream File(dotFileName, EC, sys::fs::OF_Text);

        if (EC) {
            errs() << "Error opening file: " << EC.message() << "\n";
            return PreservedAnalyses::all();
        }

        File << "digraph \"Value Flow Graph for " << F.getName().str() << "\" {\n";
        File << "  node [shape=box, style=rounded];\n";

        DenseSet<const Value*> Nodes;
        for (const Argument &Arg : F.args()) { Nodes.insert(&Arg); }
        for (const Instruction &I : instructions(F)) { Nodes.insert(&I); }

        for (const Value* V : Nodes) {
             File << "  \"" << (void*)V << "\" [label=\"" << getValueLabel(V) << "\"];\n";
        }

        for (const Instruction &I : instructions(F)) {
            for (const Use &U : I.operands()) {
                const Value *Operand = U.get();
                if (Nodes.count(Operand)) {
                    File << "  \"" << (void*)Operand << "\" -> \"" << (void*)&I 
                         << "\" [color=blue];\n";
                }
            }
        }

        File << "}\n";

        // 我们修改了IR（添加了元数据），所以不能保留所有分析结果
        return PreservedAnalyses::none();
    }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "IDInjectorAndGraphPrinter", "v0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager  &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "dot-id-graph") {
                        FPM.addPass(IDInjectorAndGraphPass());
                        return true;
                    }
                    return false;
                });
        }};
}

}