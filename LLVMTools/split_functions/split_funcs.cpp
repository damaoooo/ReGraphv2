// Reuse a bitcode buffer and LLVM's extraction passes for functions in one chunk.
// usage: split_funcs <input.bc> <out_dir> < names   (writes <out_dir>/<sha1>.ll, sha1 computed by caller)
// stdin lines: "<name>\t<output file name>"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO/ExtractGV.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Config/llvm-config.h"
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;

static void runPasses(ModulePassManager &passes, Module &module) {
  LoopAnalysisManager loopAnalyses;
  FunctionAnalysisManager functionAnalyses;
  CGSCCAnalysisManager callGraphAnalyses;
  ModuleAnalysisManager moduleAnalyses;
  PassBuilder passBuilder;
  passBuilder.registerModuleAnalyses(moduleAnalyses);
  passBuilder.registerCGSCCAnalyses(callGraphAnalyses);
  passBuilder.registerFunctionAnalyses(functionAnalyses);
  passBuilder.registerLoopAnalyses(loopAnalyses);
  passBuilder.crossRegisterProxies(loopAnalyses, functionAnalyses, callGraphAnalyses, moduleAnalyses);
  passes.run(module, moduleAnalyses);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    outs() << "split_funcs LLVM " << LLVM_VERSION_STRING << "\n";
    return 0;
  }
  ExitOnError exitOnError("split_funcs: ");
  if (argc != 3) { errs() << "usage: split_funcs <input.bc> <out_dir> < names\n"; return 2; }
  auto bitcodeBuffer = exitOnError(errorOrToExpected(MemoryBuffer::getFile(argv[1])));
  std::string line;
  while (std::getline(std::cin, line)) {
    auto tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || line.find('\t', tab + 1) != std::string::npos) {
      errs() << "invalid manifest row\n"; return 2;
    }
    auto filename = line.substr(tab + 1);
    if (filename.size() != 43 || filename.substr(40) != ".ll" ||
        filename.substr(0, 40).find_first_not_of("0123456789abcdef") != std::string::npos) {
      errs() << "invalid output filename\n"; return 2;
    }
    std::string name = line.substr(0, tab), out = std::string(argv[2]) + "/" + line.substr(tab + 1);
    LLVMContext context;
    std::unique_ptr<Module> module = exitOnError(getLazyBitcodeModule(bitcodeBuffer->getMemBufferRef(), context));
    exitOnError(module->materializeMetadata());
    GlobalValue *function = module->getFunction(name);
    if (!function) { errs() << "split_funcs: function not found: " << name << "\n"; return 1; }
    exitOnError(function->materialize());
    {
      ModulePassManager passes;
      std::vector<GlobalValue *> selectedFunctions{function};
      passes.addPass(ExtractGVPass(selectedFunctions, false, false));
      runPasses(passes, *module);
      exitOnError(module->materializeAll());
    }
    module->setModuleIdentifier("regraph-function");
    std::error_code outputError;
    ToolOutputFile outputFile(out, outputError, sys::fs::OF_None);
    if (outputError) { errs() << "split_funcs: " << outputError.message() << "\n"; return 1; }
    ModulePassManager passes;
    passes.addPass(GlobalDCEPass());
    passes.addPass(StripDeadDebugInfoPass());
    passes.addPass(StripDeadPrototypesPass());
    passes.addPass(PrintModulePass(outputFile.os(), "", false));
    runPasses(passes, *module);
    outputFile.os().flush();
    if (outputFile.os().has_error()) { errs() << "output write failed\n"; return 1; }
    outputFile.keep();
  }
  return std::cin.bad() ? 1 : 0;
}
