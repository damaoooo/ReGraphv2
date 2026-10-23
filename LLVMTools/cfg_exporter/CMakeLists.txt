add_library(MyCFGPrinterPass MODULE MyCFGPrinter.cpp)
target_link_libraries(MyCFGPrinterPass PRIVATE LLVM OpenSSL::Crypto)
