add_library(DDGPrinter MODULE DDGPrinter.cpp)
target_link_libraries(DDGPrinter PRIVATE LLVM OpenSSL::Crypto)
