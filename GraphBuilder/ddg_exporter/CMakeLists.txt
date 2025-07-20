# CMakeLists.txt (The Definitive Linking Fix)

cmake_minimum_required(VERSION 3.13)
project(DDGPrinter)

# We only need find_package to get the include paths.
find_package(LLVM 18.1 REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

# Manually add the include paths from LLVM.
include_directories(${LLVM_INCLUDE_DIRS})

# Add definitions required by LLVM.
add_definitions(${LLVM_DEFINITIONS})

# Create our shared library plugin.
add_library(DDGPrinter MODULE DDGPrinter.cpp)

# THE CRITICAL CHANGE:
# We link directly to the main libLLVM.so shared library.
# This forces a dynamic link and ensures all LLVM symbols are
# provided by the 'opt' executable at runtime, solving the
# "registered more than once" error.
target_link_libraries(DDGPrinter PRIVATE 
                        "${LLVM_LIBRARY_DIR}/libLLVM.so"
                        OpenSSL::Crypto
                        )