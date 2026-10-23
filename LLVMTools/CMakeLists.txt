cmake_minimum_required(VERSION 3.13)
project(ReLLLLVMTools LANGUAGES C CXX)

find_package(LLVM 18.1 REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Shared LLVM avoids duplicate command-line option registration in opt plugins.
add_subdirectory(cfg_exporter)
add_subdirectory(ddg_exporter)
add_subdirectory(meta_remover)
add_subdirectory(split_functions)
