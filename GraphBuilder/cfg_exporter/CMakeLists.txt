# CMakeLists.txt (最终版)

cmake_minimum_required(VERSION 3.13)
project(MyCFGPrinter)

# 找到 LLVM 18
find_package(LLVM 18.1 REQUIRED CONFIG)

# 找到 OpenSSL (用于SHA256)
find_package(OpenSSL REQUIRED)

# 添加 LLVM 的头文件路径
include_directories(${LLVM_INCLUDE_DIRS})

# 添加 LLVM 需要的编译定义
add_definitions(${LLVM_DEFINITIONS})

# 创建我们的共享库插件，源文件名为 MyCFGPrinter.cpp
add_library(MyCFGPrinterPass MODULE MyCFGPrinter.cpp)

# 将插件链接到 libLLVM.so 和 OpenSSL 的加密库
target_link_libraries(MyCFGPrinterPass PRIVATE
    "${LLVM_LIBRARY_DIR}/libLLVM.so"
    OpenSSL::Crypto
)
