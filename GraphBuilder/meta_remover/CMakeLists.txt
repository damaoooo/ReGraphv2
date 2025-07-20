cmake_minimum_required(VERSION 3.13)
project(StripAllMetadata)

# Find the LLVM 18.1 package to get necessary paths and variables.
# You can adjust the version number to match your installed LLVM.
find_package(LLVM 18.1 REQUIRED CONFIG)

# Add the required LLVM include directories.
include_directories(${LLVM_INCLUDE_DIRS})

# Add definitions required by LLVM (e.g., -D_GNU_SOURCE).
add_definitions(${LLVM_DEFINITIONS})

# Create our shared library plugin from the source file.
# The name "StripAllMetadataPass" is a good convention for the library target.
add_library(StripAllMetadataPass MODULE StripAllMetadata.cpp)

# Dynamically link to libLLVM.so.
# This ensures that when 'opt' loads our .so plugin, all LLVM symbols
# are resolved from the 'opt' executable itself, preventing symbol conflicts.
target_link_libraries(StripAllMetadataPass PRIVATE "${LLVM_LIBRARY_DIR}/libLLVM.so")