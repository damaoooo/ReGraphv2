#!/bin/bash

# ==============================================================================
# split_llvm_ir_cli.sh (v4 - CLI Optimized)
#
# A robust script to split a large LLVM IR file (.ll or .bc) into smaller
# files, one for each function.
#
# Features:
# - Uses `llvm-nm` to reliably list defined functions.
# - Uses the SHA1 hash of the function name as the filename to prevent
#   "File name too long" errors.
# - Idempotent: Skips extraction if the output file already exists, making
#   reruns fast.
# - Overwrites the function map on each run to ensure it's always clean.
# - Minimized output for clean batch processing.
#
# Dependencies: llvm-as, llvm-nm, llvm-extract, sha1sum
# ==============================================================================

# --- 0. Script Setup ---
# Exit immediately if a command exits with a non-zero status.
set -e
# The return value of a pipeline is the status of the last command to exit
# with a non-zero status, or zero if no command exited with a non-zero status.
set -o pipefail

# --- 1. Argument and Tool Checks ---

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments." >&2
    echo "Usage: $0 <input_llvm_file.ll|.bc> <output_directory>" >&2
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found." >&2
    exit 1
fi

# Check for all required LLVM and core utilities
for tool in llvm-as llvm-nm llvm-extract sha1sum; do
  if ! command -v "$tool" &> /dev/null; then
    echo "Error: Required tool '$tool' not found in PATH." >&2
    exit 1
  fi
done


# --- 2. Preparation ---

# Create a temporary file for the bitcode version of the input.
# Using mktemp is secure and avoids race conditions.
TEMP_BC_FILE=$(mktemp)

# Set up a trap to ensure the temporary file is deleted when the script exits,
# for any reason (success, failure, or interrupt).
trap 'rm -f "$TEMP_BC_FILE"' EXIT

# Determine if the input file is .ll (LLVM IR) or .bc (bitcode) and prepare it.
if [[ "$INPUT_FILE" == *.ll ]]; then
  llvm-as "$INPUT_FILE" -o "$TEMP_BC_FILE"
elif [[ "$INPUT_FILE" == *.bc ]]; then
  cp "$INPUT_FILE" "$TEMP_BC_FILE"
else
  echo "Error: Input file must have a .ll or .bc extension." >&2
  exit 1
fi

# --- 3. Extraction and Processing ---

# Ensure the output directory exists and create a clean map file for this run.
mkdir -p "$OUTPUT_DIR"
MAP_FILE="${OUTPUT_DIR}/function_map.csv"
NO_FUNCTIONS_MARKER="${OUTPUT_DIR}/.no_functions"
echo "OriginalFunctionName,HashedFileName" > "$MAP_FILE"
rm -f "$NO_FUNCTIONS_MARKER"

# Run llvm-nm on the bitcode file. Some objects legitimately have no defined
# text symbols; do not let grep/pipefail turn that into a task failure.
FUNCTION_LIST=$(llvm-nm "$TEMP_BC_FILE" | awk '$2 == "T" {print $3}' || true)

if [ -z "$FUNCTION_LIST" ]; then
    touch "$NO_FUNCTIONS_MARKER"
    echo "Warning: No defined functions found in '$INPUT_FILE'. Nothing to do." >&2
    exit 0
fi

# Loop through each function name and run llvm-extract.
for func_name in $FUNCTION_LIST; do
  # 1. Generate a safe filename by hashing the original function name.
  #    `echo -n` prevents hashing the trailing newline.
  #    `awk '{print $1}'` isolates the hash value.
  safe_filename=$(echo -n "$func_name" | sha1sum | awk '{print $1}')
  output_filename="${OUTPUT_DIR}/${safe_filename}.ll"

  # 2. Always write the mapping to this run's CSV file.
  #    Quoting handles potential special characters in names.
  echo "\"$func_name\",\"${safe_filename}.ll\"" >> "$MAP_FILE"

  # 3. If the file already exists, skip it to make reruns efficient.
  if [ -f "$output_filename" ]; then
    continue
  else
    # The -S flag outputs human-readable LLVM IR (.ll).
    llvm-extract -S --func="$func_name" "$TEMP_BC_FILE" -o "$output_filename"
  fi
done

# --- 4. Completion ---

# Final confirmation message.
echo "Processing complete. Output in '$OUTPUT_DIR'. Map file at '$MAP_FILE'."

exit 0