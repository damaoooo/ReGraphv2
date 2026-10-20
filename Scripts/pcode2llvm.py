#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer


LOGGER = logging.getLogger("pcode2llvm")

SCRIPT_DIR = Path(__file__).resolve().parent
GHIDRA_SCRIPT_DIR = SCRIPT_DIR / "ghidra"
DEFAULT_GHIDRA_HEADLESS = Path("/home/damaoooo/Downloads/ghidra_12.0/support/analyzeHeadless")

LLVM_TERMINATOR_PREFIXES = (
    "br ",
    "ret ",
    "switch ",
    "indirectbr ",
    "invoke ",
    "resume ",
    "unreachable",
)


class UnsupportedPcode(Exception):
    pass


class FunctionLiftError(Exception):
    pass


@dataclass(frozen=True)
class LLVMValue:
    width: int
    text: str
    const: Optional[int] = None
    ty: Optional[str] = None

    @property
    def llvm_type(self) -> str:
        return self.ty or f"i{self.width}"

    @property
    def is_integer(self) -> bool:
        return self.ty is None or self.ty.startswith("i")


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    if text.lower().startswith("0x"):
        return int(text, 16)
    return int(text, 10)


def _mask(width: int) -> int:
    return (1 << width) - 1


def _normalize_width(width: int) -> int:
    return max(int(width), 1)


def _const(width: int, value: int) -> LLVMValue:
    width = _normalize_width(width)
    value &= _mask(width)
    if width == 1:
        return LLVMValue(width, "true" if value else "false", value)
    return LLVMValue(width, str(value), value)


def _undef(width: int) -> LLVMValue:
    return LLVMValue(_normalize_width(width), "undef", None)


def _undef_typed(ty: str) -> LLVMValue:
    return LLVMValue(_type_width(ty), "undef", None, None if ty.startswith("i") else ty)


def _type_width(ty: str) -> int:
    if ty == "float":
        return 32
    if ty == "double":
        return 64
    if ty.startswith("i"):
        return _normalize_width(int(ty[1:]))
    if ty == "ptr":
        return 64
    raise ValueError(f"unsupported LLVM type: {ty}")


def _float_type_for_width(width: int) -> str:
    if width == 32:
        return "float"
    if width == 64:
        return "double"
    raise UnsupportedPcode(f"unsupported floating-point width: {width}")


def _varnode_width(varnode: Optional[Dict[str, Any]], pointer_bits: int) -> int:
    if not varnode:
        return pointer_bits
    size = _parse_int(varnode.get("size"), 0)
    if size <= 0:
        return pointer_bits
    return _normalize_width(size * 8)


def _varnode_key(varnode: Dict[str, Any]) -> str:
    return "{}:{}:{}:{}".format(
        varnode.get("space", ""),
        varnode.get("offset", "0"),
        varnode.get("size", "0"),
        varnode.get("pc_address", ""),
    )


def _sanitize_symbol_name(name: str, fallback: str) -> str:
    name = name or fallback
    name = re.sub(r"[^A-Za-z0-9_$.-]+", "_", name)
    name = name.strip("_")
    if not name:
        name = fallback
    if not re.match(r"^[A-Za-z_$.-]", name):
        name = f"func_{name}"
    return name


def _sanitize_local_name(name: str, fallback: str) -> str:
    name = name or fallback
    name = re.sub(r"[^A-Za-z0-9_$.-]+", "_", name)
    name = name.strip("_")
    if not name:
        name = fallback
    if not re.match(r"^[A-Za-z_$.-]", name):
        name = f"v_{name}"
    return name


def _sanitize_comment(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", str(text)).replace("\t", " ")


def _quote_llvm_string(text: str) -> str:
    return text.replace("\\", "\\5C").replace('"', "\\22")


def _host_triple() -> str:
    try:
        import llvmlite.binding as llvm

        return llvm.get_default_triple()
    except Exception:
        machine = os.uname().machine if hasattr(os, "uname") else "unknown"
        if machine in ("x86_64", "amd64"):
            return "x86_64-unknown-linux-gnu"
        if machine in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu"
        return f"{machine}-unknown-linux-gnu"


def _triple_from_ghidra_language(language: str, fallback: str) -> str:
    lang = (language or "").lower()
    if "x86:le:64" in lang:
        return "x86_64-unknown-linux-gnu"
    if "x86:le:32" in lang:
        return "i386-unknown-linux-gnu"
    if "aarch64:le:64" in lang or "arm:le:64" in lang:
        return "aarch64-unknown-linux-gnu"
    if "arm:le:32" in lang:
        return "arm-unknown-linux-gnueabi"
    if "mips:le:64" in lang:
        return "mips64el-unknown-linux-gnu"
    if "mips:be:64" in lang:
        return "mips64-unknown-linux-gnu"
    if "mips:le:32" in lang:
        return "mipsel-unknown-linux-gnu"
    if "mips:be:32" in lang:
        return "mips-unknown-linux-gnu"
    return fallback


def _dtype_width(dtype: Optional[Dict[str, Any]], pointer_bits: int) -> int:
    if not dtype:
        return pointer_bits
    length = _parse_int(dtype.get("length"), 0)
    kind = dtype.get("kind", "")
    if kind == "pointer" or length < 0:
        return pointer_bits
    if length <= 0:
        return pointer_bits
    return _normalize_width(length * 8)


def _dtype_llvm_type(
    dtype: Optional[Dict[str, Any]],
    pointer_bits: int,
    representative: Optional[Dict[str, Any]] = None,
) -> str:
    if _dtype_is_void(dtype):
        return "void"
    width = _varnode_width(representative, pointer_bits) if representative is not None else _dtype_width(dtype, pointer_bits)
    kind = dtype.get("kind", "") if dtype else ""
    name = str(dtype.get("name", "") if dtype else "").lower()
    if kind == "float" or name in {"float", "double"}:
        return _float_type_for_width(width)
    return f"i{width}"


def _varnode_llvm_type(varnode: Optional[Dict[str, Any]], pointer_bits: int) -> str:
    if not varnode:
        return f"i{pointer_bits}"
    high = varnode.get("high") or {}
    dtype = high.get("type") if isinstance(high, dict) else None
    if dtype:
        kind = dtype.get("kind", "")
        name = str(dtype.get("name", "")).lower()
        if kind == "float" or name in {"float", "double"}:
            return _float_type_for_width(_varnode_width(varnode, pointer_bits))
    return f"i{_varnode_width(varnode, pointer_bits)}"


def _dtype_is_void(dtype: Optional[Dict[str, Any]]) -> bool:
    if not dtype:
        return False
    return dtype.get("kind") == "void" or str(dtype.get("name", "")).lower() == "void"


class StrictHighPcodeLLVMEmitter:
    BINARY_INT_OPS = {
        "INT_ADD": "add",
        "INT_SUB": "sub",
        "INT_MULT": "mul",
        "INT_AND": "and",
        "INT_OR": "or",
        "INT_XOR": "xor",
        "INT_LEFT": "shl",
        "INT_RIGHT": "lshr",
        "INT_SRIGHT": "ashr",
        "INT_DIV": "udiv",
        "INT_SDIV": "sdiv",
        "INT_REM": "urem",
        "INT_SREM": "srem",
    }

    COMPARISON_OPS = {
        "INT_EQUAL": "eq",
        "INT_NOTEQUAL": "ne",
        "INT_LESS": "ult",
        "INT_LESSEQUAL": "ule",
        "INT_SLESS": "slt",
        "INT_SLESSEQUAL": "sle",
    }

    BOOL_OPS = {
        "BOOL_AND": "and",
        "BOOL_OR": "or",
        "BOOL_XOR": "xor",
    }

    TERMINATORS = {"BRANCH", "CBRANCH", "BRANCHIND", "RETURN"}

    def __init__(
        self,
        function: Dict[str, Any],
        pointer_bits: int,
        name: str,
        declarations: Dict[Tuple[str, str, Tuple[str, ...]], str],
        function_symbols: Dict[int, str],
        function_signatures: Dict[str, Tuple[str, Tuple[str, ...]]],
        internal_references: Dict[Tuple[str, str, Tuple[str, ...]], str],
        strict: bool = True,
    ):
        self.function = function
        self.pointer_bits = pointer_bits
        self.name = name
        self.declarations = declarations
        self.function_symbols = function_symbols
        self.function_signatures = function_signatures
        self.internal_references = internal_references
        self.strict = strict
        self.lines: List[str] = []
        self.values: Dict[str, LLVMValue] = {}
        self.block_exit_values: Dict[int, Dict[str, LLVMValue]] = {}
        self.deferred_phi_inputs: List[Tuple[int, str, int, Optional[Dict[str, Any]], str]] = []
        self.temp_index = 0
        self.block_map = {int(block["index"]): block for block in function.get("blocks", [])}
        self.ret_type = self._return_type()
        self.current_block: Optional[Dict[str, Any]] = None

    def emit(self) -> List[str]:
        blocks = sorted(self.function.get("blocks", []), key=lambda item: int(item["index"]))
        if not blocks:
            raise FunctionLiftError("function has no high-pcode blocks")

        arg_decls = ["ptr %mem"]
        for index, parameter in enumerate(self.function.get("parameters", [])):
            representative = parameter.get("representative")
            arg_ty = _dtype_llvm_type(parameter.get("type"), self.pointer_bits, representative)
            width = _type_width(arg_ty)
            arg_name = _sanitize_local_name(parameter.get("name", ""), f"arg{index}")
            llvm_arg = LLVMValue(width, f"%{arg_name}", ty=None if arg_ty.startswith("i") else arg_ty)
            arg_decls.append(f"{llvm_arg.llvm_type} {llvm_arg.text}")
            if representative is not None:
                self.values[_varnode_key(representative)] = llvm_arg

        self.lines.append(f"define {self.ret_type} @{self.name}({', '.join(arg_decls)}) {{")
        for block in blocks:
            self.emit_block(block)
        self.resolve_deferred_phi_inputs()
        self.lines.append("}")
        self.lines.append("")
        return self.lines

    def _return_type(self) -> str:
        dtype = self.function.get("return_type")
        return _dtype_llvm_type(dtype, self.pointer_bits)

    def emit_block(self, block: Dict[str, Any]) -> None:
        self.current_block = block
        block_index = int(block["index"])
        self.lines.append(f"bb{block_index}:")
        self.comment(
            "high-pcode block start={} in={} out={}".format(
                block.get("start", ""),
                block.get("in", []),
                block.get("out", []),
            )
        )

        ops = list(block.get("ops", []))
        phi_ops = [op for op in ops if op.get("mnemonic") == "MULTIEQUAL"]
        normal_ops = [op for op in ops if op.get("mnemonic") != "MULTIEQUAL"]

        for op in phi_ops:
            self.emit_pcode_op(op)

        terminated = False
        for op in normal_ops:
            self.emit_pcode_op(op)
            if op.get("mnemonic") in self.TERMINATORS:
                terminated = True
                break

        if not terminated:
            self.emit_fallthrough_terminator(block)
        self.block_exit_values[block_index] = dict(self.values)
        self.current_block = None

    def _needs_terminator(self) -> bool:
        for line in reversed(self.lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(";") or stripped.endswith(":"):
                continue
            return not stripped.startswith(LLVM_TERMINATOR_PREFIXES)
        return True

    def emit_fallthrough_terminator(self, block: Dict[str, Any]) -> None:
        outs = [int(item) for item in block.get("out", [])]
        if len(outs) == 1:
            self.lines.append(f"  br label %bb{outs[0]}")
        elif len(outs) == 0:
            self.emit_default_return()
        else:
            raise UnsupportedPcode(
                f"block {block.get('index')} has {len(outs)} successors but no CBRANCH terminator"
            )

    def emit_default_return(self) -> None:
        if self.ret_type == "void":
            self.lines.append("  ret void")
        else:
            self.lines.append(f"  ret {self.ret_type} {_undef_typed(self.ret_type).text}")

    def emit_pcode_op(self, op: Dict[str, Any]) -> None:
        mnemonic = op.get("mnemonic", "")
        output = op.get("output")
        inputs = op.get("inputs", [])
        self.comment(op.get("text", mnemonic))

        if mnemonic in ("COPY", "CAST"):
            self.emit_copy(output, inputs)
        elif mnemonic in self.BINARY_INT_OPS:
            self.emit_binary(mnemonic, output, inputs)
        elif mnemonic in self.COMPARISON_OPS:
            self.emit_comparison(mnemonic, output, inputs)
        elif mnemonic in self.BOOL_OPS:
            self.emit_bool_binary(mnemonic, output, inputs)
        elif mnemonic in ("BOOL_NEGATE", "BOOL_NEG"):
            self.emit_bool_negate(output, inputs)
        elif mnemonic in ("INT_NEGATE", "INT_NEG"):
            self.emit_int_negate(output, inputs)
        elif mnemonic == "INT_2COMP":
            self.emit_int_twos_complement(output, inputs)
        elif mnemonic in ("INT_ZEXT", "INT_SEXT"):
            self.emit_int_extend(mnemonic, output, inputs)
        elif mnemonic == "POPCOUNT":
            self.emit_count_intrinsic("ctpop", output, inputs, has_zero_flag=False)
        elif mnemonic == "LZCOUNT":
            self.emit_count_intrinsic("ctlz", output, inputs, has_zero_flag=True)
        elif mnemonic == "PIECE":
            self.emit_piece(output, inputs)
        elif mnemonic == "SUBPIECE":
            self.emit_subpiece(output, inputs)
        elif mnemonic == "LOAD":
            self.emit_load(output, inputs)
        elif mnemonic == "STORE":
            self.emit_store(inputs)
        elif mnemonic == "PTRADD":
            self.emit_ptradd(output, inputs)
        elif mnemonic == "PTRSUB":
            self.emit_ptrsub(output, inputs)
        elif mnemonic in ("INT_CARRY", "INT_SCARRY", "INT_SBORROW"):
            self.emit_flag(mnemonic, output, inputs)
        elif mnemonic == "MULTIEQUAL":
            self.emit_multiequal(output, inputs)
        elif mnemonic == "BRANCH":
            self.emit_branch()
        elif mnemonic == "CBRANCH":
            self.emit_cbranch(inputs)
        elif mnemonic == "RETURN":
            self.emit_return(inputs)
        elif mnemonic == "CALL":
            self.emit_call(output, inputs, indirect=False)
        elif mnemonic == "CALLIND":
            self.emit_call(output, inputs, indirect=True)
        elif mnemonic == "CALLOTHER":
            self.emit_callother(output, inputs)
        elif mnemonic == "INDIRECT":
            self.emit_indirect(output, inputs)
        elif mnemonic == "BRANCHIND":
            self.emit_branchind(inputs)
        elif mnemonic in ("FLOAT_EQUAL", "FLOAT_NOTEQUAL", "FLOAT_LESS", "FLOAT_LESSEQUAL"):
            self.emit_float_comparison(mnemonic, output, inputs)
        elif mnemonic == "FLOAT_NAN":
            self.emit_float_nan(output, inputs)
        elif mnemonic in ("FLOAT_ADD", "FLOAT_SUB", "FLOAT_MULT", "FLOAT_DIV"):
            self.emit_float_binary(mnemonic, output, inputs)
        elif mnemonic == "FLOAT_NEG":
            self.emit_float_neg(output, inputs)
        elif mnemonic in ("FLOAT_ABS", "FLOAT_SQRT", "FLOAT_CEIL", "FLOAT_FLOOR", "FLOAT_ROUND"):
            self.emit_float_intrinsic(mnemonic, output, inputs)
        elif mnemonic in ("FLOAT_INT2FLOAT", "INT2FLOAT"):
            self.emit_int_to_float(output, inputs)
        elif mnemonic in ("FLOAT_FLOAT2FLOAT", "FLOAT2FLOAT"):
            self.emit_float_to_float(output, inputs)
        elif mnemonic in ("FLOAT_TRUNC", "TRUNC"):
            self.emit_trunc(mnemonic, output, inputs)
        else:
            raise UnsupportedPcode(f"unsupported high-pcode op: {mnemonic}")

    def comment(self, text: str) -> None:
        self.lines.append(f"  ; {_sanitize_comment(text)}")

    def new_temp(self, width: int, expression: str) -> LLVMValue:
        width = _normalize_width(width)
        name = f"%p{self.temp_index}"
        self.temp_index += 1
        self.lines.append(f"  {name} = {expression}")
        return LLVMValue(width, name)

    def new_typed_temp(self, ty: str, expression: str) -> LLVMValue:
        name = f"%p{self.temp_index}"
        self.temp_index += 1
        self.lines.append(f"  {name} = {expression}")
        return LLVMValue(_type_width(ty), name, ty=None if ty.startswith("i") else ty)

    def new_raw_temp(self, expression: str) -> str:
        name = f"%p{self.temp_index}"
        self.temp_index += 1
        self.lines.append(f"  {name} = {expression}")
        return name

    def cast(self, value: LLVMValue, width: int, signed: bool = False) -> LLVMValue:
        width = _normalize_width(width)
        if value.is_integer and value.width == width:
            return value
        if value.text == "undef":
            return _undef(width)
        if not value.is_integer:
            int_bits = self.new_temp(value.width, f"bitcast {value.llvm_type} {value.text} to i{value.width}")
            return self.cast(int_bits, width, signed=signed)
        if value.const is not None:
            return _const(width, value.const)
        if width == 1:
            return self.new_temp(1, f"icmp ne i{value.width} {value.text}, 0")
        if value.width == 1:
            return self.new_temp(width, f"zext i1 {value.text} to i{width}")
        if value.width < width:
            op = "sext" if signed else "zext"
            return self.new_temp(width, f"{op} i{value.width} {value.text} to i{width}")
        return self.new_temp(width, f"trunc i{value.width} {value.text} to i{width}")

    def cast_to_type(self, value: LLVMValue, ty: str, signed: bool = False) -> LLVMValue:
        if ty == "void":
            raise UnsupportedPcode("cannot cast value to void")
        if ty.startswith("i"):
            return self.cast(value, _type_width(ty), signed=signed)
        if ty not in ("float", "double"):
            raise UnsupportedPcode(f"unsupported cast target type: {ty}")
        if value.text == "undef":
            return _undef_typed(ty)
        if value.llvm_type == ty:
            return value
        if value.llvm_type in ("float", "double"):
            op = "fpext" if _type_width(value.llvm_type) < _type_width(ty) else "fptrunc"
            return self.new_typed_temp(ty, f"{op} {value.llvm_type} {value.text} to {ty}")
        bits = self.cast(value, _type_width(ty))
        return self.new_typed_temp(ty, f"bitcast i{bits.width} {bits.text} to {ty}")

    def read_float(
        self,
        varnode: Optional[Dict[str, Any]],
        width: Optional[int] = None,
    ) -> LLVMValue:
        float_width = _normalize_width(width or _varnode_width(varnode, self.pointer_bits))
        return self.cast_to_type(self.read(varnode, float_width), _float_type_for_width(float_width))

    def read_as_type(self, varnode: Optional[Dict[str, Any]], ty: str) -> LLVMValue:
        if ty in ("float", "double"):
            return self.read_float(varnode, _type_width(ty))
        return self.read(varnode, _type_width(ty))

    def read(
        self,
        varnode: Optional[Dict[str, Any]],
        preferred_width: Optional[int] = None,
        signed: bool = False,
    ) -> LLVMValue:
        width = _normalize_width(preferred_width or _varnode_width(varnode, self.pointer_bits))
        if not varnode:
            return _undef(width)
        if varnode.get("is_constant", False):
            return _const(width, _parse_int(varnode.get("offset"), 0))

        key = _varnode_key(varnode)
        value = self.values.get(key)
        if value is not None:
            if value.is_integer:
                return self.cast(value, width, signed=signed)
            return self.cast(value, width, signed=signed)

        if varnode.get("space") == "ram" or varnode.get("is_address", False):
            address = _const(self.pointer_bits, _parse_int(varnode.get("offset"), 0))
            ptr = self.memory_pointer(address)
            loaded = self.new_temp(width, f"load i{width}, ptr {ptr.text}, align 1")
            self.values[key] = loaded
            return loaded

        return _undef(width)

    def write(self, output: Optional[Dict[str, Any]], value: LLVMValue, signed: bool = False) -> None:
        if not output:
            return
        width = _varnode_width(output, self.pointer_bits)
        if value.is_integer:
            value = self.cast(value, width, signed=signed)
        elif value.width != width:
            value = self.cast(value, width, signed=signed)
        self.values[_varnode_key(output)] = value

    def address_value(self, varnode: Optional[Dict[str, Any]]) -> LLVMValue:
        if not varnode:
            return _undef(self.pointer_bits)
        if varnode.get("is_constant", False) or varnode.get("space") == "ram" or varnode.get("is_address", False):
            return _const(self.pointer_bits, _parse_int(varnode.get("offset"), 0))
        return self.read(varnode, self.pointer_bits)

    def memory_pointer(self, address: LLVMValue) -> LLVMValue:
        address = self.cast(address, self.pointer_bits)
        if address.text == "undef":
            raise UnsupportedPcode("memory access with undefined address")
        return self.new_temp(self.pointer_bits, f"getelementptr i8, ptr %mem, i{self.pointer_bits} {address.text}")

    def emit_copy(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        self.write(output, self.read(inputs[0] if inputs else None, width))

    def emit_binary(self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        op = self.BINARY_INT_OPS[mnemonic]
        signed = op in ("sdiv", "srem", "ashr")
        left = self.read(inputs[0] if len(inputs) > 0 else None, width, signed=signed)
        right = self.read(inputs[1] if len(inputs) > 1 else None, width, signed=signed)
        if op in ("udiv", "sdiv", "urem", "srem") and right.const == 0:
            raise UnsupportedPcode(f"{mnemonic} has constant zero divisor")
        value = self.new_temp(width, f"{op} i{width} {left.text}, {right.text}")
        self.write(output, value, signed=signed)

    def emit_comparison(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        width = max(
            _varnode_width(inputs[0] if len(inputs) > 0 else None, self.pointer_bits),
            _varnode_width(inputs[1] if len(inputs) > 1 else None, self.pointer_bits),
        )
        predicate = self.COMPARISON_OPS[mnemonic]
        signed = predicate.startswith("s")
        left = self.read(inputs[0] if len(inputs) > 0 else None, width, signed=signed)
        right = self.read(inputs[1] if len(inputs) > 1 else None, width, signed=signed)
        value = self.new_temp(1, f"icmp {predicate} i{width} {left.text}, {right.text}")
        self.write(output, value)

    def emit_bool_binary(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        op = self.BOOL_OPS[mnemonic]
        left = self.read(inputs[0] if len(inputs) > 0 else None, 1)
        right = self.read(inputs[1] if len(inputs) > 1 else None, 1)
        value = self.new_temp(1, f"{op} i1 {left.text}, {right.text}")
        self.write(output, value)

    def emit_bool_negate(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        value = self.read(inputs[0] if inputs else None, 1)
        self.write(output, self.new_temp(1, f"xor i1 {value.text}, true"))

    def emit_int_negate(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        value = self.read(inputs[0] if inputs else None, width)
        self.write(output, self.new_temp(width, f"xor i{width} {value.text}, {_mask(width)}"))

    def emit_int_twos_complement(
        self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        width = _varnode_width(output, self.pointer_bits)
        value = self.read(inputs[0] if inputs else None, width)
        self.write(output, self.new_temp(width, f"sub i{width} 0, {value.text}"), signed=True)

    def emit_int_extend(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        width = _varnode_width(output, self.pointer_bits)
        signed = mnemonic == "INT_SEXT"
        value = self.read(inputs[0] if inputs else None)
        self.write(output, self.cast(value, width, signed=signed), signed=signed)

    def emit_count_intrinsic(
        self,
        intrinsic: str,
        output: Optional[Dict[str, Any]],
        inputs: Sequence[Dict[str, Any]],
        has_zero_flag: bool,
    ) -> None:
        input_width = _varnode_width(inputs[0] if inputs else None, self.pointer_bits)
        result_width = _varnode_width(output, self.pointer_bits)
        value = self.read(inputs[0] if inputs else None, input_width)
        ret_ty = f"i{input_width}"
        arg_tys = (ret_ty, "i1") if has_zero_flag else (ret_ty,)
        name = f"llvm.{intrinsic}.i{input_width}"
        self.declarations[(name, ret_ty, arg_tys)] = name
        if has_zero_flag:
            counted = self.new_temp(input_width, f"call {ret_ty} @{name}({ret_ty} {value.text}, i1 false)")
        else:
            counted = self.new_temp(input_width, f"call {ret_ty} @{name}({ret_ty} {value.text})")
        self.write(output, self.cast(counted, result_width))

    def emit_piece(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        high_width = _varnode_width(inputs[0] if len(inputs) > 0 else None, self.pointer_bits)
        low_width = _varnode_width(inputs[1] if len(inputs) > 1 else None, self.pointer_bits)
        high = self.cast(self.read(inputs[0] if len(inputs) > 0 else None, high_width), width)
        low = self.cast(self.read(inputs[1] if len(inputs) > 1 else None, low_width), width)
        shifted = self.new_temp(width, f"shl i{width} {high.text}, {low_width}")
        value = self.new_temp(width, f"or i{width} {shifted.text}, {low.text}")
        self.write(output, value)

    def emit_subpiece(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        source = self.read(inputs[0] if len(inputs) > 0 else None)
        offset = inputs[1] if len(inputs) > 1 else None
        shift_bits = _parse_int(offset.get("offset"), 0) * 8 if offset and offset.get("is_constant", False) else 0
        shifted = source
        if shift_bits:
            shifted = self.new_temp(source.width, f"lshr i{source.width} {source.text}, {shift_bits}")
        self.write(output, self.cast(shifted, width))

    def emit_load(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if not output:
            raise UnsupportedPcode("LOAD without output")
        width = _varnode_width(output, self.pointer_bits)
        address = self.address_value(inputs[-1] if inputs else None)
        ptr = self.memory_pointer(address)
        self.write(output, self.new_temp(width, f"load i{width}, ptr {ptr.text}, align 1"))

    def emit_store(self, inputs: Sequence[Dict[str, Any]]) -> None:
        if len(inputs) < 3:
            raise UnsupportedPcode("STORE expects address-space, address, value")
        value_width = _varnode_width(inputs[2], self.pointer_bits)
        value = self.read(inputs[2], value_width)
        address = self.address_value(inputs[1])
        ptr = self.memory_pointer(address)
        self.lines.append(f"  store i{value_width} {value.text}, ptr {ptr.text}, align 1")

    def emit_ptradd(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        base = self.read(inputs[0] if len(inputs) > 0 else None, width)
        index = self.read(inputs[1] if len(inputs) > 1 else None, width)
        scale = self.read(inputs[2] if len(inputs) > 2 else None, width)
        scaled = self.new_temp(width, f"mul i{width} {index.text}, {scale.text}")
        self.write(output, self.new_temp(width, f"add i{width} {base.text}, {scaled.text}"))

    def emit_ptrsub(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(output, self.pointer_bits)
        base = self.read(inputs[0] if len(inputs) > 0 else None, width)
        offset = self.read(inputs[1] if len(inputs) > 1 else None, width)
        self.write(output, self.new_temp(width, f"add i{width} {base.text}, {offset.text}"))

    def emit_flag(self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = max(
            _varnode_width(inputs[0] if len(inputs) > 0 else None, self.pointer_bits),
            _varnode_width(inputs[1] if len(inputs) > 1 else None, self.pointer_bits),
        )
        left = self.read(inputs[0] if len(inputs) > 0 else None, width)
        right = self.read(inputs[1] if len(inputs) > 1 else None, width)
        if mnemonic == "INT_CARRY":
            result = self.new_temp(width, f"add i{width} {left.text}, {right.text}")
            flag = self.new_temp(1, f"icmp ult i{width} {result.text}, {left.text}")
        elif mnemonic == "INT_SCARRY":
            result = self.new_temp(width, f"add i{width} {left.text}, {right.text}")
            flag = self.signed_overflow_flag(width, left, right, result, subtract=False)
        else:
            result = self.new_temp(width, f"sub i{width} {left.text}, {right.text}")
            flag = self.signed_overflow_flag(width, left, right, result, subtract=True)
        self.write(output, flag)

    def signed_overflow_flag(
        self, width: int, left: LLVMValue, right: LLVMValue, result: LLVMValue, subtract: bool
    ) -> LLVMValue:
        sign_bit = 1 << (width - 1)
        left_sign = self.new_temp(1, f"icmp ne i{width} {self.new_temp(width, f'and i{width} {left.text}, {sign_bit}').text}, 0")
        right_sign = self.new_temp(1, f"icmp ne i{width} {self.new_temp(width, f'and i{width} {right.text}, {sign_bit}').text}, 0")
        result_sign = self.new_temp(1, f"icmp ne i{width} {self.new_temp(width, f'and i{width} {result.text}, {sign_bit}').text}, 0")
        if subtract:
            sign_relation = self.new_temp(1, f"xor i1 {left_sign.text}, {right_sign.text}")
        else:
            sign_relation = self.new_temp(1, f"xor i1 {left_sign.text}, {right_sign.text}")
            sign_relation = self.new_temp(1, f"xor i1 {sign_relation.text}, true")
        result_diff = self.new_temp(1, f"xor i1 {left_sign.text}, {result_sign.text}")
        return self.new_temp(1, f"and i1 {sign_relation.text}, {result_diff.text}")

    def emit_float_binary(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        if output is None:
            raise UnsupportedPcode(f"{mnemonic} without output")
        width = _varnode_width(output, self.pointer_bits)
        ty = _float_type_for_width(width)
        op_map = {
            "FLOAT_ADD": "fadd",
            "FLOAT_SUB": "fsub",
            "FLOAT_MULT": "fmul",
            "FLOAT_DIV": "fdiv",
        }
        left = self.read_float(inputs[0] if len(inputs) > 0 else None, width)
        right = self.read_float(inputs[1] if len(inputs) > 1 else None, width)
        self.write(output, self.new_typed_temp(ty, f"{op_map[mnemonic]} {ty} {left.text}, {right.text}"))

    def emit_float_comparison(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        width = max(
            _varnode_width(inputs[0] if len(inputs) > 0 else None, self.pointer_bits),
            _varnode_width(inputs[1] if len(inputs) > 1 else None, self.pointer_bits),
        )
        ty = _float_type_for_width(width)
        pred_map = {
            "FLOAT_EQUAL": "oeq",
            "FLOAT_NOTEQUAL": "one",
            "FLOAT_LESS": "olt",
            "FLOAT_LESSEQUAL": "ole",
        }
        left = self.read_float(inputs[0] if len(inputs) > 0 else None, width)
        right = self.read_float(inputs[1] if len(inputs) > 1 else None, width)
        self.write(output, self.new_temp(1, f"fcmp {pred_map[mnemonic]} {ty} {left.text}, {right.text}"))

    def emit_float_nan(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        width = _varnode_width(inputs[0] if inputs else output, self.pointer_bits)
        ty = _float_type_for_width(width)
        value = self.read_float(inputs[0] if inputs else None, width)
        self.write(output, self.new_temp(1, f"fcmp uno {ty} {value.text}, {value.text}"))

    def emit_float_neg(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None:
            raise UnsupportedPcode("FLOAT_NEG without output")
        width = _varnode_width(output, self.pointer_bits)
        ty = _float_type_for_width(width)
        value = self.read_float(inputs[0] if inputs else None, width)
        self.write(output, self.new_typed_temp(ty, f"fneg {ty} {value.text}"))

    def emit_float_intrinsic(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        if output is None:
            raise UnsupportedPcode(f"{mnemonic} without output")
        width = _varnode_width(output, self.pointer_bits)
        ty = _float_type_for_width(width)
        intrinsic_map = {
            "FLOAT_ABS": "fabs",
            "FLOAT_SQRT": "sqrt",
            "FLOAT_CEIL": "ceil",
            "FLOAT_FLOOR": "floor",
            "FLOAT_ROUND": "round",
        }
        suffix = "f32" if ty == "float" else "f64"
        name = f"llvm.{intrinsic_map[mnemonic]}.{suffix}"
        self.declarations[(name, ty, (ty,))] = name
        value = self.read_float(inputs[0] if inputs else None, width)
        self.write(output, self.new_typed_temp(ty, f"call {ty} @{name}({ty} {value.text})"))

    def emit_int_to_float(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None:
            raise UnsupportedPcode("FLOAT_INT2FLOAT without output")
        width = _varnode_width(output, self.pointer_bits)
        ty = _float_type_for_width(width)
        source = self.read(inputs[0] if inputs else None)
        self.write(output, self.new_typed_temp(ty, f"sitofp i{source.width} {source.text} to {ty}"))

    def emit_float_to_float(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None:
            raise UnsupportedPcode("FLOAT_FLOAT2FLOAT without output")
        ty = _float_type_for_width(_varnode_width(output, self.pointer_bits))
        value = self.read_float(inputs[0] if inputs else None)
        self.write(output, self.cast_to_type(value, ty))

    def emit_float_trunc(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None:
            raise UnsupportedPcode("FLOAT_TRUNC without output")
        width = _varnode_width(output, self.pointer_bits)
        value = self.read_float(inputs[0] if inputs else None)
        self.write(output, self.new_temp(width, f"fptosi {value.llvm_type} {value.text} to i{width}"), signed=True)

    def emit_trunc(
        self, mnemonic: str, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]
    ) -> None:
        if mnemonic == "FLOAT_TRUNC":
            self.emit_float_trunc(output, inputs)
            return
        if output is None:
            raise UnsupportedPcode("TRUNC without output")
        source_ty = _varnode_llvm_type(inputs[0] if inputs else None, self.pointer_bits)
        if source_ty in ("float", "double"):
            self.emit_float_trunc(output, inputs)
            return
        width = _varnode_width(output, self.pointer_bits)
        self.write(output, self.cast(self.read(inputs[0] if inputs else None), width))

    def emit_multiequal(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None or self.current_block is None:
            raise UnsupportedPcode("MULTIEQUAL without output/current block")
        phi_ty = _varnode_llvm_type(output, self.pointer_bits)
        width = _type_width(phi_ty)
        preds = [int(item) for item in self.current_block.get("in", [])]
        incoming = []
        for index, pred in enumerate(preds):
            varnode = inputs[index] if index < len(inputs) else None
            if varnode is not None and not varnode.get("is_constant", False) and pred not in self.block_exit_values:
                placeholder = f"__PHI_{len(self.deferred_phi_inputs)}__"
                incoming.append(f"[ {placeholder}, %bb{pred} ]")
            else:
                placeholder = ""
                value = self.read_phi_input(pred, varnode, phi_ty)
                incoming.append(f"[ {value.text}, %bb{pred} ]")
            if placeholder:
                self.deferred_phi_inputs.append((len(self.lines), placeholder, pred, varnode, phi_ty))
        if not incoming:
            self.write(output, _undef_typed(phi_ty))
            return
        name = f"%p{self.temp_index}"
        self.temp_index += 1
        self.lines.append(f"  {name} = phi {phi_ty} {', '.join(incoming)}")
        value = LLVMValue(width, name, ty=None if phi_ty.startswith("i") else phi_ty)
        self.write(output, value)

    def resolve_deferred_phi_inputs(self) -> None:
        for line_index, placeholder, pred, varnode, ty in self.deferred_phi_inputs:
            value = self.read_phi_input(pred, varnode, ty)
            self.lines[line_index] = self.lines[line_index].replace(placeholder, value.text)

    def read_phi_input(
        self, predecessor: int, varnode: Optional[Dict[str, Any]], ty: str
    ) -> LLVMValue:
        width = _type_width(ty)
        if not varnode:
            return _undef_typed(ty)
        if varnode.get("is_constant", False):
            if ty in ("float", "double"):
                return _undef_typed(ty)
            return _const(width, _parse_int(varnode.get("offset"), 0))
        value = self.block_exit_values.get(predecessor, {}).get(_varnode_key(varnode))
        if value is None:
            return _undef_typed(ty)
        if ty in ("float", "double"):
            if value.llvm_type == ty:
                return value
            return _undef_typed(ty)
        if value.is_integer and value.width == width:
            return value
        if value.const is not None:
            return _const(width, value.const)
        return _undef(width)

    def emit_branch(self) -> None:
        if self.current_block is None:
            raise UnsupportedPcode("BRANCH outside block")
        outs = [int(item) for item in self.current_block.get("out", [])]
        if len(outs) != 1:
            raise UnsupportedPcode(f"BRANCH with {len(outs)} successors")
        self.lines.append(f"  br label %bb{outs[0]}")

    def emit_cbranch(self, inputs: Sequence[Dict[str, Any]]) -> None:
        if self.current_block is None:
            raise UnsupportedPcode("CBRANCH outside block")
        true_out = self.current_block.get("true_out")
        false_out = self.current_block.get("false_out")
        outs = [int(item) for item in self.current_block.get("out", [])]
        if true_out is None or false_out is None:
            if len(outs) != 2:
                raise UnsupportedPcode("CBRANCH without true/false successors")
            true_out, false_out = outs[0], outs[1]
        condition = self.read(inputs[1] if len(inputs) > 1 else None, 1)
        self.lines.append(f"  br i1 {condition.text}, label %bb{int(true_out)}, label %bb{int(false_out)}")

    def emit_branchind(self, inputs: Sequence[Dict[str, Any]]) -> None:
        if self.current_block is None:
            raise UnsupportedPcode("BRANCHIND outside block")
        outs = [int(item) for item in self.current_block.get("out", [])]
        if not outs:
            raise UnsupportedPcode("BRANCHIND without known successor labels")
        target = self.read(inputs[0] if inputs else None, self.pointer_bits)
        if target.text == "undef":
            raise UnsupportedPcode("BRANCHIND target is undefined")
        target_ptr = self.new_raw_temp(f"inttoptr i{self.pointer_bits} {target.text} to ptr")
        destinations = ", ".join(f"label %bb{item}" for item in outs)
        self.lines.append(f"  indirectbr ptr {target_ptr}, [{destinations}]")

    def emit_return(self, inputs: Sequence[Dict[str, Any]]) -> None:
        if self.ret_type == "void":
            self.lines.append("  ret void")
            return
        value_input = inputs[1] if len(inputs) > 1 else (inputs[0] if inputs else None)
        if self.ret_type in ("float", "double"):
            value = self.read_float(value_input, _type_width(self.ret_type))
        else:
            value = self.read(value_input, _type_width(self.ret_type))
        value = self.cast_to_type(value, self.ret_type)
        self.lines.append(f"  ret {self.ret_type} {value.text}")

    def emit_call(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]], indirect: bool) -> None:
        if not inputs:
            raise UnsupportedPcode("CALL without target")
        arg_tys_for_inputs = [_varnode_llvm_type(item, self.pointer_bits) for item in inputs[1:]]
        args = [self.read_as_type(item, ty) for item, ty in zip(inputs[1:], arg_tys_for_inputs)]
        ret_ty = _varnode_llvm_type(output, self.pointer_bits) if output else "void"
        arg_text = ", ".join(f"{arg.llvm_type} {arg.text}" for arg in args)

        if indirect:
            target = self.read(inputs[0], self.pointer_bits)
            if target.text == "undef":
                raise UnsupportedPcode("CALLIND target is undefined")
            function_ptr = self.new_raw_temp(f"inttoptr i{self.pointer_bits} {target.text} to ptr")
            if output:
                value = self.new_typed_temp(ret_ty, f"call {ret_ty} {function_ptr}({arg_text})")
                self.write(output, value)
            else:
                self.lines.append(f"  call {ret_ty} {function_ptr}({arg_text})")
            return

        target = inputs[0]
        if not (target.get("is_constant", False) or target.get("space") == "ram" or target.get("is_address", False)):
            raise UnsupportedPcode("CALL target is not a constant/address")
        target_offset = _parse_int(target.get("offset"), 0)
        if target_offset in self.function_symbols:
            self.emit_direct_internal_call(target_offset, output, args)
            return

        arg_tys = tuple(arg.llvm_type for arg in args)
        sig = "_".join([ret_ty.replace("*", "p")] + [ty for ty in arg_tys])
        callee = f"pcode_call_{target_offset:x}_{_sanitize_symbol_name(sig, 'sig')}"
        self.declarations[(callee, ret_ty, arg_tys)] = callee
        if output:
            value = self.new_typed_temp(ret_ty, f"call {ret_ty} @{callee}({arg_text})")
            self.write(output, value)
        else:
            self.lines.append(f"  call {ret_ty} @{callee}({arg_text})")

    def emit_direct_internal_call(
        self,
        target_offset: int,
        output: Optional[Dict[str, Any]],
        args: Sequence[LLVMValue],
    ) -> None:
        callee = self.function_symbols[target_offset]
        callee_ret_ty, callee_arg_tys = self.function_signatures[callee]
        call_args = ["ptr %mem"]
        for index, arg_ty in enumerate(callee_arg_tys):
            arg = args[index] if index < len(args) else _undef_typed(arg_ty)
            cast_arg = self.cast_to_type(arg, arg_ty)
            call_args.append(f"{arg_ty} {cast_arg.text}")
        arg_tys = tuple(["ptr"] + list(callee_arg_tys))
        self.internal_references[(callee, callee_ret_ty, arg_tys)] = callee
        arg_text = ", ".join(call_args)
        if callee_ret_ty == "void":
            self.lines.append(f"  call void @{callee}({arg_text})")
            if output is not None:
                self.write(output, _undef(_varnode_width(output, self.pointer_bits)))
            return

        value = self.new_typed_temp(callee_ret_ty, f"call {callee_ret_ty} @{callee}({arg_text})")
        if output is not None:
            self.write(output, value)

    def emit_callother(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        callother_id = _parse_int(inputs[0].get("offset"), 0) if inputs and inputs[0].get("is_constant", False) else 0
        arg_tys_for_inputs = [_varnode_llvm_type(item, self.pointer_bits) for item in inputs[1:]]
        args = [self.read_as_type(item, ty) for item, ty in zip(inputs[1:], arg_tys_for_inputs)]
        ret_ty = _varnode_llvm_type(output, self.pointer_bits) if output else "void"
        arg_tys = tuple(arg.llvm_type for arg in args)
        callee = f"pcode_callother_{callother_id:x}_{ret_ty}_{'_'.join(arg_tys) or 'void'}"
        callee = _sanitize_symbol_name(callee, "pcode_callother")
        self.declarations[(callee, ret_ty, arg_tys)] = callee
        arg_text = ", ".join(f"{arg.llvm_type} {arg.text}" for arg in args)
        if output:
            value = self.new_typed_temp(ret_ty, f"call {ret_ty} @{callee}({arg_text})")
            self.write(output, value)
        else:
            self.lines.append(f"  call {ret_ty} @{callee}({arg_text})")

    def emit_indirect(self, output: Optional[Dict[str, Any]], inputs: Sequence[Dict[str, Any]]) -> None:
        if output is None:
            return
        if len(inputs) >= 2:
            self.write(output, self.read(inputs[1], _varnode_width(output, self.pointer_bits)))
        else:
            self.write(output, _undef(_varnode_width(output, self.pointer_bits)))


def _read_pcode_dump(path: Path) -> List[Dict[str, Any]]:
    functions: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                functions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return functions


def _declaration_lines(declarations: Dict[Tuple[str, str, Tuple[str, ...]], str]) -> List[str]:
    lines: List[str] = []
    for name, ret_ty, arg_tys in sorted(declarations):
        args = ", ".join(arg_tys)
        lines.append(f"declare {ret_ty} @{name}({args})")
    if lines:
        lines.append("")
    return lines


def _function_return_type(function: Dict[str, Any], pointer_bits: int) -> str:
    dtype = function.get("return_type")
    return _dtype_llvm_type(dtype, pointer_bits)


def _function_arg_types(function: Dict[str, Any], pointer_bits: int) -> Tuple[str, ...]:
    arg_types: List[str] = []
    for parameter in function.get("parameters", []):
        representative = parameter.get("representative")
        arg_types.append(_dtype_llvm_type(parameter.get("type"), pointer_bits, representative))
    return tuple(arg_types)


def _assign_function_names(functions: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    used_names: Dict[str, int] = {}
    names: Dict[int, str] = {}
    for function in functions:
        if function.get("decompile_error"):
            continue
        entry = _parse_int(function.get("entry_offset"), 0)
        fallback = f"func_{entry:x}" if entry else "func_unknown"
        base_name = _sanitize_symbol_name(function.get("name", ""), fallback)
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        name = base_name if count == 0 else f"{base_name}_{entry:x}_{count}"
        names[entry] = name
    return names


def _emit_llvm_module(
    functions: List[Dict[str, Any]],
    source_binary: Path,
    target_mode: str = "ghidra",
    strict: bool = True,
    fail_unsupported: bool = False,
) -> Tuple[str, Dict[str, int]]:
    program = functions[0].get("program", {}) if functions else {}
    pointer_size = _parse_int(program.get("pointer_size"), 8)
    pointer_bits = _normalize_width(pointer_size * 8)
    host_triple = _host_triple()
    triple = host_triple if target_mode == "host" else _triple_from_ghidra_language(
        program.get("language", ""), host_triple
    )

    headers = [
        f'; ModuleID = "{_quote_llvm_string(source_binary.name)}"',
        f'source_filename = "{_quote_llvm_string(str(source_binary))}"',
        f'target triple = "{triple}"',
        "",
        "; Generated by Scripts/pcode2llvm.py from Ghidra decompiler High P-Code.",
        "; Functions with unsupported P-Code are skipped in strict mode; no placeholder semantics are emitted.",
        "",
    ]

    declarations: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
    internal_references: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
    body_lines: List[str] = []
    function_symbols = _assign_function_names(functions)
    function_signatures = {
        name: (_function_return_type(function, pointer_bits), _function_arg_types(function, pointer_bits))
        for function in functions
        if not function.get("decompile_error")
        for entry, name in [(_parse_int(function.get("entry_offset"), 0), function_symbols.get(_parse_int(function.get("entry_offset"), 0), ""))]
        if name
    }
    defined_names = set()
    stats = {"records": len(functions), "lifted": 0, "skipped": 0, "decompile_failed": 0}

    for function in functions:
        if function.get("decompile_error"):
            stats["skipped"] += 1
            stats["decompile_failed"] += 1
            LOGGER.warning("Skipping %s: decompile failed: %s", function.get("name"), function.get("decompile_error"))
            continue
        entry = _parse_int(function.get("entry_offset"), 0)
        name = function_symbols[entry]
        try:
            emitter = StrictHighPcodeLLVMEmitter(
                function,
                pointer_bits,
                name,
                declarations,
                function_symbols,
                function_signatures,
                internal_references,
                strict=strict,
            )
            body_lines.extend(emitter.emit())
            defined_names.add(name)
            stats["lifted"] += 1
        except (UnsupportedPcode, FunctionLiftError) as exc:
            stats["skipped"] += 1
            LOGGER.warning("Skipping %s: %s", function.get("name"), exc)
            if fail_unsupported:
                raise

    for key in internal_references:
        name = key[0]
        if name not in defined_names:
            declarations[key] = name

    module_text = "\n".join(headers + _declaration_lines(declarations) + body_lines)
    return module_text, stats


def _verify_llvm_ir(ir_text: str) -> None:
    try:
        import llvmlite.binding as llvm

        module = llvm.parse_assembly(ir_text)
        module.verify()
    except ModuleNotFoundError:
        LOGGER.warning("llvmlite is not installed; skipping LLVM parser verification")


def _remove_stale_output(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _find_analyze_headless(
    analyze_headless: Optional[str] = None,
    ghidra_home: Optional[str] = None,
) -> Path:
    candidates: List[Path] = []
    if analyze_headless:
        candidates.append(Path(analyze_headless).expanduser())
    env_headless = os.environ.get("GHIDRA_ANALYZE_HEADLESS")
    if env_headless:
        candidates.append(Path(env_headless).expanduser())
    for home in (ghidra_home, os.environ.get("GHIDRA_HOME"), os.environ.get("GHIDRA_INSTALL_DIR")):
        if home:
            candidates.append(Path(home).expanduser() / "support" / "analyzeHeadless")
    candidates.append(DEFAULT_GHIDRA_HEADLESS)
    which = shutil.which("analyzeHeadless")
    if which:
        candidates.append(Path(which))

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    searched = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Could not find Ghidra analyzeHeadless. Searched:\n  {searched}")


def _run_ghidra_export(
    input_binary: Path,
    dump_path: Path,
    project_root: Path,
    analyze_headless: Path,
    keep_project: bool,
    no_analysis: bool,
    function_limit: int,
    decompile_timeout: int,
    analysis_timeout: int,
    max_cpu: int,
    verbose: bool,
) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    project_name = "{}_{}".format(
        _sanitize_symbol_name(input_binary.stem, "binary"),
        uuid.uuid4().hex[:8],
    )

    cmd = [
        str(analyze_headless),
        str(project_root),
        project_name,
        "-import",
        str(input_binary),
        "-overwrite",
        "-scriptPath",
        str(GHIDRA_SCRIPT_DIR),
        "-postScript",
        "PcodeDump.java",
        str(dump_path),
        str(max(function_limit, 0)),
        str(max(decompile_timeout, 1)),
    ]
    if no_analysis:
        cmd.append("-noanalysis")
    if analysis_timeout > 0:
        cmd.extend(["-analysisTimeoutPerFile", str(analysis_timeout)])
    if max_cpu > 0:
        cmd.extend(["-max-cpu", str(max_cpu)])
    if not keep_project:
        cmd.append("-deleteProject")

    LOGGER.info("Running Ghidra headless: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if verbose or result.returncode != 0:
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Ghidra headless failed with exit code {result.returncode}")
    if not dump_path.exists():
        raise FileNotFoundError(f"Ghidra did not create expected High P-Code dump: {dump_path}")


def lift_binary_to_llvm(
    input_binary: str,
    output_llvm_ir: str,
    target_mode: str = "host",
    verbose: bool = False,
    ghidra_home: Optional[str] = None,
    analyze_headless: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_project: bool = False,
    no_analysis: bool = False,
    function_limit: int = 0,
    decompile_timeout: int = 60,
    analysis_timeout: int = 300,
    max_cpu: int = 0,
    dump_jsonl: Optional[str] = None,
    verify: bool = True,
    fail_unsupported: bool = False,
    allow_partial: bool = False,
) -> bool:
    """Lift one binary to LLVM IR through Ghidra decompiler High P-Code."""
    if target_mode not in ("host", "ghidra"):
        LOGGER.error("Invalid target_mode %r. Must be 'host' or 'ghidra'", target_mode)
        return False

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    owned_temp: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        input_path = Path(input_binary).expanduser().resolve()
        output_path = Path(output_llvm_ir).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input binary does not exist: {input_path}")

        analyze_path = _find_analyze_headless(analyze_headless, ghidra_home)
        if work_dir:
            project_root = Path(work_dir).expanduser().resolve()
        elif keep_project:
            project_root = Path(tempfile.mkdtemp(prefix="pcode2llvm-ghidra-")).resolve()
        else:
            owned_temp = tempfile.TemporaryDirectory(prefix="pcode2llvm-")
            project_root = Path(owned_temp.name).resolve()

        dump_path = (
            Path(dump_jsonl).expanduser().resolve()
            if dump_jsonl
            else project_root / "high_pcode_dump.jsonl"
        )

        _run_ghidra_export(
            input_binary=input_path,
            dump_path=dump_path,
            project_root=project_root,
            analyze_headless=analyze_path,
            keep_project=keep_project,
            no_analysis=no_analysis,
            function_limit=function_limit,
            decompile_timeout=decompile_timeout,
            analysis_timeout=analysis_timeout,
            max_cpu=max_cpu,
            verbose=verbose,
        )

        functions = _read_pcode_dump(dump_path)
        ir_text, stats = _emit_llvm_module(
            functions,
            input_path,
            target_mode=target_mode,
            strict=True,
            fail_unsupported=fail_unsupported,
        )
        if verify:
            _verify_llvm_ir(ir_text)
        if stats["lifted"] <= 0:
            LOGGER.error("No functions were lifted from %s", input_path)
            _remove_stale_output(output_path)
            return False
        if stats["skipped"] > 0 and not allow_partial:
            LOGGER.error(
                "Lift was incomplete for %s: lifted=%d skipped=%d decompile_failed=%d. "
                "Use --allow-partial only for debugging.",
                input_path,
                stats["lifted"],
                stats["skipped"],
                stats["decompile_failed"],
            )
            _remove_stale_output(output_path)
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ir_text, encoding="utf-8")
        LOGGER.info(
            "Lifted %d/%d functions to %s (skipped=%d, decompile_failed=%d)",
            stats["lifted"],
            stats["records"],
            output_path,
            stats["skipped"],
            stats["decompile_failed"],
        )
        return True
    except Exception:
        LOGGER.error("Failed to lift binary through Ghidra High P-Code", exc_info=verbose)
        if verbose:
            raise
        return False
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


app = typer.Typer(
    add_completion=False,
    help="Ghidra High P-Code to LLVM IR lifter for ReGraph BFSD preprocessing.",
)


@app.command()
def main(
    file: str = typer.Option(..., "-f", "--file", help="Binary file to analyze"),
    output: str = typer.Option(..., "-o", "--output", help="Output LLVM IR file (.ll)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    target: str = typer.Option("host", "--target", help="Target triple source: host or ghidra"),
    ghidra_home: Optional[str] = typer.Option(None, "--ghidra-home", help="Ghidra installation root"),
    analyze_headless: Optional[str] = typer.Option(
        None, "--analyze-headless", help="Path to Ghidra support/analyzeHeadless"
    ),
    work_dir: Optional[str] = typer.Option(None, "--work-dir", help="Ghidra project work directory"),
    keep_project: bool = typer.Option(False, "--keep-project", help="Keep the temporary Ghidra project"),
    no_analysis: bool = typer.Option(False, "--no-analysis", help="Skip Ghidra auto-analysis"),
    function_limit: int = typer.Option(0, "--function-limit", help="Limit exported functions; 0 means all"),
    decompile_timeout: int = typer.Option(
        60, "--decompile-timeout", help="Decompiler timeout per function in seconds"
    ),
    analysis_timeout: int = typer.Option(
        300, "--analysis-timeout", help="Ghidra analysis timeout per file in seconds"
    ),
    max_cpu: int = typer.Option(0, "--max-cpu", help="Limit Ghidra analysis CPU cores; 0 uses Ghidra default"),
    dump_jsonl: Optional[str] = typer.Option(None, "--dump-jsonl", help="Keep/export High P-Code JSONL here"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify generated LLVM IR with llvmlite"),
    fail_unsupported: bool = typer.Option(
        False,
        "--fail-unsupported",
        help="Fail the whole run on the first unsupported function instead of skipping it",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Write output and exit successfully even if some functions are skipped",
    ),
):
    if target not in ("host", "ghidra"):
        raise typer.BadParameter("target must be one of: host, ghidra")

    success = lift_binary_to_llvm(
        input_binary=file,
        output_llvm_ir=output,
        target_mode=target,
        verbose=verbose,
        ghidra_home=ghidra_home,
        analyze_headless=analyze_headless,
        work_dir=work_dir,
        keep_project=keep_project,
        no_analysis=no_analysis,
        function_limit=function_limit,
        decompile_timeout=decompile_timeout,
        analysis_timeout=analysis_timeout,
        max_cpu=max_cpu,
        dump_jsonl=dump_jsonl,
        verify=verify,
        fail_unsupported=fail_unsupported,
        allow_partial=allow_partial,
    )
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
