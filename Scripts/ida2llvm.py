import idapro  # 必须是第一个 import，用于 idalib
import idc
import ida_idaapi
import ida_kernwin
import idautils
import ida_name
import ida_bytes
import ida_ida
import ida_funcs
import ida_typeinf
import ida_segment
import ida_nalt
import ida_hexrays
import ida_auto
import idaapi
import logging
import array
import gc
import re
import hashlib
import os
import struct
import subprocess
import time
from collections import Counter
import numpy as np
import traceback
import sys
import typer
import zlib
import llvmlite.binding as llvm
from llvmlite import ir
from llvmlite.ir import _utils as llvm_ir_utils
from llvmlite.ir.values import FormattedConstant

from contextlib import suppress
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact_state  # noqa: E402

log_dir = Path(__file__).resolve().parents[1] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_dir / 'ida2llvm.log',
    filemode='a',  # 追加模式
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ============================================================================
# 常量定义
# ============================================================================
i8 = ir.IntType(8)
i8ptr = i8.as_pointer()
# 注意：import 时还没有打开数据库，inf_is_64bit() 返回的是 idalib 默认值，
# 真实值由 _init_target_state() 在 open_database 之后设置。
ptrsize = 64
is_big_endian = False
byteorder = "little"
ptext = {}  # 缓存反编译的函数: {地址: 反编译结果}
refreshed_funcs = set()  # 避免重复触发无缓存反编译
lifting_funcs = set()  # 正在定义函数体的函数，防止递归重入
lifted_funcs = set()  # 已经完成函数体定义的函数
function_signature_cache = {}  # 缓存函数签名: {地址: ida_typeinf.tinfo_t}
name_cache = {}
func_cache = {}
segtype_cache = {}
llvm_type_cache = {}
abi_size_cache = {}
type_size_cache = {}
target_data = None
lift_stats = {"defined": 0, "failed": 0, "no_decompile": 0, "approx": 0, "unresolved_data": 0}
# 因无法忠实翻译而被拒绝的函数，按原因计数（也计入 lift_stats["failed"]）
unsupported_reasons = Counter()
# 当前函数用到的近似表示（函数仍会输出，但在 sidecar 状态文件中逐函数记录）
current_approx = set()
# 每个函数的结果：{"ea", "name", "status", "reason"/"approx"}，写入 <output>.lift.json
function_records = []
# 内容无法读取（非 .bss 且未加载）的数据项，保留为 external 声明；前 1000 个写入 sidecar
unresolved_data_items = []
# 每个 binary 独立的 LLVM 类型上下文，避免同名结构体在同一进程处理多个 binary 时串用
llvm_context = ir.Context()
# 全局数据项地址 -> ir.GlobalVariable，用于把指针初始值解析成符号引用
global_by_ea = {}
# microcode 字符串常量的独立编号，不受模块中其他全局对象数量影响
cstr_counter = [0]
# 新建、尚未输出声明的结构体类型（流式输出时只处理新增部分）
pending_type_decls = []
_CACHE_MISS = object()

# 线程局部存储段大小（Windows FS段标准大小）
FS_SEGMENT_SIZE = 0x10000  # 64KB

ENABLE_SIGNATURE_REFRESH = False
ENABLE_DATA_TYPE_GUESS = False
SIZE_MAP = {
    "byte": 8,
    "word": 16,
    "dword": 32,
    "qword": 64
}
FLOAT_RESULT_OPS = {
    ida_hexrays.m_i2f,
    ida_hexrays.m_u2f,
    ida_hexrays.m_f2f,
    ida_hexrays.m_fneg,
    ida_hexrays.m_fadd,
    ida_hexrays.m_fsub,
    ida_hexrays.m_fmul,
    ida_hexrays.m_fdiv,
}
FLOAT_L_OPS = FLOAT_RESULT_OPS | {ida_hexrays.m_f2i, ida_hexrays.m_f2u}
FLOAT_R_OPS = {
    ida_hexrays.m_fadd,
    ida_hexrays.m_fsub,
    ida_hexrays.m_fmul,
    ida_hexrays.m_fdiv,
}
# __PAIR*__ 用于 mop_p（高低两半拼接），保留为不透明声明，宽度与 defs.h 一致
PAIR_HELPER_TYPES = {16: 8, 32: 16, 64: 32, 128: 64}


def _reset_lift_state():
    """清理一次 binary lift 过程中的全局缓存。"""
    global target_data, llvm_context
    ptext.clear()
    refreshed_funcs.clear()
    lifting_funcs.clear()
    lifted_funcs.clear()
    function_signature_cache.clear()
    name_cache.clear()
    func_cache.clear()
    segtype_cache.clear()
    llvm_type_cache.clear()
    abi_size_cache.clear()
    type_size_cache.clear()
    for key in lift_stats:
        lift_stats[key] = 0
    unsupported_reasons.clear()
    current_approx.clear()
    function_records.clear()
    unresolved_data_items.clear()
    _set_sizing_layout(None)
    target_data = None
    llvm_context = ir.Context()
    global_by_ea.clear()
    cstr_counter[0] = 0
    pending_type_decls.clear()


# 源 binary 的目标三元组与 data layout，取自 clang-18 对该 triple 的默认值（tests 中逐项与 clang 核对）。
# 键：(处理器族, 位宽, 字节序)。只支持 ELF；其他格式/处理器没有可靠的 layout，source 模式下直接失败。
SOURCE_TARGETS = {
    ("x86", 64, "little"): ("x86_64-unknown-linux-gnu",
                            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"),
    ("x86", 32, "little"): ("i386-unknown-linux-gnu",
                            "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"),
    ("arm", 32, "little"): ("arm-unknown-linux-gnueabi", "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"),
    ("arm", 64, "little"): ("aarch64-unknown-linux-gnu", "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"),
    ("mips", 32, "big"): ("mips-unknown-linux-gnu", "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"),
    ("mips", 32, "little"): ("mipsel-unknown-linux-gnu", "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"),
    ("mips", 64, "big"): ("mips64-unknown-linux-gnuabi64", "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"),
    ("mips", 64, "little"): ("mips64el-unknown-linux-gnuabi64", "e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"),
    ("ppc", 32, "big"): ("powerpc-unknown-linux-gnu", "E-m:e-p:32:32-Fn32-i64:64-n32"),
    ("ppc", 64, "big"): ("powerpc64-unknown-linux-gnu", "E-m:e-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"),
    ("ppc", 64, "little"): ("powerpc64le-unknown-linux-gnu", "e-m:e-Fn32-i64:64-n32:64-S128-v256:256:256-v512:512:512"),
    ("riscv", 32, "little"): ("riscv32-unknown-linux-gnu", "e-m:e-p:32:32-i64:64-n32-S128"),
    ("riscv", 64, "little"): ("riscv64-unknown-linux-gnu", "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"),
}
# 输出模式：source（默认）= 源目标的 triple/layout，类型大小、成员偏移、指针宽度、端序与源 binary 一致；
# host = 旧行为的兼容模式：宿主 triple/layout，但类型大小按源指针宽度计算，二者不一致，不做内存语义转换
TARGET_MODES = ("source", "host")
MEMORY_CONTRACTS = {"source": "source-layout", "host": "host-compat-unconverted"}
# 计算类型大小所用的 layout；None 表示 host 兼容模式下沿用旧的 "e/E[-p:32:32]" 计算方式
sizing_layout = None
int_alloc_size_cache = {}
# 源处理器族（x86/arm/mips/...），部分 helper 的语义依赖它（如 __CFSUB__ 在 ARM/PPC 上取反）
source_family = None


class UnsupportedTarget(RuntimeError):
    """source 模式下无法确定源目标的 triple/data layout。"""


def _processor_family(proc: str) -> str:
    name = proc.lower()
    if name == "metapc":
        return "x86"
    for family in ("arm", "mips", "ppc", "riscv"):
        if name.startswith(family):
            return family
    if name in ("aarch64", "arm64"):
        return "arm"
    raise UnsupportedTarget(f"processor {proc!r}")


def _layout_pointer_bits(layout: str) -> int:
    for spec in layout.split("-"):
        if spec.startswith("p:"):
            return int(spec.split(":")[1])
    return 64  # LLVM 默认地址空间 0 指针为 64 位


def source_target_spec(proc: str, bits: int, big_endian: bool, is_elf: bool = True) -> dict:
    """源 binary 对应的 LLVM 目标；layout 的指针宽度和端序必须与 IDA 一致，否则失败而不是猜。"""
    if not is_elf:
        raise UnsupportedTarget("only ELF inputs have a source target table")
    endian = "big" if big_endian else "little"
    key = (_processor_family(proc), bits, endian)
    if key not in SOURCE_TARGETS:
        raise UnsupportedTarget(f"no data layout for {key}")
    triple, layout = SOURCE_TARGETS[key]
    if _layout_pointer_bits(layout) != bits or layout.startswith("E") != big_endian:
        raise UnsupportedTarget(f"layout {layout!r} disagrees with source {key}")
    return {"triple": triple, "datalayout": layout, "family": key[0], "bits": bits, "endian": endian}


def _set_sizing_layout(layout):
    """设置类型大小计算所用的 layout，并清空依赖它的缓存。"""
    global sizing_layout, target_data
    sizing_layout = layout
    target_data = None
    int_alloc_size_cache.clear()
    llvm_type_cache.clear()
    abi_size_cache.clear()
    type_size_cache.clear()


def _init_target_state():
    """在 open_database 之后读取当前 binary 的指针宽度和字节序。"""
    global ptrsize, is_big_endian, byteorder, target_data, source_family
    source_family = None
    with suppress(Exception):
        source_family = _processor_family(ida_ida.inf_get_procname())
    ptrsize = 64 if ida_ida.inf_is_64bit() else 32
    is_big_endian = bool(ida_ida.inf_is_be())
    byteorder = "big" if is_big_endian else "little"
    target_data = None
    llvm_type_cache.clear()
    abi_size_cache.clear()
    type_size_cache.clear()


def _get_target_data():
    """复用 llvmlite target data，避免热路径重复构造。source 模式下就是输出模块的 data layout。"""
    global target_data
    if target_data is None:
        if sizing_layout is not None:
            layout = sizing_layout
        else:
            layout = "E" if is_big_endian else "e"
            if ptrsize == 32:
                layout += "-p:32:32"
        target_data = llvm.create_target_data(layout)
    return target_data


def _int_alloc_size(width: int) -> int:
    size = int_alloc_size_cache.get(width)
    if size is None:
        size = ir.IntType(width).get_abi_size(_get_target_data())
        int_alloc_size_cache[width] = size
    return size


def _get_abi_size(typ: ir.Type) -> int:
    key = id(typ)
    entry = abi_size_cache.get(key)
    if entry is None:
        # 同时持有 typ，防止对象被回收后 id 被复用导致取到错误的 size。
        entry = (typ, typ.get_abi_size(_get_target_data()))
        abi_size_cache[key] = entry
    return entry[1]


def _type_size(typ: ir.Type) -> int:
    """
    类型在内存中占用的字节数（结构体都是 packed 的）。source 模式下与输出 data layout 的
    alloc size 完全一致（如 i80 在 x86_64 上占 16 字节），保证成员偏移和数组步长与 LLVM 一致。
    """
    if isinstance(typ, ir.IntType):
        if sizing_layout is None:
            return (typ.width + 7) // 8
        return _int_alloc_size(typ.width)
    if isinstance(typ, ir.PointerType):
        return ptrsize // 8
    if isinstance(typ, ir.FloatType):
        return 4
    if isinstance(typ, ir.DoubleType):
        return 8
    key = id(typ)
    entry = type_size_cache.get(key)
    if entry is None:
        if isinstance(typ, ir.ArrayType):
            size = typ.count * _type_size(typ.element)
        elif isinstance(typ, ir.IdentifiedStructType) and typ.packed and typ.elements is not None:
            size = sum(_type_size(el) for el in typ.elements)
        else:
            size = _get_abi_size(typ)
        entry = (typ, size)
        type_size_cache[key] = entry
    return entry[1]


def _get_name_cached(ea):
    name = name_cache.get(ea)
    if name is None:
        name = ida_name.get_name(ea)
        name_cache[ea] = name
    return name


def _get_func_cached(ea):
    func = func_cache.get(ea, _CACHE_MISS)
    if func is _CACHE_MISS:
        func = ida_funcs.get_func(ea)
        func_cache[ea] = func
    return func


def _get_segtype_cached(ea):
    segtype = segtype_cache.get(ea, _CACHE_MISS)
    if segtype is _CACHE_MISS:
        segtype = ida_segment.segtype(ea)
        segtype_cache[ea] = segtype
    return segtype


def _note_approx(tag: str):
    """标记当前函数用到了近似（非逐位等价）的表示，例如不透明 helper 或未建模的寄存器入口值。"""
    current_approx.add(tag)


def _is_extern_segment(ea) -> bool:
    """segtype() 返回的是枚举值（SEG_DATA=3、SEG_BSS=9 ...），不能按位与 SEG_XTRN(=1)。"""
    return _get_segtype_cached(ea) == ida_segment.SEG_XTRN


def _tif_cache_key(tif: ida_typeinf.tinfo_t, width: int = -1):
    """为 IDA tinfo 生成保守缓存键；失败时禁用该次缓存。"""
    try:
        return (tif.dstr(), tif.get_size(), tif.get_type_name() or "", width)
    except Exception:
        return None


def _function_state_key(func_name: str, ea: int):
    """为函数声明/定义状态生成稳定键值。"""
    if ea is None or ea == ida_idaapi.BADADDR:
        return f"name:{func_name}"
    return f"ea:{ea:#x}"


def _sanitize_symbol_name(name: str) -> str:
    """将 IDA 名称转换为更稳妥的 LLVM 局部名。"""
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    sanitized = sanitized.strip("_")
    return sanitized or "unnamed"


def _get_register_storage(mop: ida_hexrays.mop_t, blk: ir.Block, builder: ir.IRBuilder, is_read: bool = True):
    """
    为 Hex-Rays 的寄存器操作数分配一个稳定的栈槽。

    这里不试图完整模拟物理寄存器别名关系，只提供一个可读可写的 SSA 外存储。
    寄存器的入口值未知：初始化为 freeze undef（未知但固定的值），不能假定为 0。
    第一次访问就是读取时，说明读到了未建模的入口值，记为近似。
    """
    func = blk.parent
    reg_name = f"reg_{_sanitize_symbol_name(mop.dstr())}"
    reg_width = ptrsize if mop.size <= 0 else mop.size * 8

    if reg_name not in func.lvars:
        reg_type = ir.IntType(reg_width)
        with builder.goto_entry_block():
            reg_ptr = builder.alloca(reg_type, name=reg_name)
            builder.store(_freeze_undef(builder, reg_type), reg_ptr)
        func.lvars[reg_name] = reg_ptr
        if is_read:
            _note_approx("register_livein")

    return func.lvars[reg_name]

def lift_tif(tif: ida_typeinf.tinfo_t, width = -1) -> ir.Type:
    cache_key = _tif_cache_key(tif, width)
    if cache_key is not None and cache_key in llvm_type_cache:
        return llvm_type_cache[cache_key]

    res = _lift_tif_uncached(tif, width)
    if cache_key is not None:
        llvm_type_cache[cache_key] = res
    return res


def _lift_tif_uncached(tif: ida_typeinf.tinfo_t, width = -1) -> ir.Type:
    """
    将IDA类型转换为对应的LLVM类型。
    如果IDA类型是数组/结构体/复合类型，则递归执行类型转换。

    :param tif: 要转换的IDA类型
    :type tif: ida_typeinf.tinfo_t
    :raises NotImplementedError: 不支持可变长度结构体
    :return: 转换后的LLVM类型
    :rtype: ir.Type
    """
    if tif.is_func():
        ida_rettype = tif.get_rettype()                           
        ida_args = (tif.get_nth_arg(i) for i in range(tif.get_nargs()))
        is_vararg = tif.is_vararg_cc()                               
        llvm_rettype = lift_tif(ida_rettype)                            
        llvm_args = (lift_tif(arg) for arg in ida_args)
        return ir.FunctionType(llvm_rettype, llvm_args, var_arg = is_vararg)

    elif tif.is_ptr():
        child_tif = tif.get_ptrarr_object()
        if child_tif.is_void():
            return ir.IntType(8).as_pointer()
        return lift_tif(child_tif).as_pointer()

    elif tif.is_array():
        child_tif = tif.get_ptrarr_object()
        element = lift_tif(child_tif)
        count = tif.get_array_nelems()
        if count == 0:
            # 元素数量不确定的数组 = 指针类型
            tif.convert_array_to_ptr()
            return lift_tif(tif)
        element_size = child_tif.get_size()
        if sizing_layout is not None and 0 < element_size != ida_typeinf.BADSIZE:
            # 元素在输出 layout 下的步长必须等于源 binary 的元素大小（如 i386 的 long double 是 12 字节）
            element = _sized_member_type(element, element_size)
        return ir.ArrayType(element, count)

    elif tif.is_void():
        return ir.VoidType()

    elif tif.is_udt():
        return _lift_udt(tif)

    elif tif.is_bool():
        return ir.IntType(1)

    elif tif.is_char():
        return ir.IntType(8)

    elif tif.is_float():
        return ir.FloatType()

    elif tif.is_double():
        return ir.DoubleType()

    # elif tif.is_ldouble():
    #     return ir.DoubleType()  # llvmlite不支持long double

    elif (
        tif.is_decl_int() or tif.is_decl_uint() or tif.is_uint() or tif.is_int()
        or tif.is_decl_int16() or tif.is_decl_uint16() or tif.is_uint16() or tif.is_int16()
        or tif.is_decl_int32() or tif.is_decl_uint32() or tif.is_uint32() or tif.is_int32()
        or tif.is_decl_int64() or tif.is_decl_uint64() or tif.is_uint64() or tif.is_int64()
        or tif.is_decl_int128() or tif.is_decl_uint128() or tif.is_uint128() or tif.is_int128()
        or tif.is_ext_arithmetic() or tif.is_arithmetic()
    ):
        # 整数、枚举、long double 等都按字节宽度映射为整数
        size_bits = tif.get_size() * 8
        if sizing_layout is not None and 16 < tif.get_size() != ida_typeinf.BADSIZE:
            # 超过 128 位的算术类型（如 __m256 的整体视图）保留原始字节数，不收缩成指针宽度
            return ir.ArrayType(i8, size_bits // 8)
        # 防止创建过大的整数类型（限制为128位）
        if size_bits > 128 or size_bits <= 0:
            # logging.warning(f"Invalid type size {tif.get_size()} bytes, using pointer size instead")
            return ir.IntType(ptrsize)
        return ir.IntType(size_bits)
        
    else:
        if width != -1:
            if width <= 0:
                return ir.IntType(ptrsize)
            # 未知类型的数据项按原始字节数保留存储；初始值用紧凑的字节串，大对象也不收缩成一个指针宽度
            return ir.ArrayType(ir.IntType(8), width)
        else:
            return ir.IntType(ptrsize)

def _sized_member_type(llvm_type: ir.Type, size: int) -> ir.Type:
    """成员的 LLVM 类型大小和 IDA 不一致时，用同样字节数的整数/字节数组替代。"""
    if _type_size(llvm_type) == size:
        return llvm_type
    if size in (1, 2, 4, 8, 16) and _type_size(ir.IntType(size * 8)) == size:
        return ir.IntType(size * 8)
    return ir.ArrayType(i8, size)


def _lift_udt(tif: ida_typeinf.tinfo_t) -> ir.Type:
    """
    按 IDA 给出的成员偏移把 struct/union 转为 packed 的 LLVM 结构体，
    保证 LLVM 中的大小和成员偏移与二进制一致。
    """
    is_union = tif.is_union()
    size = tif.get_size()
    if size == ida_typeinf.BADSIZE or size < 0:
        size = 0
    type_name = tif.get_type_name()
    if not type_name:
        kind = "union" if is_union else "struct"
        digest = zlib.crc32(tif.dstr().encode("utf-8", "replace"))
        type_name = f"anon.{kind}.{size}.{digest:08x}"

    context = llvm_context
    if type_name in context.identified_types:
        return context.identified_types[type_name]

    # 先创建命名类型，成员里自引用的指针可以直接拿到它
    struct_t = context.get_identified_type(type_name)
    struct_t.packed = True
    elements = []
    if is_union:
        if size > 0:
            elements.append(ir.ArrayType(i8, size))
    else:
        udt_data = ida_typeinf.udt_type_data_t()
        tif.get_udt_details(udt_data)
        cur = 0
        for idx in range(udt_data.size()):
            member = udt_data.at(idx)
            if member.is_bitfield() or member.offset % 8 or member.size % 8:
                continue
            off = member.offset // 8
            msize = member.size // 8
            if off < cur or msize <= 0 or (size and off + msize > size):
                continue
            member_type = _sized_member_type(lift_tif(member.type), msize)
            if off > cur:
                elements.append(ir.ArrayType(i8, off - cur))
            elements.append(member_type)
            cur = off + msize
        if cur < size:
            elements.append(ir.ArrayType(i8, size - cur))
    struct_t.set_body(*elements)
    pending_type_decls.append(struct_t)
    return struct_t


def _is_fp_type(typ: ir.Type) -> bool:
    return isinstance(typ, (ir.FloatType, ir.DoubleType, ir.HalfType))


def _fp_bits(typ: ir.Type) -> int:
    if isinstance(typ, ir.DoubleType):
        return 64
    if isinstance(typ, ir.HalfType):
        return 16
    return 32


def _resize_int(src: ir.Value, dst_type: ir.IntType, builder: ir.IRBuilder, signed: bool = False) -> ir.Value:
    src_width = src.type.width
    if src_width == dst_type.width:
        return src
    if src_width > dst_type.width:
        return builder.trunc(src, dst_type)
    # i1 是布尔值，扩展时总是 0/1
    if signed and src_width > 1:
        return builder.sext(src, dst_type)
    return builder.zext(src, dst_type)


def typecast(src: ir.Value, dst_type: ir.Type, builder: ir.IRBuilder, signed: bool = False) -> ir.Value:
    """
    将源值转换为目标类型，转换指令插入到 builder 中。

    microcode 是按位语义的：整数与浮点之间的隐式转换是位重解释（bitcast），
    真正的数值转换只出现在 f2i/f2u/i2f/u2f/f2f 指令中，由 lift_insn 显式生成。
    signed 只影响整数扩展时使用 sext 还是 zext。
    """
    src_type = src.type
    if src_type is dst_type:
        return src
    if isinstance(src_type, ir.VoidType):
        return _zero_initializer_for_type(dst_type)
    if src_type == dst_type:
        return src

    src_is_ptr = isinstance(src_type, ir.PointerType)
    dst_is_ptr = isinstance(dst_type, ir.PointerType)
    src_is_int = isinstance(src_type, ir.IntType)
    dst_is_int = isinstance(dst_type, ir.IntType)

    if src_is_ptr and dst_is_ptr:
        return builder.bitcast(src, dst_type)
    if src_is_int and dst_is_int:
        return _resize_int(src, dst_type, builder, signed)
    if src_is_ptr and dst_is_int:
        return builder.ptrtoint(src, dst_type)
    if src_is_int and dst_is_ptr:
        return builder.inttoptr(src, dst_type)

    src_is_fp = _is_fp_type(src_type)
    dst_is_fp = _is_fp_type(dst_type)
    if src_is_fp and dst_is_fp:
        if _fp_bits(src_type) < _fp_bits(dst_type):
            return builder.fpext(src, dst_type)
        return builder.fptrunc(src, dst_type)
    if src_is_int and dst_is_fp:
        bits = _resize_int(src, ir.IntType(_fp_bits(dst_type)), builder, signed)
        return builder.bitcast(bits, dst_type)
    if src_is_fp and dst_is_int:
        bits = builder.bitcast(src, ir.IntType(_fp_bits(src_type)))
        return _resize_int(bits, dst_type, builder, signed)
    if src_is_fp and dst_is_ptr:
        bits = builder.bitcast(src, ir.IntType(_fp_bits(src_type)))
        return builder.inttoptr(bits, dst_type)
    if src_is_ptr and dst_is_fp:
        bits = builder.ptrtoint(src, ir.IntType(_fp_bits(dst_type)))
        return builder.bitcast(bits, dst_type)

    if isinstance(src_type, (ir.IdentifiedStructType, ir.LiteralStructType, ir.ArrayType)) or isinstance(
        dst_type, (ir.IdentifiedStructType, ir.LiteralStructType, ir.ArrayType)
    ):
        # 聚合类型之间、聚合与标量之间：经由内存做位重解释，临时槽取两者中较大的类型
        slot_type = src_type if _type_size(src_type) >= _type_size(dst_type) else dst_type
        with builder.goto_entry_block():
            tmp = builder.alloca(slot_type)
        builder.store(src, builder.bitcast(tmp, src_type.as_pointer()))
        return builder.load(builder.bitcast(tmp, dst_type.as_pointer()))

    return builder.bitcast(src, dst_type)


def _zero_initializer_for_type(typ: ir.Type):
    if isinstance(typ, (ir.FloatType, ir.DoubleType)):
        return ir.Constant(typ, 0.0)
    if isinstance(typ, ir.PointerType):
        return ir.Constant(typ, None)
    if isinstance(typ, ir.IntType):
        return ir.Constant(typ, 0)
    return ir.Constant(typ, None)


def _local_storage_type(typ: ir.Type) -> ir.Type:
    if isinstance(typ, ir.IntType) and typ.width == 1:
        return ir.IntType(32)
    return typ


def storecast(src, dst, builder):
    """
    将目标的类型转换为源的指针类型。
    """
    if src is None or isinstance(src.type, ir.VoidType):
        return dst
    if dst != None and dst.type != src.type.as_pointer():
        dst = typecast(dst, src.type.as_pointer(), builder) 
    return dst

def get_offset_to(builder: ir.IRBuilder, arg: ir.Value, off: int = 0) -> ir.Value:
    """
    根据偏移量对值进行索引。

    :param arg: 要索引的值
    :type arg: ir.Value
    :param off: 索引偏移量，默认为0
    :type off: int, optional
    :return: 按偏移量索引后的值
    :rtype: ir.Value
    """
    def byte_offset_pointer() -> ir.Value:
        byte_ptr = typecast(arg, ir.IntType(8).as_pointer(), builder)
        if off == 0:
            return byte_ptr
        return builder.gep(byte_ptr, (ir.Constant(ir.IntType(ptrsize), off),))

    if isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.ArrayType):
        arr = arg.type.pointee
        size = _type_size(arr.element)
        if size <= 0 or off % size != 0:
            return byte_offset_pointer()
        return builder.gep(arg, (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), off // size),))
    # elif isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.LiteralStructType):
    #     return typecast(arg, ir.IntType(8).as_pointer(), builder)
    elif isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.IdentifiedStructType):
        return byte_offset_pointer()
    elif isinstance(arg.type, ir.PointerType) and off != 0:
        size = _type_size(arg.type.pointee)
        if size <= 0 or off % size != 0:
            return byte_offset_pointer()
        return builder.gep(arg, (ir.Constant(ir.IntType(32), off // size),))
    else:
        return arg


def _block_ends_with_terminator(blk: ir.Block) -> bool:
    term = getattr(blk, "terminator", None)
    return term is not None and len(blk.instructions) > 0 and blk.instructions[-1] is term


def _ensure_fallthrough(builder: ir.IRBuilder, blk: ir.Block, next_blk: ir.Block):
    if next_blk is None or _block_ends_with_terminator(blk):
        return
    if getattr(blk, "terminator", None) is not None:
        blk.terminator = None
    with builder.goto_block(blk):
        builder.branch(next_blk)

def _require_terminated_blocks(func: ir.Function):
    """不在文本层补 ret/br：缺终止指令说明控制流没有被完整提升，拒绝该函数。"""
    for blk in func.blocks:
        if not _block_ends_with_terminator(blk):
            raise UnsupportedMicrocode("unterminated_block", blk.name)


def dedereference(arg: ir.Value) -> ir.Value:
    """
    "反解引用"操作：从已加载的值中获取原始内存地址。
    
    在LLVM中，LoadInstruction会从内存地址加载值（解引用）。
    当我们需要获取原始内存地址时，需要"反解引用"。
    
    需要这个操作的原因：
    - IDA微代码将所有局部变量(LVARS)视为寄存器
    - 而在提升时我们将所有LVARS视为栈变量（符合LLVM SSA形式）

    :param arg: 需要反解引用的值
    :type arg: ir.Value
    :raises NotImplementedError: 参数不是LoadInstr类型
    :return: 原始内存地址
    :rtype: ir.Value
    """
    if isinstance(arg, ir.LoadInstr):
        return arg.operands[0]
    elif isinstance(arg.type, ir.PointerType):
        return arg
    else:
        raise NotImplementedError(f"not implemented: get reference for object {arg} of type {arg.type}")


def _default_function_tinfo() -> ida_typeinf.tinfo_t:
    """void (...)：没有可用类型信息时的函数类型。"""
    ida_func_details = ida_typeinf.func_type_data_t()
    void = ida_typeinf.tinfo_t()
    void.create_simple_type(ida_typeinf.BTF_VOID)
    ida_func_details.rettype = void
    ida_func_details.cc = ida_typeinf.CM_CC_ELLIPSIS | ida_typeinf.CC_CDECL_OK
    function_tinfo = ida_typeinf.tinfo_t()
    function_tinfo.create_func(ida_func_details)
    return function_tinfo


def lift_type_from_address(ea: int, pfunc=None):
    """从地址获取类型信息。"""
    if _get_func_cached(ea) != None and _is_extern_segment(ea):
        # 假设这是一个返回void且接受可变参数的函数
        return _default_function_tinfo()

    if pfunc is not None and getattr(pfunc, "type", None) is not None:
        function_signature_cache[ea] = pfunc.type
        return pfunc.type

    if ea in ptext:
        pfunc_cached = ptext.get(ea)
        if pfunc_cached is not None and getattr(pfunc_cached, "type", None) is not None:
            function_signature_cache[ea] = pfunc_cached.type
            return pfunc_cached.type
        # Stale or invalid cache entry, drop and fall back to IDA type info.
        with suppress(KeyError):
            del ptext[ea]

    cached_tif = function_signature_cache.get(ea)
    if cached_tif is not None:
        return cached_tif
            
    tif = ida_typeinf.tinfo_t()
    has_tinfo = ida_nalt.get_tinfo(tif, ea)
    if not has_tinfo:
        ida_typeinf.guess_tinfo(tif, ea)
    if tif.is_func() or tif.is_funcptr():
        function_signature_cache[ea] = tif
    return tif

def analyze_insn(module, ida_insn, ea):
    """
    Analyzes function call instructions for parameter count mismatches.
    
    Problem: Sometimes IDA's type propagation is incomplete, causing a mismatch
    between the number of arguments in a call and the function's actual signature.
    For example, function A calls function B with 3 arguments, but B's signature
    expects 4 arguments.
    
    Solution: Force re-decompilation of both caller and callee to refresh type
    information and resolve the mismatch.
    
    :param module: LLVM module being constructed
    :param ida_insn: Microcode instruction to analyze
    :param ea: Address of the instruction
    """
    if ida_insn.opcode == ida_hexrays.m_call:
        callnum = len(ida_insn.d.f.args)
        if ida_insn.l.t == ida_hexrays.mop_v: 
            temp_ea = ida_insn.l.g
            func_name = _get_name_cached(temp_ea)
            temp_func = _get_func_cached(temp_ea)
            if ((temp_func is not None)
            and (temp_func.flags & ida_funcs.FUNC_THUNK)): 
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(temp_func)
                if tfunc_ea != ida_idaapi.BADADDR:
                    temp_ea = tfunc_ea
                    func_name = _get_name_cached(temp_ea)

            tif = lift_type_from_address(temp_ea)
            if tif.is_func() or tif.is_funcptr():
                argnum = tif.get_nargs()
                with suppress(KeyError):
                    if func_name == "":
                        func_name = f"data_{hex(ea)[2:]}"
                    if hasattr(module.get_global(func_name
                                                 ), "args"):
                        argnum = len(module.get_global(func_name).args)

                if callnum != argnum:
                    ida_hf = ida_hexrays.hexrays_failure_t()
                    if temp_ea not in refreshed_funcs:
                        refreshed_funcs.add(temp_ea)
                        try:
                            # 重新反编译被调用函数
                            pfunc = ida_hexrays.decompile(temp_ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE)
                            if pfunc != None:
                                ptext[temp_ea] = pfunc
                                function_signature_cache[temp_ea] = pfunc.type
                        except:
                            pass

                    if ea not in refreshed_funcs:
                        refreshed_funcs.add(ea)
                        try:
                            # 重新反编译调用者函数
                            pfunc = ida_hexrays.decompile(ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE) 
                            if pfunc != None:
                                ptext[ea] = pfunc
                                function_signature_cache[ea] = pfunc.type
                        except:
                            return

    if ida_insn.l.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.l.d, ea)
    if ida_insn.r.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.r.d, ea)
    if ida_insn.d.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.d.d, ea)
    return


def _infer_value_type_from_mop(mop):
    if mop is None or mop.t == ida_hexrays.mop_z or mop.size <= 0:
        return None
    if mop.t == ida_hexrays.mop_l:
        with suppress(Exception):
            return lift_tif(mop.l.var().tif)
    if mop.t == ida_hexrays.mop_d:
        insn = mop.d
        if insn.opcode in FLOAT_RESULT_OPS:
            return float_type(mop.size)
        if insn.opcode == ida_hexrays.m_call and insn.l.t == ida_hexrays.mop_v:
            with suppress(Exception):
                tif = lift_type_from_address(insn.l.g)
                if tif.is_funcptr():
                    tif = tif.get_ptrarr_object()
                if tif.is_func():
                    ida_func_details = ida_typeinf.func_type_data_t()
                    tif.get_func_details(ida_func_details)
                    ret_type = lift_tif(ida_func_details.rettype)
                    if not isinstance(ret_type, ir.VoidType):
                        return ret_type
        with suppress(Exception):
            if insn.is_fpinsn():
                return float_type(mop.size)

    size_bits = mop.size * 8
    if 0 < size_bits <= 128:
        return ir.IntType(size_bits)
    return ir.IntType(ptrsize)


def _infer_return_type_from_mba(mba):
    for index in range(mba.qty):
        ida_blk = mba.get_mblock(index)
        ida_insn = ida_blk.head
        while ida_insn is not None:
            if ida_insn.opcode == ida_hexrays.m_ret:
                for mop in (ida_insn.l, ida_insn.r, ida_insn.d):
                    ret_type = _infer_value_type_from_mop(mop)
                    if ret_type is not None and not isinstance(ret_type, ir.VoidType):
                        return ret_type
            ida_insn = ida_insn.next
    return None

def lift_from_address(module: ir.Module, ea: int, typ: ir.Type = None):
    if typ is None:
        tif = lift_type_from_address(ea)
        typ = lift_tif(tif)
    return _lift_from_address(module, ea, typ) 

def _lift_from_address(module: ir.Module, ea: int, typ: ir.Type):
    if isinstance(typ, ir.FunctionType):
        func_name = _get_name_cached(ea)
        if func_name == "":
            func_name = f"data_{hex(ea)[2:]}"
        res = module.get_global(func_name)
        res.lvars = dict() 
        if ea in ptext:
            pfunc = ptext[ea]
        else:
            return res

        if ENABLE_SIGNATURE_REFRESH:
            # refresh function call.
            # some functions caller and signature maybe mismatch. refresh with re-decompile.
            mba = pfunc.mba
            for index in range(mba.qty):
                ida_blk = mba.get_mblock(index)
                ida_insn = ida_blk.head
                while ida_insn is not None:
                    analyze_insn(module, ida_insn, ea)
                    ida_insn = ida_insn.next

            if ea in ptext:
                pfunc = ptext[ea]
            else:
                return res
        
        mba = pfunc.mba
        if isinstance(typ.return_type, ir.VoidType):
            inferred_ret = _infer_return_type_from_mba(mba)
            if inferred_ret is not None:
                typ = ir.FunctionType(inferred_ret, typ.args, var_arg=typ.var_arg)
                if isinstance(res, ir.Function) and res.is_declaration:
                    _retype_function(res, typ)

        for index in range(mba.qty):
            res.append_basic_block(name = f"@{index}")
        for index, blk in enumerate(res.blocks):
            blk.next_blk = res.blocks[index + 1] if index + 1 < len(res.blocks) else None

        # mop_S 的偏移是 mba 栈帧内的字节偏移，范围 [0, fullsize)
        res.stack_size = max(int(getattr(mba, "fullsize", 0) or 0), ptrsize // 8)

        # Map IDA block start addresses to LLVM blocks for jump resolution.
        res.block_by_ea = {}
        for index in range(mba.qty):
            ida_blk = mba.get_mblock(index)
            blk_ea = getattr(ida_blk, "start_ea", None)
            if blk_ea is None:
                blk_ea = getattr(ida_blk, "start", None)
            if blk_ea is not None:
                res.block_by_ea[blk_ea] = res.blocks[index]

        ida_func_details = ida_typeinf.func_type_data_t()
        tif = lift_type_from_address(ea, pfunc) 
        tif.get_func_details(ida_func_details)
        names = [] 

        builder = ir.IRBuilder(res.entry_basic_block)

        with builder.goto_entry_block():
            # declare function results as stack variable
            if not isinstance(typ.return_type, ir.VoidType):
                res.lvars["funcresult"] = builder.alloca(typ.return_type, name = "funcresult")
                builder.store(_zero_initializer_for_type(typ.return_type), res.lvars["funcresult"])

            for lvar in list(pfunc.lvars): 
                if lvar.is_result_var:
                    if lvar.is_arg_var and "funcresult" in res.lvars:
                        names.append("funcresult")
                    continue
                arg_t = _local_storage_type(lift_tif(lvar.tif))
                res.lvars[lvar.name] = builder.alloca(arg_t, name = lvar.name)
                if lvar.is_arg_var:
                    names.append(lvar.name)

            # if function is variadic, declare va_start intrinsic
            if tif.is_vararg_cc() and typ.var_arg:  
                # va_list 布局依赖目标 ABI（x86-64 SysV 24 字节、AArch64 32 字节、
                # i386/ARM32/MIPS 为一个指针），统一分配 32 字节、16 字节对齐以覆盖常见目标
                ptr = builder.alloca(ir.ArrayType(i8, 32), name = "ArgList")
                ptr.align = 16
                res.lvars["ArgList"] = ptr
                va_start = module.declare_intrinsic('llvm.va_start', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
                # va_start 的参数是 va_list 存储本身的地址
                builder.call(va_start, (builder.bitcast(ptr, i8ptr), ))

            # store stack variables
            for arg, arg_n in zip(res.args, names):   
                arg = typecast(arg, res.lvars[arg_n].type.pointee, builder) 
                builder.store(arg, res.lvars[arg_n]) 

        # lift each bblk in cfg
        for index, blk in enumerate(res.blocks):
            next_blk = res.blocks[index + 1] if index + 1 < len(res.blocks) else None
            ida_blk = mba.get_mblock(index)
            ida_insn = ida_blk.head
            while ida_insn is not None:
                lift_insn(ida_insn, blk, builder, next_blk)
                if _block_ends_with_terminator(blk):
                    break
                ida_insn = ida_insn.next

            _ensure_fallthrough(builder, blk, next_blk)

        with builder.goto_block(res.blocks[-1]):
            if not _block_ends_with_terminator(res.blocks[-1]):
                if tif.is_vararg_cc() and typ.var_arg:
                    ptr = res.lvars["ArgList"]
                    va_end = module.declare_intrinsic('llvm.va_end', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
                    builder.call(va_end, (builder.bitcast(ptr, i8ptr), ))
                if isinstance(typ.return_type, ir.VoidType):
                    builder.ret_void()
                else:
                    builder.ret(builder.load(res.lvars["funcresult"]))
        _finalize_nodefault_switches(res, builder)
        _require_terminated_blocks(res)
        stack_slot = res.lvars.get("stack")
        if stack_slot is not None:
            # .cold 等拆分出的函数片段会访问父函数栈帧，fullsize 不一定覆盖全部访问
            needed = max(res.stack_size, getattr(res, "stack_extent", 0))
            if needed != stack_slot.allocated_type.count:
                stack_slot.allocated_type = ir.ArrayType(i8, needed)
                stack_slot.type = stack_slot.allocated_type.as_pointer()
        with suppress(AttributeError):
            del res.block_by_ea
        with suppress(AttributeError):
            del res.lvars
        with suppress(AttributeError):
            del res.stack_size
        with suppress(AttributeError):
            del res.stack_extent
        with suppress(AttributeError):
            del res.nodefault_switches
        for blk in res.blocks:
            with suppress(AttributeError):
                del blk.next_blk
        return res

    data = _read_bytes(ea, _type_size(typ))
    if data is None:
        return None  # 内容未知：调用方生成无初始值（external）的全局声明
    return _constant_from_bytes(module, typ, data, 0)


def _read_bytes(ea: int, size: int):
    """
    一次性读取一段数据。
    - 内容完整可读：返回字节；
    - 位于 .bss（SEG_BSS，加载时清零）：返回全零；
    - 其他不可读/未加载的数据：返回 None，表示内容未知，不能当作已知的 0。
    """
    if size <= 0:
        return b""
    data = ida_bytes.get_bytes(ea, size)
    if data is not None and len(data) == size and ida_bytes.is_loaded(ea) and ida_bytes.is_loaded(ea + size - 1):
        return data
    if _get_segtype_cached(ea) == ida_segment.SEG_BSS:
        return b"\x00" * size
    lift_stats["unresolved_data"] += 1
    if len(unresolved_data_items) < 1000:
        seg = ida_segment.getseg(ea)
        unresolved_data_items.append({"ea": hex(ea), "size": size, "segment": ida_segment.get_segm_name(seg) if seg else None})
    return None


def _resolve_thunk(ea: int) -> int:
    """thunk 函数解析到真实目标，与调用点（lift_mop 的 mop_v）一致。"""
    func = _get_func_cached(ea)
    if func is not None and func.flags & ida_funcs.FUNC_THUNK:
        target, _ = ida_funcs.calc_thunk_func_target(func)
        if target != ida_idaapi.BADADDR:
            return target
    return ea


def _declare_function_at(module: ir.Module, ea: int):
    """为函数起始地址取得（必要时创建）函数声明；名字被非函数全局占用时返回 None。"""
    ea = _resolve_thunk(ea)
    name = _get_name_cached(ea) or f"data_{hex(ea)[2:]}"
    existing = module.globals.get(name)
    if existing is not None:
        return existing if isinstance(existing, ir.Function) else None
    tif = lift_type_from_address(ea)
    if tif.is_funcptr():
        tif = tif.get_ptrarr_object()
    if not tif.is_func():
        tif = _default_function_tinfo()
    with suppress(Exception):
        return lift_function(module, name, True, ea, tif)
    return None


def _pointer_constant(module: ir.Module, typ: ir.PointerType, value: int):
    """
    把指针初始值解析成符号引用：数据项 -> @global（或其内部偏移），函数 -> @func。
    无法解析时保留原始地址（inttoptr），不冒充 null。
    """
    if value == 0:
        return ir.Constant(typ, None)
    target = global_by_ea.get(value)
    if target is None:
        func = _get_func_cached(value)
        if func is not None and func.start_ea == value:
            target = _declare_function_at(module, value)
    if target is not None:
        if isinstance(target, ir.Function):
            # 函数定义时可能按反编译签名改类型；ir 的 bitcast() 在类型相同时直接返回函数本身，
            # 之后改类型会让初始值与全局变量类型不一致，所以强制生成类型固定的 bitcast 常量
            return FormattedConstant(typ, f"bitcast ({target.type} {target.get_reference()} to {typ})")
        return target if target.type == typ else target.bitcast(typ)
    head = ida_bytes.get_item_head(value)
    target = global_by_ea.get(head)
    if target is not None and head != value:
        inner = target.bitcast(i8ptr).gep([ir.Constant(ir.IntType(ptrsize), value - head)])
        return inner if typ == i8ptr else inner.bitcast(typ)
    return ir.Constant(ir.IntType(ptrsize), value).inttoptr(typ)


_ARRAY_CODES = {2: "H", 4: "I", 8: "Q"}


def _int_array_values(data: bytes, off: int, end: int, nbytes: int):
    """按被分析 binary 的字节序把 data[off:end] 解成无符号整数序列。"""
    code = _ARRAY_CODES.get(nbytes)
    if code is not None and array.array(code).itemsize == nbytes:
        values = array.array(code, data[off:end])
        if byteorder != sys.byteorder:
            values.byteswap()
        return values
    return [int.from_bytes(data[pos:pos + nbytes], byteorder) for pos in range(off, end, nbytes)]


def _constant_from_bytes(module: ir.Module, typ: ir.Type, data: bytes, off: int):
    """按被分析 binary 的字节序，从已读出的字节中构造 LLVM 常量。"""
    if isinstance(typ, ir.IntType):
        nbytes = max(1, (typ.width + 7) // 8)
        value = int.from_bytes(data[off:off + nbytes], byteorder)
        if typ.width % 8:
            value &= (1 << typ.width) - 1
        return typ(value)
    if isinstance(typ, (ir.FloatType, ir.DoubleType)):
        fmt = ("f" if isinstance(typ, ir.FloatType) else "d")
        fmt = (">" if is_big_endian else "<") + fmt
        value = struct.unpack_from(fmt, data, off)[0]
        return typ(np.float32(value) if fmt[-1] == "f" else np.float64(value))
    if isinstance(typ, ir.PointerType):
        nbytes = ptrsize // 8
        return _pointer_constant(module, typ, int.from_bytes(data[off:off + nbytes], byteorder))
    if isinstance(typ, ir.ArrayType):
        element = typ.element
        sub_size = _type_size(element)
        end = off + sub_size * typ.count
        if not any(data[off:end]):
            # 全零数组：zeroinitializer，避免为每个元素构造 Python/LLVM 常量对象
            return ir.Constant(typ, None)
        if isinstance(element, ir.IntType) and element.width == 8:
            return ir.Constant(typ, bytearray(data[off:end]))
        if isinstance(element, ir.IntType) and element.width % 8 == 0:
            # 直接生成与 ir.Constant(typ, [...]) 相同的文本，避免为每个元素构造 Python/LLVM 常量对象
            values = _int_array_values(data, off, end, element.width // 8)
            prefix = f"{element} "
            return FormattedConstant(typ, "[" + ", ".join(prefix + str(v) for v in values) + "]")
        return ir.Constant.literal_array([
            _constant_from_bytes(module, element, data, pos)
            for pos in range(off, end, sub_size)
        ])
    if isinstance(typ, (ir.LiteralStructType, ir.IdentifiedStructType)):
        if not any(data[off:off + _type_size(typ)]):
            return ir.Constant(typ, None)
        elements = []
        for el in typ.elements:
            elements.append(_constant_from_bytes(module, el, data, off))
            off += _type_size(el)
        return ir.Constant(typ, elements)
    raise NotImplementedError(f"unsupported constant type {typ}")

def str2size(str_size: str) -> int:
    """
    将表示内存大小的字符串转换为位数。

    :param str_size: 描述大小的字符串
    :type str_size: str
    :return: 字符串的大小，以位为单位
    :rtype: int
    """
    if str_size not in SIZE_MAP:
        raise AssertionError(f"字符串大小必须是 {list(SIZE_MAP.keys())} 之一，得到的是 '{str_size}'")
    return SIZE_MAP[str_size]


# lifter 自己实现了函数体的 helper：__readfs*/__writefs* 访问 virtual_fs（近似的 TLS），__halt 用 hlt
_IMPLEMENTED_HELPER_PREFIXES = ("__readfs", "__writefs")
# 与 libc 同名、语义一致的 helper（Hex-Rays 识别出的内联 memset/strcpy 等），调用它们不算近似
_LIBC_HELPERS = {
    "memset", "memcpy", "memmove", "memcmp", "strcpy", "strncpy", "strcmp", "strncmp",
    "strlen", "strcat", "strncat", "strchr", "strrchr", "abs", "labs", "llabs", "fabs", "fabsf",
}


def _is_implemented_helper(name: str) -> bool:
    return name == "__halt" or name.startswith(_IMPLEMENTED_HELPER_PREFIXES)


def _note_helper(name: str):
    if name.startswith(_IMPLEMENTED_HELPER_PREFIXES):
        _note_approx("virtual_fs")
    elif name not in _LIBC_HELPERS and name != "__halt":
        _note_approx(f"opaque_helper:{name}")


def lift_intrinsic_function(module: ir.Module, func_name: str):
    """
    将IDA宏转换为LLVM内置函数。

    Hexray的反编译器在微代码层面识别高级函数。
    这些 ida_hexrays.mop_t 对象被标记为 ida_hexrays.mop_h（辅助函数成员）
    
    这改善了反编译器输出，表示无法映射到良好C代码的操作。
    （参考: https://hex-rays.com/blog/igors-tip-of-the-week-67-decompiler-helpers/）

    相关的#define宏请参考IDA SDK: `defs.h` 和 `pro.h`。

    LLVM内置函数具有公认的名称和语义，并且必须遵循某些限制。

    :param module: LLVM模块
    :type module: ir.Module
    :param func_name: 函数名称
    :type func_name: str
    :raises NotImplementedError: 不支持的函数
    :return: LLVM内置函数
    :rtype: ir.Function
    """
    # 如果已存在，直接返回
    existing = module.globals.get(func_name)
    if isinstance(existing, ir.Function):
        return existing

    if func_name.startswith("__PAIR") and func_name.endswith("__"):
        bits = int(func_name[len("__PAIR"):-2])
        if bits not in PAIR_HELPER_TYPES:
            raise UnsupportedMicrocode(f"pair_width_{bits}", func_name)
        half = ir.IntType(PAIR_HELPER_TYPES[bits])
        return module.declare_intrinsic(func_name, fnty=ir.FunctionType(ir.IntType(bits), [half, half]))

    if func_name == "__halt":
        fty = ir.FunctionType(ir.VoidType(), [])
        f = ir.Function(module, fty, "__halt")
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        builder.asm(fty, "hlt", "", (), True)
        builder.ret_void()
        return f

    elif func_name.startswith("__readfs"):
        _, size = func_name.split("__readfs")
        size = str2size(size)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), 65536)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.initializer = fs_reg_typ(None)

        fty = ir.FunctionType(ir.IntType(size), [ir.IntType(32),])

        f = ir.Function(module, fty, func_name)
        offset, = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        fs_reg = typecast(fs_reg, ir.IntType(8).as_pointer(), builder)
        pointer = builder.gep(fs_reg, (offset,))
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        res = builder.load(pointer)
        builder.ret(res)

        return f

    elif func_name.startswith("__writefs"):
        _, size = func_name.split("__writefs")
        size = str2size(size)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), FS_SEGMENT_SIZE)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.initializer = fs_reg_typ(None)            

        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32), ir.IntType(size)])

        f = ir.Function(module, fty, func_name)
        offset, value  = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        fs_reg = typecast(fs_reg, ir.IntType(8).as_pointer(), builder)
        pointer = builder.gep(fs_reg, (offset,))
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        builder.store(value, pointer)
        builder.ret_void()

        return f

    else:
        raise NotImplementedError(f"不支持的内置函数: {func_name}")  

# ============================================================================
# 有确定语义的 Hex-Rays helper：实现为宽度正确的 LLVM 运算 / 标准 intrinsic
# 依据 IDA 9.4 include/defs.h（行号见注释）和 MSVC intrinsic 的文档语义；不在此表中的 helper
# 保持为不透明外部调用并打 approx 标记。
# ============================================================================

def _llvm_intrinsic(module: ir.Module, name: str, fnty: ir.FunctionType) -> ir.Function:
    existing = module.globals.get(name)
    if isinstance(existing, ir.Function):
        return existing
    return ir.Function(module, fnty, name)


def _int_arg(value: ir.Value, width: int, builder: ir.IRBuilder) -> ir.Value:
    return typecast(value, ir.IntType(width), builder)


def _rotate(builder, value, count, width, right):
    """defs.h:233 __ROL__：count>0 时左移 count%W；否则右移 (-count)%W。等价于 fshl(x, x, count mod W)
    （W 是 2 的幂，count 的补码低位即模 W 的余数）；count 为 W 的倍数时机器语义为恒等，fshl 亦然。"""
    module = builder.block.parent.parent
    ty = ir.IntType(width)
    x = _int_arg(value, width, builder)
    amount = count if count.type == ty else (
        builder.trunc(count, ty) if count.type.width > width else builder.sext(count, ty))
    if right:  # defs.h:260-263：__RORn__(v, c) == __ROL__(v, -c)
        amount = builder.sub(ty(0), amount)
    fshl = _llvm_intrinsic(module, f"llvm.fshl.i{width}", ir.FunctionType(ty, [ty, ty, ty]))
    return builder.call(fshl, [x, x, amount])


def _pair(builder, high, low, width):
    """defs.h:181-188 __PAIRn__/__SPAIRn__：(high << W/2) | (uintW/2)low，结果位模式与高半是否有符号无关。"""
    half = width // 2
    ty = ir.IntType(width)
    hi = builder.zext(_int_arg(high, half, builder), ty)
    lo = builder.zext(_int_arg(low, half, builder), ty)
    return builder.or_(builder.shl(hi, ty(half)), lo)


def _sign_bit(builder, value, width):
    return builder.icmp_signed("<", value, ir.IntType(width)(0))


def _flag_helper(name, builder, x, y, width):
    """defs.h:278-349 __OFSUB__/__OFADD__/__CFSUB__/__CFADD__（两个操作数同宽时）。结果为 int8 0/1。"""
    x, y = _int_arg(x, width, builder), _int_arg(y, width, builder)
    if name == "__CFADD__":
        flag = builder.icmp_unsigned(">", x, builder.add(x, y))
    elif name == "__CFSUB__":
        flag = builder.icmp_unsigned("<", x, y)
        if source_family in ("arm", "ppc"):  # defs.h:312 SUB_WITH_CARRY
            flag = builder.not_(flag)
    else:
        result = builder.sub(x, y) if name == "__OFSUB__" else builder.add(x, y)
        sx, sy, sr = (_sign_bit(builder, v, width) for v in (x, y, result))
        # OFSUB: (sx ^ sy) & (sx ^ sr)；OFADD: (!sx ^ sy) & (sx ^ sr)
        lhs = builder.xor(sx, sy) if name == "__OFSUB__" else builder.xor(builder.not_(sx), sy)
        flag = builder.and_(lhs, builder.xor(sx, sr))
    return builder.zext(flag, ir.IntType(8))


_HELPER_WIDTH_SUFFIX = {"1": 8, "2": 16, "4": 32, "8": 64}
_INTERLOCKED_RMW = {"Exchange": "xchg", "ExchangeAdd": "add", "Or": "or", "And": "and", "Xor": "xor"}


def _interlocked_width(suffix: str):
    return {"": 32, "8": 8, "16": 16, "64": 64}.get(suffix)


def _lift_semantic_helper(name: str, ret_type: ir.Type, args: list, builder: ir.IRBuilder):
    """
    精确实现 name 对应的 helper：返回结果值；没有返回值的 helper（store/fence）返回 True；
    不认识或无法确定语义的 helper 返回 None（由调用方声明为不透明外部函数）。
    宽度以调用信息为准并与名字中的宽度交叉校验，不一致时拒绝。
    """
    ret_width = ret_type.width if isinstance(ret_type, ir.IntType) else None

    m = re.fullmatch(r"__RO([LR])([1248])__", name)
    if m and len(args) == 2:
        width = _HELPER_WIDTH_SUFFIX[m.group(2)]
        if ret_width != width:
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name} returns {ret_type}")
        count = args[1] if isinstance(args[1].type, ir.IntType) else _as_int(args[1], 4, builder)
        return _rotate(builder, args[0], count, width, right=m.group(1) == "R")

    m = re.fullmatch(r"__S?PAIR(16|32|64|128)__", name)
    if m and len(args) == 2:
        width = int(m.group(1))
        if ret_width != width:
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name} returns {ret_type}")
        return _pair(builder, args[0], args[1], width)

    bswap = {"_byteswap_ushort": 16, "_byteswap_ulong": 32, "_byteswap_uint64": 64}
    if name in bswap and len(args) == 1:
        width = bswap[name]
        if ret_width != width:
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name} returns {ret_type}")
        ty = ir.IntType(width)
        fn = _llvm_intrinsic(builder.block.parent.parent, f"llvm.bswap.i{width}", ir.FunctionType(ty, [ty]))
        return builder.call(fn, [_int_arg(args[0], width, builder)])

    if name == "__clz" and len(args) == 1 and ret_width:
        # ARM CLZ：输入为 0 时结果为位宽（ctlz 的 is_zero_poison = false）
        width = args[0].type.width if isinstance(args[0].type, ir.IntType) else ret_width
        ty = ir.IntType(width)
        fn = _llvm_intrinsic(builder.block.parent.parent, f"llvm.ctlz.i{width}",
                             ir.FunctionType(ty, [ty, ir.IntType(1)]))
        return typecast(builder.call(fn, [_int_arg(args[0], width, builder), ir.IntType(1)(0)]), ret_type, builder)

    if name in ("__CFADD__", "__CFSUB__", "__OFADD__", "__OFSUB__") and len(args) == 2:
        widths = {a.type.width for a in args if isinstance(a.type, ir.IntType)}
        if len(widths) == 1 and len(args) == 2 and (name != "__CFSUB__" or source_family is not None):
            return typecast(_flag_helper(name, builder, args[0], args[1], widths.pop()), ret_type, builder)
        return None  # 混合宽度依赖 C 的整型提升与符号性，调用信息不足以确定：保持不透明

    if name in ("_mm_load_si128", "_mm_loadu_si128") and len(args) == 1:
        if _type_size(ret_type) != 16:
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name} returns {ret_type}")
        ptr = typecast(args[0], ret_type.as_pointer(), builder)
        return builder.load(ptr, align=16 if name == "_mm_load_si128" else 1)

    if name in ("_mm_store_si128", "_mm_storeu_si128") and len(args) == 2:
        if _type_size(args[1].type) != 16:
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name} stores {args[1].type}")
        ptr = typecast(args[0], args[1].type.as_pointer(), builder)
        builder.store(args[1], ptr, align=16 if name == "_mm_store_si128" else 1)
        return True

    if name == "_mm_mfence" and not args:
        builder.fence("seq_cst")
        return True

    if name == "alloca" and len(args) == 1:
        size = args[0] if isinstance(args[0].type, ir.IntType) else _as_int(args[0], ptrsize // 8, builder)
        return typecast(builder.alloca(i8, size=size), ret_type, builder)

    m = re.fullmatch(r"_Interlocked(CompareExchange|Exchange|ExchangeAdd|Or|And|Xor|Increment|Decrement)(8|16|64)?", name)
    if m:
        op, width = m.group(1), _interlocked_width(m.group(2) or "")
        expected_args = {"CompareExchange": 3, "Increment": 1, "Decrement": 1}.get(op, 2)
        if width is None or len(args) != expected_args or ret_width not in (None, width):
            raise UnsupportedMicrocode("helper_width_mismatch", f"{name}({len(args)} args) returns {ret_type}")
        ty = ir.IntType(width)
        ptr = typecast(args[0], ty.as_pointer(), builder)
        if op == "CompareExchange":  # (Destination, Exchange, Comparand) -> 初值
            pair = builder.cmpxchg(ptr, _int_arg(args[2], width, builder), _int_arg(args[1], width, builder),
                                   "seq_cst", "seq_cst")
            result = builder.extract_value(pair, 0)
        elif op in ("Increment", "Decrement"):  # 返回新值
            rmw = "add" if op == "Increment" else "sub"
            old = builder.atomic_rmw(rmw, ptr, ty(1), "seq_cst")
            result = builder.add(old, ty(1)) if op == "Increment" else builder.sub(old, ty(1))
        else:  # 返回旧值
            result = builder.atomic_rmw(_INTERLOCKED_RMW[op], ptr, _int_arg(args[1], width, builder), "seq_cst")
        return typecast(result, ret_type, builder) if ret_width else result

    return None


def _retype_function(func: ir.Function, ftype: ir.FunctionType):
    """把一个尚无函数体的 ir.Function 改成新的函数类型。"""
    func.ftype = ftype
    func.type = ftype.as_pointer()
    # 声明还没有函数体，重置命名空间，形参和后续 SSA 编号与直接定义时一致
    func.scope = llvm_ir_utils.NameScope()
    func.args = tuple(ir.Argument(func, t) for t in ftype.args)
    func.return_value = ir.ReturnValue(func, ftype.return_type)


def lift_function(module: ir.Module, func_name: str, is_declare: bool, ea = None, tif: ida_typeinf.tinfo_t = None):
    """
    根据函数名声明函数。
    如果 `is_declare` 为False，还会通过递归提升其在IDA反编译输出中的指令来定义函数。
    如果给出了 `tif`，则强制使用给定的函数类型。
    主要工作在 `lift_from_address` 中完成。

    :param module: 函数的父模块
    :type module: ir.Module
    :param func_name: 要提升的函数名
    :type func_name: str
    :param is_declare: 是否只是声明函数，不定义
    :type is_declare: bool
    :param tif: 函数类型，默认为None
    :type tif: ida_typeinf.tinfo_t, optional
    :return: 提升后的函数
    :rtype: ir.Function
    """
    if func_name == "":
        func_name = f"data_{hex(ea)[2:]}"
    if _is_implemented_helper(func_name):
        # 只有 lifter 自己实现的 helper 走内置实现；与其他 helper 同名的真实函数（如静态链接的
        # memcpy）照常提升，不被硬编码声明顶替
        return lift_intrinsic_function(module, func_name)

    func_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, func_name)
    if func_ea != ida_idaapi.BADADDR and _is_extern_segment(func_ea):
        is_declare = True 

    if func_ea == ida_idaapi.BADADDR:
        func_ea = ea

    assert func_ea != ida_idaapi.BADADDR
    if tif is None:
        tif = lift_type_from_address(func_ea) 
    elif tif.is_func() or tif.is_funcptr():
        function_signature_cache[func_ea] = tif

    typ = lift_tif(tif) 
    state_key = _function_state_key(func_name, func_ea)

    existing = None
    with suppress(KeyError):
        existing = module.get_global(func_name)

    if existing is not None:
        if not isinstance(existing, ir.Function):
            return existing
        if (
            is_declare
            or not existing.is_declaration
            or state_key in lifting_funcs
            or state_key in lifted_funcs
        ):
            return existing
        res = existing
        if isinstance(typ, ir.FunctionType) and existing.ftype != typ:
            # 先前在调用点按 IDA 存储的签名生成了声明；定义时以反编译出的签名为准，
            # 否则形参与 lvar 对不上、带返回值的函数会缺少 funcresult。
            # 之前 lift 的函数都已流式写出，改签名不会影响尚在内存中的调用指令。
            _retype_function(existing, typ)
        if isinstance(existing.type, ir.PointerType) and isinstance(existing.type.pointee, ir.FunctionType):
            typ = existing.type.pointee
    else:
        if isinstance(typ, ir.PointerType):
            logging.debug(f"Unexpected pointer type for function {func_name}: {typ}")
        res = ir.Function(module, typ, func_name)
        res._source_ea = func_ea

    if is_declare:
        return res

    lifting_funcs.add(state_key)
    try:
        res = lift_from_address(module, func_ea, typ)
        lifted_funcs.add(state_key)
        return res
    finally:
        lifting_funcs.discard(state_key)

def calc_instsize(typ):
    """
    计算指令宽度（以位为单位）。
    """
    if isinstance(typ, ir.PointerType):
        return ptrsize
    elif isinstance(typ, ir.ArrayType):
        return -1
    elif isinstance(typ, ir.IdentifiedStructType):
        return -1
    elif isinstance(typ, ir.FloatType):
        return 32
    elif isinstance(typ, ir.DoubleType):
        return 64
    else:
        return typ.width

def _cast_slot(slot: ir.Value, mop_size: int, knowntyp, builder: ir.IRBuilder) -> ir.Value:
    """把存储槽指针转换成 knowntyp* 或 mop 宽度的整数指针。"""
    if mop_size == -1:
        return slot
    if knowntyp is not None:
        return typecast(slot, knowntyp.as_pointer(), builder)
    if calc_instsize(slot.type.pointee) != mop_size * 8:
        return typecast(slot, ir.IntType(mop_size * 8).as_pointer(), builder)
    return slot


def lift_mop(mop: ida_hexrays.mop_t, blk: ir.Block, builder: ir.IRBuilder, dest = False, knowntyp = None) -> ir.Value:
    """
    Lifts a microcode operand (mop) to an LLVM value.

    Microcode operands come in many types:
    - mop_n: Immediate values (constants)
    - mop_r: Register values
    - mop_l: Local variables (stack-allocated in LLVM)
    - mop_v: Global variables/functions
    - mop_S: Stack variables
    - mop_d: Nested instructions (recursive lifting)
    - mop_f: Function call information
    - mop_a: Addresses of other operands
    - mop_h: Helper/intrinsic functions
    - mop_str: String constants
    - mop_c: Switch cases
    - mop_fn: Floating point constants
    - mop_p: Pair operations

    :param mop: The microcode operand to lift
    :param blk: Current LLVM basic block
    :param builder: LLVM IR builder
    :param dest: Whether this operand is a destination (true = return pointer, false = load value)
    :param knowntyp: Optional known value type of the operand (never a pointer-to-storage type)
    :return: Corresponding LLVM value
    """
    builder.position_at_end(blk)
    t = mop.t
    if knowntyp is not None and isinstance(knowntyp, ir.VoidType):
        knowntyp = None
    if t == ida_hexrays.mop_l: # 局部变量
        lvar_ref = mop.l
        lvar = lvar_ref.var()
        name = "funcresult" if lvar.is_result_var else lvar.name
        llvm_arg = blk.parent.lvars[name]
        llvm_arg = get_offset_to(builder, llvm_arg, lvar_ref.off)
        llvm_arg = _cast_slot(llvm_arg, mop.size, knowntyp, builder)
        return llvm_arg if dest else builder.load(llvm_arg)
    elif t == ida_hexrays.mop_n: # 立即数
        return ir.Constant(ir.IntType(mop.size * 8), mop.nnn.value)
    elif t == ida_hexrays.mop_d: # 另一个指令
        is_callarg = isinstance(mop, ida_hexrays.mcallarg_t)
        value_type = lift_tif(mop.type) if is_callarg else knowntyp
        if value_type is None and mop.size > 0:
            sub = mop.d
            if sub.opcode in (ida_hexrays.m_call, ida_hexrays.m_icall) and sub.d.t == ida_hexrays.mop_f:
                # 调用结果被使用：用 microcode 调用信息里的返回类型，
                # 避免被调函数声明为 void（如未带类型的导入函数）时丢掉返回值
                value_type = lift_tif(sub.d.f.return_type)
                if isinstance(value_type, ir.VoidType):
                    value_type = ir.IntType(mop.size * 8)
        d = lift_insn(mop.d, blk, builder, result_type=value_type)
        if d is None:
            return None
        size = mop.size
        if isinstance(d.type, ir.VoidType):
            if knowntyp is not None or is_callarg or size > 0:
                # 返回 void 的调用结果被当作值使用：不能用 0 顶替
                raise UnsupportedMicrocode("void_call_result_used", mop.dstr())
        elif size == -1:
            pass
        elif is_callarg:
            d = typecast(d, value_type, builder, signed=mop.type.is_signed())
        elif knowntyp is not None:
            d = typecast(d, knowntyp, builder)
        elif calc_instsize(d.type) != size * 8:
            d = typecast(d, ir.IntType(size * 8), builder)
        return d
    elif t == ida_hexrays.mop_r: # 寄存器值
        reg_ptr = _get_register_storage(mop, blk, builder, is_read=not dest)
        reg_ptr = _cast_slot(reg_ptr, mop.size if mop.size > 0 else -1, knowntyp, builder)
        return reg_ptr if dest else builder.load(reg_ptr)
    elif t == ida_hexrays.mop_S: # 栈变量
        name = "stack"
        func = blk.parent
        if name not in func.lvars:
            stack_size = getattr(func, "stack_size", ptrsize // 8)
            with builder.goto_entry_block():
                slot = builder.alloca(ir.ArrayType(i8, stack_size), name = name)
                func.lvars[name] = slot
                func.lvars["stack.i8"] = builder.bitcast(slot, i8ptr)
        # 按字节偏移访问，栈槽大小在函数提升完后按实际访问范围再定
        off = mop.s.off
        base = func.lvars["stack.i8"]
        llvm_arg = base if off == 0 else builder.gep(base, (ir.Constant(ir.IntType(ptrsize), off),))
        width = mop.size if mop.size > 0 else ptrsize // 8
        func.stack_extent = max(getattr(func, "stack_extent", 0), off + width)
        llvm_arg = _cast_slot(llvm_arg, mop.size, knowntyp, builder)
        return llvm_arg if dest else builder.load(llvm_arg)
    elif t == ida_hexrays.mop_b: # 基本块编号（用于jmp/call指令）
        return blk.parent.blocks[mop.b]
    elif t == ida_hexrays.mop_v: # 全局变量
        module = blk.parent.parent
        ea = mop.g
        name = _get_name_cached(ea)
        if name == "":
            name = f"data_{hex(ea)[2:]}"

        g = module.globals.get(name)
        if isinstance(g, ir.Function):
            return g
        if g is None:
            tif = lift_type_from_address(ea)
            func = _get_func_cached(ea)
            is_func_start = func is not None and func.start_ea == ea
            is_xtrn = _is_extern_segment(ea)
            # 函数指针变量（非导入）按数据处理：作为调用目标时先加载其值再间接调用
            if tif.is_func() or is_func_start or (tif.is_funcptr() and is_xtrn):
                if tif.is_funcptr():
                    tif = tif.get_ptrarr_object()
                if not tif.is_func():
                    # 函数起始地址但 IDA 没有给出函数类型（如作为回调被取地址），
                    # 仍按函数处理，避免生成同名数据全局变量而丢掉函数定义
                    tif = _default_function_tinfo()
                # if function is a thunk function, define the actual function instead
                if ((func is not None)
                and (func.flags & ida_funcs.FUNC_THUNK)):
                    tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(func)
                    if tfunc_ea != ida_idaapi.BADADDR:
                        ea = tfunc_ea
                        name = _get_name_cached(ea)
                        if name == "":
                            name = f"data_{hex(ea)[2:]}"
                        tif = lift_type_from_address(ea)
                # 这里只需要确保函数符号存在。函数体统一在 insertAllFunctions() 中定义，
                # 避免在提升调用点时递归进入被调函数体，导致深调用链压垮 Python 栈。
                return lift_function(module, name, True, ea, tif)

            typ = lift_tif(tif)
            g_cmt = lift_from_address(module, ea, typ)
            g = ir.GlobalVariable(module, typ if g_cmt is None else g_cmt.type, name = name)
            g._source_ea = ea
            if g_cmt is not None:
                g.initializer = g_cmt
            global_by_ea[ea] = g

        if isinstance(g.type.pointee, (ir.IdentifiedStructType, ir.ArrayType)):
            g = builder.gep(g, (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)))
        g = _cast_slot(g, mop.size, knowntyp, builder)
        return g if dest else builder.load(g)
    elif t == ida_hexrays.mop_f: # 函数调用信息
        mcallinfo = mop.f
        f_args = []
        f_ret = []
        # 返回值寄存器是写入目标，取其存储地址
        for i in range(mcallinfo.retregs.size()):
            mopt = mcallinfo.retregs.at(i)
            f_ret.append(lift_mop(mopt, blk, builder, True))
        for arg in mcallinfo.args:
            typ = lift_tif(arg.type)
            f_arg = lift_mop(arg, blk, builder, False, typ)

            if arg.t == ida_hexrays.mop_h and f_arg == None:
                _note_helper(arg.helper)
                f_arg = blk.parent.parent.declare_intrinsic(arg.helper, fnty=ir.FunctionType(typ, []))

            if arg.t == ida_hexrays.mop_r and f_arg == None:
                llvm_arg = _get_register_storage(arg, blk, builder)
                f_arg = llvm_arg if mop.size == -1 else builder.load(llvm_arg)

            f_arg = typecast(f_arg, typ, builder)
            f_args.append(f_arg)
        return f_ret, f_args
    elif t == ida_hexrays.mop_a: # 操作数地址(mop_l\mop_v\mop_S\mop_r)
        val = lift_mop(mop.a, blk, builder, True)
        if isinstance(mop, (ida_hexrays.mcallarg_t, ida_hexrays.mop_addr_t)):
            val = typecast(val, lift_tif(mop.type), builder)
        elif knowntyp != None:
            val = typecast(val, knowntyp, builder)
        return val
    elif t == ida_hexrays.mop_h: # 辅助函数编号
        if _is_implemented_helper(mop.helper):
            _note_helper(mop.helper)
            return lift_intrinsic_function(blk.parent.parent, mop.helper)
        # 其他 helper 由调用点根据 mcallinfo 声明（宽度与调用信息一致）
        return None
    elif t == ida_hexrays.mop_str: # 字符串常量
        g = _make_cstr_global(blk.parent.parent, mop.cstr)
        return typecast(g, ir.IntType(8).as_pointer(), builder)
    elif t == ida_hexrays.mop_c: # switch分支和目标
        mcases = {}
        for i in range(mop.c.size()):
            dst = mop.c.targets[i]
            if mop.c.values[i].size() == 0:
                mcases["default"] = dst
            for j in range(mop.c.values[i].size()):
                src = mop.c.values[i][j]
                mcases[src] = dst
        return mcases
    elif t == ida_hexrays.mop_fn:
        typ = float_type(mop.size)
        try:
            fp = mop.fpc.fnum.float
        except Exception as e:
            # 读不出的浮点常量不能用一个普通常量顶替
            raise UnsupportedMicrocode("fp_constant_unreadable", f"{type(e).__name__}: {e}")
        return ir.Constant(typ, fp)
    elif t == ida_hexrays.mop_p:
        if dest:
            # 写入寄存器对需要拆分高低两半，调用 __PAIR 的结果不是可写存储
            raise UnsupportedMicrocode("pair_destination", mop.dstr())
        width = mop.size * 8
        if width not in PAIR_HELPER_TYPES:
            raise UnsupportedMicrocode(f"pair_width_{width}", mop.dstr())
        high = lift_mop(mop.pair.hop, blk, builder, dest)
        low = lift_mop(mop.pair.lop, blk, builder, dest)
        return _pair(builder, high, low, width)
    elif t == ida_hexrays.mop_sc:
        pass
    elif t == ida_hexrays.mop_z:
        return None
    mop_descs = {ida_hexrays.mop_r: "寄存器值",
                ida_hexrays.mop_n: "立即数",
                ida_hexrays.mop_d: "另一个指令",
                ida_hexrays.mop_l: "局部变量",
                ida_hexrays.mop_S: "栈变量",
                ida_hexrays.mop_b: "基本块编号（用于jmp/call指令）",
                ida_hexrays.mop_v: "全局变量",
                ida_hexrays.mop_f: "函数调用信息",
                ida_hexrays.mop_a: r"操作数地址(mop_l\mop_v\mop_S\mop_r)",
                ida_hexrays.mop_h: "辅助函数编号",
                ida_hexrays.mop_str: "字符串常量",
                ida_hexrays.mop_c: "switch分支和目标",
                ida_hexrays.mop_fn: "浮点常量",
                ida_hexrays.mop_p: "配对操作",
                ida_hexrays.mop_sc: "分散操作信息"
    }
    raise NotImplementedError(f"未实现: {mop.dstr()} 类型 {mop_descs[t]}")

def _make_cstr_global(module: ir.Module, text: str) -> ir.GlobalVariable:
    """microcode 字符串常量 -> 私有常量全局：按 UTF-8 字节定长，补结尾 \\0，名字不与已有符号冲突。"""
    try:
        raw = text.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        raw = text.encode("utf-8", "replace")
    raw += b"\x00"  # C 字符串字面量，原始二进制中以 \0 结尾
    str_type = ir.ArrayType(i8, len(raw))
    while True:
        name = f"cstr_{cstr_counter[0]}"
        cstr_counter[0] += 1
        if name not in module.globals:
            break
    g = ir.GlobalVariable(module, str_type, name=name)
    g.initializer = ir.Constant(str_type, bytearray(raw))
    g.linkage = "private"
    g.global_constant = True
    return g


def _store_as(l: ir.Value, d: ir.Value, blk: ir.Block, builder: ir.IRBuilder, d_typ: ir.Type = None, signed: bool = True):
    """
    将值 l 写入目标存储 d（d 是 lift_mop(dest=True) 得到的存储地址）。
    """
    if d is None:  # 目标不存在
        return l

    d = dedereference(d)
    if d_typ:
        d = typecast(d, d_typ, builder, signed)
    assert isinstance(d.type, ir.PointerType)

    if l is None or isinstance(l.type, ir.VoidType):
        return

    aggregate = (ir.IdentifiedStructType, ir.LiteralStructType, ir.ArrayType)
    if isinstance(l.type, aggregate) or isinstance(d.type.pointee, aggregate):
        # 聚合值整体写入，或者把标量的位写入聚合存储的起始位置
        d = typecast(d, l.type.as_pointer(), builder)
    else:
        l = typecast(l, d.type.pointee, builder, signed)

    return builder.store(l, d)


def _store_to_address(value: ir.Value, address: ir.Value, builder: ir.IRBuilder):
    """把 value 写到 address（一个地址值，整数或指针）指向的内存。"""
    if value is None or isinstance(value.type, ir.VoidType):
        return None
    ptr = typecast(address, value.type.as_pointer(), builder)
    return builder.store(value, ptr)

def create_intrinsic_function(module: ir.Module, func_name: str, ftif):
    """
    为IDA辅助函数创建内置函数。
    """
    argtypes = [lift_tif(arg.type) for arg in ftif.args]
    var_arg = bool(ftif.is_vararg())
    if var_arg:
        argtypes = argtypes[:int(ftif.solid_args)]
    rettype = lift_tif(ftif.return_type)
    # 签名完全来自调用信息（宽度正确，void 就是 void），不再把 void 改成 i8*；
    # 同名 helper 已按另一组调用信息声明过时，由 _lift_call 在调用点转换类型
    return module.declare_intrinsic(func_name, fnty=ir.FunctionType(rettype, argtypes, var_arg=var_arg))

def float_type(size: int) -> ir.Type:
    """
    按字节宽度返回 LLVM 浮点类型：4 -> float，8 -> double。
    其他宽度（x87 80 位扩展精度、16 字节、半精度等）不静默收窄成 double，拒绝该函数。
    """
    if size == 4:
        return ir.FloatType()
    if size == 8:
        return ir.DoubleType()
    raise UnsupportedMicrocode(f"fp_width_{size}", "only 32/64-bit floating point is lifted exactly")

# ============================================================================
# 指令提升辅助函数
# ============================================================================

def _as_int(value: ir.Value, size: int, builder: ir.IRBuilder, signed: bool = False) -> ir.Value:
    """把操作数按位解释为 size 字节的整数。"""
    return typecast(value, ir.IntType(size * 8), builder, signed)


def _handle_binary_arithmetic(l, r, d, op_func, blk, builder, ida_insn, allow_ptr=False, is_sub=False):
    """
    处理二元整数运算（加、减、乘、除、位运算、移位等）的辅助函数。

    :param l: 左操作数
    :param r: 右操作数
    :param d: 目标
    :param op_func: 操作函数（例如 builder.add, builder.mul）
    :param blk: 当前基本块
    :param builder: IR构建器
    :param ida_insn: IDA指令（用于获取大小信息）
    :param allow_ptr: 是否允许指针算术（add/sub）
    :param is_sub: 是否是减法（指针减整数时对偏移取负）
    """
    size = ida_insn.d.size
    # microcode 中整数运算作用在浮点值上时是位运算语义（如 fabs 的掩码），按位重解释
    if _is_fp_type(l.type):
        l = builder.bitcast(l, ir.IntType(_fp_bits(l.type)))
    if _is_fp_type(r.type):
        r = builder.bitcast(r, ir.IntType(_fp_bits(r.type)))

    l_is_ptr = isinstance(l.type, ir.PointerType)
    r_is_ptr = isinstance(r.type, ir.PointerType)
    if allow_ptr and l_is_ptr and not r_is_ptr:
        offset = _as_int(r, size, builder)
        if is_sub:
            offset = builder.neg(offset)
        math = builder.gep(typecast(l, i8ptr, builder), (offset, ))
        math = typecast(math, l.type, builder)
    elif allow_ptr and r_is_ptr and not l_is_ptr and not is_sub:
        offset = _as_int(l, size, builder)
        math = builder.gep(typecast(r, i8ptr, builder), (offset, ))
        math = typecast(math, r.type, builder)
    else:
        l = _as_int(l, size, builder)
        r = _as_int(r, size, builder)
        math = op_func(l, r)

    d = storecast(l, d, builder)
    return _store_as(math, d, blk, builder)


_FP_CMP_UNORDERED = {"!="}


def _compare(l, r, cmp_op, builder, ida_insn, signed=False):
    """生成整数或浮点比较，返回 i1。"""
    l_size = ida_insn.l.size
    if ida_insn.is_fpinsn() or (_is_fp_type(l.type) and _is_fp_type(r.type)):
        typ = float_type(l_size)
        l = typecast(l, typ, builder)
        r = typecast(r, typ, builder)
        if cmp_op in _FP_CMP_UNORDERED:
            return builder.fcmp_unordered(cmp_op, l, r)
        return builder.fcmp_ordered(cmp_op, l, r)
    l = _as_int(l, l_size, builder)
    r = typecast(r, l.type, builder)
    if signed:
        return builder.icmp_signed(cmp_op, l, r)
    return builder.icmp_unsigned(cmp_op, l, r)


def _handle_comparison(l, r, d, cmp_op, blk, builder, ida_insn, signed=False):
    """
    处理比较操作（setz, setnz, setg 等）的辅助函数。
    """
    cond = _compare(l, r, cmp_op, builder, ida_insn, signed)
    result_type = ir.IntType(ida_insn.d.size * 8)
    result = builder.select(cond, result_type(1), result_type(0))
    return _store_as(result, d, blk, builder)


class UnsupportedMicrocode(RuntimeError):
    """无法忠实翻译的 microcode：拒绝整个函数（走失败路径并按 reason 计数），不编造语义。"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _branch_target(builder, ida_insn, d, next_blk):
    target_blk = _resolve_block_target(builder.block.parent, ida_insn.d)
    if target_blk is not None:
        return target_blk
    if isinstance(d, ir.Block):
        return d
    # 不能把无法解析的跳转目标替换成下一个块，那会编造一条 CFG 边
    raise UnsupportedMicrocode("jcc_unresolved_target", ida_insn.dstr())


def _freeze_undef(builder: ir.IRBuilder, typ: ir.Type) -> ir.Value:
    """freeze undef：一个未知但在各次使用间保持一致的值（llvmlite 没有 freeze 构造器）。"""
    inst = ir.instructions.Instruction(builder.block, typ, "freeze", [ir.Constant(typ, ir.Undefined)])
    builder._insert(inst)
    return inst


def _finalize_nodefault_switches(func: ir.Function, builder: ir.IRBuilder):
    """
    Hex-Rays 只在 case 覆盖所有可达索引值时省略 jtbl 的 default（前驱已做范围检查或索引被掩码），
    此时 default 不可达。函数体提升完再追加 unreachable 块，避免打乱 mba 块号与出口块。
    """
    switches = getattr(func, "nodefault_switches", None)
    if not switches:
        return
    unreachable_blk = func.append_basic_block(name="jtbl.nodefault")
    with builder.goto_block(unreachable_blk):
        builder.unreachable()
    for switch in switches:
        switch.default = unreachable_blk


def _handle_conditional_jump(l, r, d, next_blk, cmp_op, builder, ida_insn, signed=False):
    """
    处理条件跳转操作（jz, jnz, jg, jl 等）的辅助函数。
    """
    cond = _compare(l, r, cmp_op, builder, ida_insn, signed)
    return builder.cbranch(cond, _branch_target(builder, ida_insn, d, next_blk), next_blk)

def _resolve_block_target(func, mop):
    if mop is None:
        return None
    if mop.t == ida_hexrays.mop_b:
        return func.blocks[mop.b]
    if mop.t == ida_hexrays.mop_v:
        return getattr(func, "block_by_ea", {}).get(mop.g)
    return None

def _handle_float_binary_op(l, r, d, op_func, blk, builder, ida_insn):
    """
    处理浮点二元运算（fadd, fsub, fmul, fdiv）的辅助函数。
    """
    typ = float_type(ida_insn.l.size)
    l = typecast(l, typ, builder)
    r = typecast(r, typ, builder)
    math = op_func(l, r)
    d = storecast(l, d, builder)
    return _store_as(math, d, blk, builder)


def _overflow_bit(l, r, op_func, builder, ida_insn):
    """调用 llvm.*.with.overflow.iN 并取出溢出位。"""
    size = ida_insn.l.size
    l = _as_int(l, size, builder)
    r = _as_int(r, size, builder)
    return builder.extract_value(op_func(l, r), 1)


def _shift_carry_bit(l, r, builder, ida_insn, left):
    """shl/shr 最后移出的那一位（即 x86 CF）。"""
    size = ida_insn.l.size
    width = ir.IntType(size * 8)
    l = _as_int(l, size, builder)
    r = typecast(r, width, builder)
    if left:
        amount = builder.sub(width(size * 8), r)
    else:
        amount = builder.sub(r, width(1))
    return builder.trunc(builder.lshr(l, amount), ir.IntType(1))


# 只有地址/段选择子语义、值不会被使用的操作数，不需要提升
_SKIP_L_OPS = {ida_hexrays.m_ldx, ida_hexrays.m_icall, ida_hexrays.m_nop, ida_hexrays.m_und}
_SKIP_R_OPS = {ida_hexrays.m_stx, ida_hexrays.m_ijmp, ida_hexrays.m_nop, ida_hexrays.m_und}
# 这些指令的 d 操作数是值（地址/调用信息），而不是写入目标
_D_AS_VALUE_OPS = {ida_hexrays.m_call, ida_hexrays.m_icall, ida_hexrays.m_stx}
_SKIP_D_OPS = {ida_hexrays.m_nop}

_INT_BINARY_OPS = {
    ida_hexrays.m_mul: "mul",
    ida_hexrays.m_udiv: "udiv",
    ida_hexrays.m_sdiv: "sdiv",
    ida_hexrays.m_umod: "urem",
    ida_hexrays.m_smod: "srem",
    ida_hexrays.m_or: "or_",
    ida_hexrays.m_and: "and_",
    ida_hexrays.m_xor: "xor",
    ida_hexrays.m_shl: "shl",
    ida_hexrays.m_shr: "lshr",
    ida_hexrays.m_sar: "ashr",
}
_SET_OPS = {
    ida_hexrays.m_setnz: ("!=", False),
    ida_hexrays.m_setz: ("==", False),
    ida_hexrays.m_setae: (">=", False),
    ida_hexrays.m_setb: ("<", False),
    ida_hexrays.m_seta: (">", False),
    ida_hexrays.m_setbe: ("<=", False),
    ida_hexrays.m_setg: (">", True),
    ida_hexrays.m_setge: (">=", True),
    ida_hexrays.m_setl: ("<", True),
    ida_hexrays.m_setle: ("<=", True),
}
_JCC_OPS = {
    ida_hexrays.m_jnz: ("!=", False),
    ida_hexrays.m_jz: ("==", False),
    ida_hexrays.m_jae: (">=", False),
    ida_hexrays.m_jb: ("<", False),
    ida_hexrays.m_ja: (">", False),
    ida_hexrays.m_jbe: ("<=", False),
    ida_hexrays.m_jg: (">", True),
    ida_hexrays.m_jge: (">=", True),
    ida_hexrays.m_jl: ("<", True),
    ida_hexrays.m_jle: ("<=", True),
}
_FLOAT_BINARY_OPS = {
    ida_hexrays.m_fadd: "fadd",
    ida_hexrays.m_fsub: "fsub",
    ida_hexrays.m_fmul: "fmul",
    ida_hexrays.m_fdiv: "fdiv",
}
# 真正的空操作。m_ijmp / m_push / m_pop / m_und / m_ext 需要建模或拒绝，不能当作空操作
_NO_OP_OPS = {ida_hexrays.m_nop}
# 需要栈指针 / 跳转目标模型、lifter 无法忠实翻译的指令
_REJECT_OPS = {ida_hexrays.m_ijmp: "m_ijmp", ida_hexrays.m_push: "m_push", ida_hexrays.m_pop: "m_pop"}
# ldx/stx 的段选择子被忽略（按平坦内存建模），这只对 cs/ds/ss/es 成立；fs/gs 基址（TLS 等）不能丢
_NONFLAT_SELECTORS = {"fs", "gs"}


def _check_flat_selector(sel_mop, ida_insn):
    if sel_mop is None or sel_mop.t != ida_hexrays.mop_r:
        return
    name = sel_mop.dstr().split(".")[0].split("{")[0].strip().lower()
    if name in _NONFLAT_SELECTORS:
        raise UnsupportedMicrocode(f"segment_selector_{name}", ida_insn.dstr())


_AGGREGATE_TYPES = (ir.IdentifiedStructType, ir.LiteralStructType, ir.ArrayType)


def _callinfo(ida_insn):
    d_mop = ida_insn.d
    return d_mop.f if d_mop is not None and d_mop.t == ida_hexrays.mop_f else None


def _callinfo_return_type(ida_insn):
    callinfo = _callinfo(ida_insn)
    if callinfo is None:
        return None
    tif = callinfo.return_type
    if tif is None or tif.empty():
        return None
    return lift_tif(tif)


def _check_single_retreg(ida_insn):
    callinfo = _callinfo(ida_insn)
    if callinfo is not None and callinfo.retregs.size() > 1:
        # 多寄存器返回（如 rax:rdx）没有可靠的拆分规则，直接、间接、helper 调用统一拒绝
        raise UnsupportedMicrocode("call_multi_retregs", ida_insn.dstr())


def _abi_class(typ: ir.Type) -> str:
    if _is_fp_type(typ):
        return "fp"
    if isinstance(typ, _AGGREGATE_TYPES):
        return "aggregate"
    return "int"


def _callinfo_signature(ida_insn, args) -> ir.FunctionType:
    """
    间接调用 / 无类型被调目标的调用签名，取自 microcode 的 mcallinfo：
    返回类型、固定参数个数（solid_args）和是否变参。签名只由调用信息决定，
    外层指令需要的类型只在调用之后做转换（lift_mop），不能改变调用的 ABI。
    聚合返回值、多返回寄存器、缺少调用信息时明确拒绝，不猜一个通用类型。
    """
    callinfo = _callinfo(ida_insn)
    if callinfo is None:
        raise UnsupportedMicrocode("call_without_callinfo", ida_insn.dstr())
    _check_single_retreg(ida_insn)
    ret_type = _callinfo_return_type(ida_insn)
    if ret_type is None:
        raise UnsupportedMicrocode("call_unknown_return_type", ida_insn.dstr())
    if isinstance(ret_type, _AGGREGATE_TYPES):
        # 聚合返回值的 ABI（sret / 多寄存器）无法从按值返回的 LLVM 类型推出
        raise UnsupportedMicrocode("call_aggregate_return", ida_insn.dstr())
    var_arg = bool(callinfo.is_vararg())
    fixed = min(int(callinfo.solid_args), len(args)) if var_arg else len(args)
    return ir.FunctionType(ret_type, [arg.type for arg in args[:fixed]], var_arg=var_arg)


def _store_call_result(ret, rets, ftype, blk, builder):
    if isinstance(ftype.return_type, ir.VoidType) or not rets:
        return
    if len(rets) > 1:
        raise UnsupportedMicrocode("call_multi_retregs", f"{len(rets)} return registers")
    slot = dedereference(rets[0])
    if (
        isinstance(ret.type, _AGGREGATE_TYPES)
        and isinstance(slot.type, ir.PointerType)
        and _type_size(slot.type.pointee) < _type_size(ret.type)
    ):
        # 聚合返回值整体写进更小的返回寄存器槽会越界
        raise UnsupportedMicrocode("call_aggregate_into_smaller_slot", str(ret.type))
    _store_as(ret, rets[0], blk, builder)


def _lift_ext(ida_insn, l, r, d, blk, builder):
    """
    m_ext（未转成 microcode 的机器指令，如 tzcnt、VIA PadLock）：建模为调用一个未知外部函数。
    LLVM 视未知外部调用为可读写任意内存、有任意副作用，是保守而非丢弃副作用的模型。
    """
    module = blk.parent.parent
    mnemonic = ""
    with suppress(Exception):
        mnemonic = idc.print_insn_mnem(ida_insn.ea) or ""
    name = "__ida_ext_" + _sanitize_symbol_name(mnemonic or "unknown")
    _note_approx(f"opaque_ext:{mnemonic or 'unknown'}")
    args = [v for v in (l, r) if v is not None and not isinstance(v, ir.Block)]
    has_result = d is not None and ida_insn.d.size > 0
    ret_type = ir.IntType(ida_insn.d.size * 8) if has_result else ir.VoidType()
    fnty = ir.FunctionType(ret_type, [arg.type for arg in args])
    callee = module.globals.get(name)
    if not isinstance(callee, ir.Function):
        if callee is not None:
            name = module.get_unique_name(name)
        callee = ir.Function(module, fnty, name)
    target = callee if callee.function_type == fnty else typecast(callee, fnty.as_pointer(), builder)
    ret = builder.call(target, args)
    if has_result:
        _store_as(ret, d, blk, builder)
    return ret


def _lift_call(ida_insn, l, d, blk, builder, result_type):
    rets, args = d
    _check_single_retreg(ida_insn)
    if not isinstance(l.type, ir.PointerType) or not isinstance(l.type.pointee, ir.FunctionType):
        # 被调目标没有函数类型：签名来自 mcallinfo，不再统一假定返回 i8*
        ftype = _callinfo_signature(ida_insn, args)
        l = typecast(l, ftype.as_pointer(), builder)
    else:
        ftype = l.type.pointee
        callinfo_ret = _callinfo_return_type(ida_insn)
        # 调用信息里结果宽度为 0（如 "=> int .0"）表示本调用点不使用返回值，此时返回类型不影响语义
        result_used = ida_insn.d.size > 0
        if result_used and callinfo_ret is not None and not isinstance(callinfo_ret, ir.VoidType):
            if isinstance(ftype.return_type, ir.VoidType):
                # 被调函数声明为 void，但本调用点的调用信息给出了返回类型：以调用信息为准（只改调用点）
                ftype = ir.FunctionType(callinfo_ret, ftype.args, var_arg=ftype.var_arg)
                l = typecast(l, ftype.as_pointer(), builder)
            elif _abi_class(callinfo_ret) != _abi_class(ftype.return_type):
                # 声明与调用信息的返回值不在同一类寄存器（整数/浮点/聚合）：无法确定 ABI
                raise UnsupportedMicrocode("call_return_abi_conflict", f"{ftype.return_type} vs {callinfo_ret}")
        if (
            result_used
            and isinstance(ftype.return_type, _AGGREGATE_TYPES)
            and ida_insn.l.t != ida_hexrays.mop_h
        ):
            # 按值返回聚合：LLVM 调用不体现源 ABI（sret / 寄存器拆分），记为近似
            _note_approx("aggregate_return_call")

        params = ftype.args
        for i, param_type in enumerate(params):
            if i >= len(args):
                # 调用点缺少的实参（声明比 microcode 实参多）：undef，并记为近似
                _note_approx("call_missing_args")
                args.append(ir.Constant(param_type, ir.Undefined))
            elif args[i].type != param_type:
                args[i] = typecast(args[i], param_type, builder)
        if not ftype.var_arg and len(args) > len(params):
            # microcode 给出的实参比声明多（声明来自不准确的 IDA 存储类型）：
            # 保留全部实参，只在调用点转换被调函数类型，不丢数据流
            ftype = ir.FunctionType(ftype.return_type, list(params) + [arg.type for arg in args[len(params):]])
            l = typecast(l, ftype.as_pointer(), builder)
        # 可变参数函数：多出来的实参原样传入

    ret = builder.call(l, args)
    _store_call_result(ret, rets, l.type.pointee, blk, builder)
    return ret


def lift_insn(ida_insn: ida_hexrays.minsn_t, blk: ir.Block, builder: ir.IRBuilder, next_blk=None, result_type=None) -> ir.Instruction:
    """
    Lifts a single IDA microcode instruction to LLVM IR.

    This is the main instruction translation function. It processes IDA Hexrays
    microcode instructions and emits corresponding LLVM IR instructions.

    Process:
    1. Lift left operand (l), right operand (r), and destination (d)
    2. Handle special cases (load operations use addresses, not values)
    3. Create unknown intrinsic functions if needed (mop_h helpers)
    4. Dispatch to appropriate handler based on opcode

    :param ida_insn: IDA microcode instruction to lift
    :param blk: Current LLVM basic block
    :param builder: LLVM IR builder for emitting instructions
    :return: The generated LLVM instruction, or None for no-ops
    """
    builder.position_at_end(blk)
    op = ida_insn.opcode
    l_mop = ida_insn.l
    r_mop = ida_insn.r
    d_mop = ida_insn.d

    if op in _REJECT_OPS:
        # 先于操作数提升检查，保证记录的是指令本身的拒绝原因
        raise UnsupportedMicrocode(_REJECT_OPS[op], ida_insn.dstr())
    if op == ida_hexrays.m_ldx:
        _check_flat_selector(l_mop, ida_insn)
    elif op == ida_hexrays.m_stx:
        _check_flat_selector(r_mop, ida_insn)

    l_known_type = None
    r_known_type = None
    if op in FLOAT_L_OPS and l_mop.t == ida_hexrays.mop_d and l_mop.size > 0:
        l_known_type = float_type(l_mop.size)
    if op in FLOAT_R_OPS and r_mop.t == ida_hexrays.mop_d and r_mop.size > 0:
        r_known_type = float_type(r_mop.size)

    l = None if op in _SKIP_L_OPS else lift_mop(l_mop, blk, builder, knowntyp=l_known_type)
    r = None if op in _SKIP_R_OPS else lift_mop(r_mop, blk, builder, knowntyp=r_known_type)
    # 指令目标通常是写入地址；call/icall 的 d 是调用信息，stx 的 d 是要写入的内存地址值
    d = None if op in _SKIP_D_OPS else lift_mop(d_mop, blk, builder, op not in _D_AS_VALUE_OPS)

    # 有确定语义的 helper 直接展开成 LLVM 运算；其余声明为不透明外部函数并标记
    if l is None and op == ida_hexrays.m_call and l_mop.t == ida_hexrays.mop_h:
        _check_single_retreg(ida_insn)
        rets, args = d
        ret_type = _callinfo_return_type(ida_insn) or ir.VoidType()
        value = _lift_semantic_helper(l_mop.helper, ret_type, args, builder)
        if value is not None:
            if value is True:  # 没有返回值的 helper（store/fence）
                return None
            if rets:
                _store_as(value, rets[0], blk, builder)
            return value
        _note_helper(l_mop.helper)
        l = create_intrinsic_function(blk.parent.parent, l_mop.helper, d_mop.f)

    if next_blk is None:
        next_blk = getattr(blk, "next_blk", None)
        if next_blk is None:
            blocks = blk.parent.blocks
            with suppress(ValueError, AttributeError):
                blk_index = blocks.index(blk)
                next_blk = blocks[blk_index + 1] if blk_index + 1 < len(blocks) else None

    if op == ida_hexrays.m_mov:  # mov  l, d
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif op == ida_hexrays.m_ldx:  # ldx  {l=sel,r=off}, d  从内存加载值
        # ldx 只搬运数据：非 32/64 位的浮点寄存器加载（如 AArch64 q 寄存器整体拷贝）按同宽整数搬运
        if ida_insn.is_fpinsn() and d_mop.size in (4, 8):
            typ = float_type(d_mop.size)
        else:
            typ = ir.IntType(d_mop.size * 8)
        r = typecast(r, typ.as_pointer(), builder)
        r = builder.load(r)
        d = storecast(r, d, builder)
        return _store_as(r, d, blk, builder)
    elif op == ida_hexrays.m_stx:  # stx  l, {r=sel, d=off}  写入 d 指向的内存
        return _store_to_address(l, d, builder)
    elif op == ida_hexrays.m_add:
        return _handle_binary_arithmetic(l, r, d, builder.add, blk, builder, ida_insn, allow_ptr=True)
    elif op == ida_hexrays.m_sub:
        return _handle_binary_arithmetic(l, r, d, builder.sub, blk, builder, ida_insn, allow_ptr=True, is_sub=True)
    elif op in _INT_BINARY_OPS:
        return _handle_binary_arithmetic(l, r, d, getattr(builder, _INT_BINARY_OPS[op]), blk, builder, ida_insn)
    elif op in _SET_OPS:
        cmp_op, signed = _SET_OPS[op]
        return _handle_comparison(l, r, d, cmp_op, blk, builder, ida_insn, signed=signed)
    elif op in _JCC_OPS:
        cmp_op, signed = _JCC_OPS[op]
        return _handle_conditional_jump(l, r, d, next_blk, cmp_op, builder, ida_insn, signed=signed)
    elif op == ida_hexrays.m_call:  # call   l d   l是mop_v或mop_b或mop_h
        return _lift_call(ida_insn, l, d, blk, builder, result_type)
    elif op == ida_hexrays.m_goto:  # goto   l    l是mop_v或mop_b
        target_blk = _resolve_block_target(builder.block.parent, l_mop)
        if target_blk is not None:
            return builder.branch(target_blk)
        if isinstance(l, ir.Block):
            return builder.branch(l)
        raise UnsupportedMicrocode("goto_unresolved_target", ida_insn.dstr())
    elif op == ida_hexrays.m_jcnd:  # jcnd   l,    d   d is mop_v or mop_b
        if _is_fp_type(l.type):
            l = builder.bitcast(l, ir.IntType(_fp_bits(l.type)))
        elif isinstance(l.type, ir.PointerType):
            l = builder.ptrtoint(l, ir.IntType(ptrsize))
        if not isinstance(l.type, ir.IntType):
            l = _as_int(l, l_mop.size, builder)
        if l.type.width != 1:
            l = builder.icmp_unsigned('!=', l, l.type(0))
        return builder.cbranch(l, _branch_target(builder, ida_insn, d, next_blk), next_blk)
    elif op == ida_hexrays.m_ret:  # ret  返回
        func = blk.parent
        ret_type = func.type.pointee.return_type
        if (
            not isinstance(ret_type, ir.VoidType)
            and l is not None
            and not isinstance(l.type, ir.VoidType)
            and "funcresult" in func.lvars
        ):
            _store_as(l, func.lvars["funcresult"], blk, builder)
        exit_blk = func.blocks[-1]
        if blk is not exit_blk:
            if getattr(blk, "terminator", None) is not None and not _block_ends_with_terminator(blk):
                blk.terminator = None
            return builder.branch(exit_blk)
        return
    elif op in _NO_OP_OPS:
        return
    elif op == ida_hexrays.m_und:  # und d：d 的值变为未定义
        if d is not None:
            slot = dedereference(d)
            _store_as(_freeze_undef(builder, slot.type.pointee), slot, blk, builder)
        return
    elif op == ida_hexrays.m_ext:  # ext in1, in2, out1：外部机器指令
        return _lift_ext(ida_insn, l, r, d, blk, builder)
    elif op == ida_hexrays.m_ldc:  # ldc  l=const,d   加载常量
        value = ir.Constant(ir.IntType(l_mop.size * 8), l_mop.nnn.value)
        return _store_as(value, d, blk, builder)
    elif op == ida_hexrays.m_neg:  # neg  l, d   取负
        l = _as_int(l, d_mop.size, builder)
        l = builder.neg(l)
        d = storecast(l, d, builder)
        return _store_as(l, d, blk, builder)
    elif op == ida_hexrays.m_lnot:  # lnot l, d   逻辑非
        l = _as_int(l, l_mop.size, builder)
        cmp = builder.icmp_unsigned("==", l, l.type(0))
        result = builder.zext(cmp, ir.IntType(d_mop.size * 8))
        return _store_as(result, d, blk, builder)
    elif op == ida_hexrays.m_bnot:  # bnot l, d   按位非
        l = _as_int(l, d_mop.size, builder)
        l = builder.not_(l)
        return _store_as(l, d, blk, builder)
    elif op == ida_hexrays.m_xds:  # xds  l, d   符号扩展
        l = _as_int(l, l_mop.size, builder)
        return _store_as(_as_int(l, d_mop.size, builder, signed=True), d, blk, builder)
    elif op == ida_hexrays.m_xdu:  # xdu  l, d   无符号扩展
        l = _as_int(l, l_mop.size, builder)
        return _store_as(_as_int(l, d_mop.size, builder, signed=False), d, blk, builder, signed=False)
    elif op == ida_hexrays.m_low:  # low  l, d   取低位部分
        l = _as_int(l, l_mop.size, builder)
        return _store_as(_as_int(l, d_mop.size, builder), d, blk, builder)
    elif op == ida_hexrays.m_high:  # high l, d   取高位部分
        l = _as_int(l, l_mop.size, builder)
        shift = (l_mop.size - d_mop.size) * 8
        if shift > 0:
            l = builder.lshr(l, l.type(shift))
        return _store_as(_as_int(l, d_mop.size, builder), d, blk, builder)
    elif op == ida_hexrays.m_cfadd:  # cfadd l, r, d=carry      (l+r) 的进位
        bit = _overflow_bit(l, r, builder.uadd_with_overflow, builder, ida_insn)
        return _store_as(bit, d, blk, builder)
    elif op == ida_hexrays.m_ofadd:  # ofadd l, r, d=overf      (l+r) 的有符号溢出
        bit = _overflow_bit(l, r, builder.sadd_with_overflow, builder, ida_insn)
        return _store_as(bit, d, blk, builder)
    elif op == ida_hexrays.m_cfshl:  # cfshl l, r, d=carry      (l<<r) 的进位
        return _store_as(_shift_carry_bit(l, r, builder, ida_insn, left=True), d, blk, builder)
    elif op == ida_hexrays.m_cfshr:  # cfshr l, r, d=carry      (l>>r) 的进位
        return _store_as(_shift_carry_bit(l, r, builder, ida_insn, left=False), d, blk, builder)
    elif op == ida_hexrays.m_sets:  # sets  l,d=byte  SF=1Sign
        l = _as_int(l, l_mop.size, builder)
        cond = builder.icmp_signed("<", l, l.type(0))
        result_type = ir.IntType(d_mop.size * 8)
        result = builder.select(cond, result_type(1), result_type(0))
        return _store_as(result, d, blk, builder)
    elif op == ida_hexrays.m_seto:  # seto  l,  r, d=byte  OF=1Overflow of (l-r)
        bit = _overflow_bit(l, r, builder.ssub_with_overflow, builder, ida_insn)
        return _store_as(bit, d, blk, builder)
    elif op == ida_hexrays.m_setp:  # setp  l,  r, d=byte  PF=1Unordered/Parity  *F
        if ida_insn.is_fpinsn():
            typ = float_type(l_mop.size)
            cond = builder.fcmp_unordered("uno", typecast(l, typ, builder), typecast(r, typ, builder))
            return _store_as(cond, d, blk, builder)
        func_name = f"setp_{l_mop.size}_{d_mop.size}"
        _note_approx("opaque_setp")
        module = blk.parent.parent
        if func_name in module.globals:
            f_setp = module.get_global(func_name)
        else:
            f_setp = module.declare_intrinsic(func_name, fnty=ir.FunctionType(ir.IntType(d_mop.size*8), [ir.IntType(l_mop.size*8), ir.IntType(32)]))
        l = _as_int(l, l_mop.size, builder)
        l = builder.call(f_setp, [l, ir.Constant(ir.IntType(32), d_mop.size)])
        return _store_as(l, d, blk, builder)
    elif op == ida_hexrays.m_jtbl:  # jtbl   l, r=mcases    Table jump
        l = _as_int(l, l_mop.size, builder)
        func = blk.parent
        blocks = func.basic_blocks
        if "default" in r:
            switch = builder.switch(l, blocks[r["default"]])
        else:
            # 没有 default：先占位，函数提升完后由 _finalize_nodefault_switches 指向 unreachable 块。
            # “case 覆盖了所有可能值”是 Hex-Rays 的结论而非已证明的事实，记为近似
            _note_approx("jtbl_assumed_exhaustive")
            switch = builder.switch(l, blocks[r[next(iter(r))]])
            if not hasattr(func, "nodefault_switches"):
                func.nodefault_switches = []
            func.nodefault_switches.append(switch)
        for value, target in r.items():
            if isinstance(value, int):
                switch.add_case(value, blocks[target])
        return switch
    elif op == ida_hexrays.m_icall:  # icall  {l=sel, r=off} d    间接调用
        rets, args = d
        ftype = _callinfo_signature(ida_insn, args)
        f = typecast(r, ftype.as_pointer(), builder)
        ret = builder.call(f, args)
        _store_call_result(ret, rets, ftype, blk, builder)
        return ret
    elif op == ida_hexrays.m_f2i or op == ida_hexrays.m_f2u:  # 浮点 -> (无符号)整数
        l = typecast(l, float_type(l_mop.size), builder)
        int_type = ir.IntType(d_mop.size * 8)
        value = builder.fptosi(l, int_type) if op == ida_hexrays.m_f2i else builder.fptoui(l, int_type)
        return _store_as(value, d, blk, builder)
    elif op == ida_hexrays.m_i2f or op == ida_hexrays.m_u2f:  # (无符号)整数 -> 浮点
        l = _as_int(l, l_mop.size, builder)
        typ = float_type(d_mop.size)
        value = builder.sitofp(l, typ) if op == ida_hexrays.m_i2f else builder.uitofp(l, typ)
        return _store_as(value, d, blk, builder)
    elif op == ida_hexrays.m_f2f:  # 改变浮点精度
        l = typecast(l, float_type(l_mop.size), builder)
        target_type = float_type(d_mop.size)
        l = typecast(l, target_type, builder)
        if d != None and d.type != target_type.as_pointer():
            d = typecast(d, target_type.as_pointer(), builder)
        return _store_as(l, d, blk, builder)
    elif op == ida_hexrays.m_fneg:  # -l => d
        l = typecast(l, float_type(l_mop.size), builder)
        return _store_as(builder.fneg(l), d, blk, builder)
    elif op in _FLOAT_BINARY_OPS:
        return _handle_float_binary_op(l, r, d, getattr(builder, _FLOAT_BINARY_OPS[op]), blk, builder, ida_insn)
    else:
        raise NotImplementedError(f"未实现 {ida_insn.dstr()}")


class BIN2LLVMController():
    """
    IDA到LLVM IR翻译过程的主控制器。
    
    该类统筹完整的二进制到LLVM的提升工作流程：
    1. 通过反编译所有函数并收集元数据来初始化
    2. 为所有数据项创建LLVM全局变量
    3. 将.text段函数提升为LLVM IR
    4. 将生成的IR模块保存到文件
    """
    def __init__(self, target_mode: str = "source"):
        """使用空的LLVM模块初始化控制器。"""
        self.m = ir.Module(context=llvm_context)
        self._name_cache = {}
        self._func_cache = {}
        self._function_eas = []
        self._stream_file = None
        self._streamed_functions = set()
        self._streamed_types = set()
        if target_mode == "ida":
            target_mode = "source"  # 旧名字：现在的 source 模式给出完整的源目标 layout
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        self.target_mode = target_mode
        self.memory_contract = MEMORY_CONTRACTS[target_mode]
        if target_mode == "host":
            self._set_module_target_from_host()
            _set_sizing_layout(None)
        else:
            self._set_module_target_from_source()

    def _get_name(self, ea):
        return _get_name_cached(ea)

    def _get_func(self, ea):
        return _get_func_cached(ea)

    def _set_module_target_from_host(self):
        """
        host 兼容模式：宿主 triple/layout。类型大小仍按源指针宽度计算，32 位/大端源 binary 的
        内存布局与输出 layout 不一致（不做内存语义转换），只用于复现旧数据。
        """
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        triple = llvm.get_default_triple()
        target_machine = llvm.Target.from_triple(triple).create_target_machine()
        self.m.triple = triple
        self.m.data_layout = str(target_machine.target_data)

    def _set_module_target_from_source(self):
        """source 模式：源目标的 triple 与 data layout，类型大小按同一 layout 计算。"""
        is_elf = True
        with suppress(AttributeError):
            is_elf = ida_ida.inf_get_filetype() == ida_ida.f_ELF
        spec = source_target_spec(ida_ida.inf_get_procname(), ptrsize, is_big_endian, is_elf)
        self.m.triple = spec["triple"]
        self.m.data_layout = spec["datalayout"]
        _set_sizing_layout(spec["datalayout"])

    def insertAllFunctions(self):
        """
        将.text段的所有函数提升为LLVM IR。
        
        遍历二进制文件中的所有函数，并将.text段中的函数
        翻译为LLVM IR表示。
        """
        function_eas = self._function_eas or list(idautils.Functions())
        for f_ea in function_eas:
            self.insertFunctionAtEa(f_ea)

    def _decompile_function(self, ea):
        no_cache = ea in refreshed_funcs
        if no_cache:
            ida_hf = ida_hexrays.hexrays_failure_t()
            pfunc = ida_hexrays.decompile(ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE)
        else:
            pfunc = ida_hexrays.decompile(ea)
        if pfunc is not None and getattr(pfunc, "type", None) is not None:
            ptext[ea] = pfunc
            function_signature_cache[ea] = pfunc.type
        return pfunc

    def _drop_partial_definition(self, ea):
        """提升中途失败的函数只保留声明，不输出半截函数体。"""
        name = self._get_name(ea) or f"data_{hex(ea)[2:]}"
        func = self.m.globals.get(name)
        if not isinstance(func, ir.Function) or getattr(func, "_streamed_definition", False):
            return
        func.blocks.clear()
        for attr in ("lvars", "block_by_ea", "stack_size", "stack_extent", "nodefault_switches"):
            with suppress(AttributeError):
                delattr(func, attr)

    def _trim_ptext(self, keep_ea=None):
        for cached_ea in list(ptext):
            if cached_ea != keep_ea:
                with suppress(KeyError):
                    del ptext[cached_ea]

    def insertFunctionAtEa(self, ea):
        """
        将给定地址的特定函数提升为LLVM IR。
        
        :param ea: 要提升的函数的有效地址
        """
        current_approx.clear()
        record = {"ea": hex(ea), "name": self._get_name(ea)}
        try:
            pfunc = ptext.get(ea)
            if pfunc is None or getattr(pfunc, "type", None) is None:
                with suppress(KeyError):
                    del ptext[ea]
                pfunc = self._decompile_function(ea)
            if pfunc is None or getattr(pfunc, "type", None) is None:
                lift_stats["no_decompile"] += 1
                record["status"] = "no_decompile"
                return

            typ = pfunc.type
            function_signature_cache[ea] = typ
            func_name = self._get_name(ea)
            lifted = lift_function(self.m, func_name, False, ea, typ)
            defined = isinstance(lifted, ir.Function) and not lifted.is_declaration
            self._stream_function_if_enabled(lifted)
            if defined:
                # 写出之后才计为成功
                lift_stats["defined"] += 1
                record["status"] = "lifted"
                record["llvm_name"] = lifted.name
                if current_approx:
                    lift_stats["approx"] += 1
                    record["approx"] = sorted(current_approx)
            else:
                record["status"] = "not_defined"
        except Exception as e:
            lift_stats["failed"] += 1
            if isinstance(e, UnsupportedMicrocode):
                unsupported_reasons[e.reason] += 1
                record.update(status="rejected", reason=e.reason, detail=str(e)[:200])
            else:
                record.update(status="error", reason=f"{type(e).__name__}: {e}"[:300])
            logging.debug(f"在 {hex(ea)} 处提升函数失败: {e}")
            self._drop_partial_definition(ea)
        finally:
            function_records.append(record)
            with suppress(KeyError):
                del ptext[ea]
            self._trim_ptext()
            # 释放 Hex-Rays 内部缓存的 cfunc，不影响后续反编译结果，显著降低峰值内存
            ida_hexrays.mark_cfunc_dirty(ea, False)

    def create_global(self, ea, width, str_dict, pending):
        """
        Creates LLVM global variables for IDA data items.
        
        Process:
        1. Extract data item name and type information from IDA
        2. Handle special cases:
           - Strings: Create constant string globals
           - External functions: Create declarations
           - Thunk functions: Resolve to actual target
           - Data items: Create initialized global variables
        
        :param ea: Address of the data item
        :param width: Size of the data item in bytes
        :param str_dict: Dictionary mapping addresses to string data
        :param pending: 收集 (global, ea, type)，初始值在所有全局变量创建后再计算，
                        这样指针初始值可以引用任意位置的全局变量
        """
        # 获取数据项名称和类型信息
        name = self._get_name(ea)
        if name == "":
            name = f"data_{hex(ea)[2:]}"

        # 如果数据项是字符串，创建字符串全局变量
        if ea in str_dict:
            strlen = str_dict[ea]
            str_csnt = ida_bytes.get_bytes(ea, strlen) or b""
            if len(str_csnt) < strlen:
                str_csnt = str_csnt + b"\x00" * (strlen - len(str_csnt))
            strType = ir.ArrayType(ir.IntType(8), strlen)
            g = ir.GlobalVariable(self.m, strType, name=name)
            g.initializer = ir.Constant(strType, bytearray(str_csnt))
            g.linkage = "private"
            g.global_constant = True
            global_by_ea[ea] = g
            return

        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, ea):
            if not ENABLE_DATA_TYPE_GUESS:
                return
            ida_typeinf.guess_tinfo(tif, ea)

        # 如果数据项是外部函数，创建声明；普通数据段里的函数指针变量按数据处理
        is_xtrn = _is_extern_segment(ea)
        if tif.is_func() or (tif.is_funcptr() and is_xtrn):
            if tif.is_funcptr():
                tif = tif.get_ptrarr_object()
            # if function is a thunk function, define the actual function instead
            func = self._get_func(ea)
            if ((func is not None)
            and (func.flags & ida_funcs.FUNC_THUNK)): 
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(func)
                if tfunc_ea != ida_idaapi.BADADDR:
                    ea = tfunc_ea
                    name = self._get_name(ea)
                    func = self._get_func(ea)
            
            # 为声明函数创建定义，
            if ((func is None)
            # 或者函数是库函数，
            or (func.flags & ida_funcs.FUNC_LIB) 
            # 或者函数在XTRN段中声明，
            or _is_extern_segment(ea)):
                # 返回函数声明
                g = lift_function(self.m, name, True, ea, tif)

        # 其他数据项，可能是 int/float/数组/结构体
        else:
            typ = lift_tif(tif, width)
            g = ir.GlobalVariable(self.m, typ, name = name)
            g._source_ea = ea
            global_by_ea[ea] = g
            pending.append((g, ea, typ))

    def initialize(self):
        """
        Initializes the LLVM module by preparing all necessary metadata.
        
        This crucial preparation step:
        1. Decompiles all functions in the binary using IDA Hexrays
        2. Collects all string constants identified by IDA
        3. Creates LLVM global variables for all data items in non-executable segments
        
        Function addresses and available type signatures are collected up front.
        Decompiled function bodies are kept only while each function is lifted.
        
        Note: Decompilation may fail for some functions (e.g., library stubs,
        corrupted code), which are silently skipped.
        """
        # 步骤1：收集函数地址和IDA已有签名；函数体按需反编译，避免长期持有所有pfunc/mba。
        self._function_eas = list(idautils.Functions())
        for func in self._function_eas:
            tif = ida_typeinf.tinfo_t()
            try:
                if ida_nalt.get_tinfo(tif, func) and (tif.is_func() or tif.is_funcptr()):
                    function_signature_cache[func] = tif
            except Exception as e:
                logging.debug(f"在 {hex(func)} 处获取函数签名失败: {e}")

        # 步骤2：收集所有字符串常量
        str_dict = {}
        for s in idautils.Strings():
            str_dict[s.ea] = s.length

        # 步骤3：为非执行段中的所有数据项创建全局变量（先建符号，再算初始值）
        pending = []
        for i in range(idaapi.get_segm_qty()):
            segm = idaapi.getnseg(i)
            if segm is None:
                continue
            if segm.perm & idaapi.SEGPERM_EXEC:
                continue
            for head in idautils.Heads(segm.start_ea, segm.end_ea):
                end = ida_bytes.get_item_end(head)
                if end <= head:
                    continue
                self.create_global(head, end - head, str_dict, pending)
        for g, ea, typ in pending:
            init = lift_from_address(self.m, ea, typ)
            if init is not None:
                g.initializer = init
            # init 为 None：内容未知，保留为 external 声明

    def begin_stream_to_file(self, filename, header_lines=()):
        self._stream_file = open(filename, 'w')
        for line in header_lines:
            self._stream_file.write(f"; {line}\n")
        _write_module_header(self.m, self._stream_file)

    def _stream_function_if_enabled(self, func):
        if self._stream_file is None:
            return
        if not isinstance(func, ir.Function) or func.is_declaration:
            return
        if func.name in self._streamed_functions:
            return

        self._stream_new_type_declarations()
        self._stream_file.write(str(func))
        self._stream_file.write("\n")
        self._streamed_functions.add(func.name)
        func.blocks.clear()
        func._streamed_definition = True

    def _stream_new_type_declarations(self):
        if self._stream_file is None:
            return
        # 只输出上次之后新建的类型，总开销 O(F + T)
        for typ in pending_type_decls:
            if typ.name in self._streamed_types:
                continue
            _write_llvm_text(typ.get_declaration(), self._stream_file)
            self._streamed_types.add(typ.name)
        pending_type_decls.clear()

    def finish_stream_to_file(self, fsync=False):
        if self._stream_file is None:
            return
        try:
            _write_module_remainder(self.m, self._stream_file, self._streamed_types)
            if fsync:
                self._stream_file.flush()
                os.fsync(self._stream_file.fileno())
        finally:
            self.close_stream()

    def close_stream(self):
        if self._stream_file is not None:
            self._stream_file.close()
            self._stream_file = None

    def save_to_file(self, filename):
        """
        将LLVM IR模块以文本格式保存到文件。
        
        输出的是人类可读的LLVM IR（.ll格式），可以：
        - 使用clang/llc编译
        - 使用opt优化
        - 使用各种LLVM工具分析
        
        :param filename: 输出.ll文件的路径
        """
        with open(filename, 'w') as f:
            _write_module(self.m, f)


def _write_llvm_text(text: str, output):
    # 原样写出：缺终止指令的函数在提升阶段已被拒绝，这里不做任何“语法修复”
    output.write(text)
    if not text.endswith("\n"):
        output.write("\n")


def _write_module_header(module: ir.Module, output):
    output.write(f'; ModuleID = "{module.name}"\n')
    output.write(f'target triple = "{module.triple}"\n')
    output.write(f'target datalayout = "{module.data_layout}"\n\n')


def _mark_weak_declarations(module: ir.Module):
    """
    源 binary 中的弱符号（如 __gmon_start__、_ITM_deregisterTMCloneTable）可能未定义，代码会先判空再用。
    普通外部声明会让 LLVM 假定其地址非空并删除这些判断，所以弱符号的声明使用 extern_weak。
    """
    for value in module.globals.values():
        ea = getattr(value, "_source_ea", None)
        if ea is None:
            continue
        is_declaration = (value.is_declaration if isinstance(value, ir.Function)
                          else isinstance(value, ir.GlobalVariable) and value.initializer is None)
        if is_declaration and ida_name.is_weak_name(ea):
            value.linkage = "extern_weak"


def _write_module_remainder(module: ir.Module, output, skip_type_names=None):
    skip_type_names = skip_type_names or set()
    _mark_weak_declarations(module)
    for type_name, typ in module.get_identified_types().items():
        if type_name in skip_type_names:
            continue
        _write_llvm_text(typ.get_declaration(), output)

    for global_value in module.globals.values():
        if isinstance(global_value, ir.Function) and getattr(global_value, "_streamed_definition", False):
            continue
        _write_llvm_text(str(global_value), output)

    for metadata_line in module._get_metadata_lines():
        _write_llvm_text(metadata_line, output)


def _write_module(module: ir.Module, output):
    _write_module_header(module, output)
    _write_module_remainder(module, output)


# ============================================================================
# 公开接口：供其他 Python 文件调用的函数
# ============================================================================

# 同一份 lifter 源码的标识：输出文件头和 sidecar 都记录它，task1 --resume 据此判断旧输出是否可复用
LIFTER_ID = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
# 退出码：0 成功；3 没有任何函数被成功提升（不用 2，2 是 click 的参数错误）；1 异常
EXIT_OK, EXIT_ERROR, EXIT_NO_FUNCTIONS = 0, 1, 3


def _source_info() -> dict:
    info = {"bits": ptrsize, "endian": byteorder}
    with suppress(Exception):
        info["proc"] = str(ida_ida.inf_get_procname())
    return info


def _ida_version():
    with suppress(Exception):
        return str(ida_kernwin.get_kernel_version())
    return None


def _input_info(path: str) -> dict:
    info = {"path": os.path.abspath(path)}
    with suppress(OSError):
        st = os.stat(path)
        info.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return info


def _verify_llvm_file(path: str) -> tuple[bool, str]:
    """发布前用 llvm-as 解析并校验整个模块；工具缺失、超时或校验失败都不发布。"""
    tool = os.environ.get("REGRAPH_LLVM_AS", "llvm-as")
    timeout = int(os.environ.get("REGRAPH_VERIFY_TIMEOUT", "3600"))
    try:
        proc = subprocess.run([tool, "-o", os.devnull, path], stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, f"verifier {tool!r} not found"
    except subprocess.TimeoutExpired:
        return False, f"verifier timed out after {timeout}s"
    return proc.returncode == 0, (proc.stderr or "")[-2000:]


def _release_heap():
    """把已释放的 malloc 堆归还给系统（glibc），否则关闭数据库后的空闲内存仍计入本进程 RSS。"""
    with suppress(Exception):
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _current_rss_kb():
    with suppress(OSError):
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    return None


def _previous_output(out: str, sidecar: str):
    """本次运行前已存在的输出及其状态；失败重跑时原样保留，只在 sidecar 中记录其身份。"""
    if not os.path.exists(out):
        return None
    state = artifact_state.read_json(sidecar)
    valid, reason = artifact_state.check_output(state, out)
    info = {"path": out, "valid": valid, "reason": reason}
    if state:
        info.update(status=state.get("status"), lifter_id=state.get("lifter_id"),
                    sha256=(state.get("output") or {}).get("sha256"))
    return info


def run_lift(input_binary: str, output_llvm_ir: str, target_mode: str = "source", verifier=_verify_llvm_file) -> dict:
    """
    提升一个 binary 并发布 <output> 与 <output>.lift.json：
      1. 流式写 <output>.tmp.<host>.<pid> 并 fsync；
      2. 至少一个函数成功时，用 llvm-as 校验临时文件；
      3. 记录输出的 sha256/size，原子替换成 <output>，再原子写 sidecar。
    没有成功函数、校验失败或异常时不发布：已有的 <output> 原样保留（不删除），sidecar 记录本次失败和
    旧产物身份，下游据 status/sha256 拒绝把旧产物当成本次结果。在两次替换之间被杀时，新 <output> 与
    旧 sidecar 的 sha256 不一致，同样被拒绝。返回 sidecar 内容。
    """
    _reset_lift_state()
    started = time.time()
    out = os.path.abspath(output_llvm_ir)
    sidecar = out + artifact_state.LIFT_SUFFIX
    artifact_state.cleanup_stale_tmp(out)
    artifact_state.cleanup_stale_tmp(sidecar)
    tmp = artifact_state.tmp_path_for(out)
    result = {
        "schema": artifact_state.SCHEMA,
        "stage": "lift",
        "status": "error",
        "lifter_id": LIFTER_ID,
        "ida_version": None,
        "ida_install_id": artifact_state.ida_install_id(os.environ.get("IDADIR")),
        "input": _input_info(input_binary),
        "output_path": out,
        "target_mode": target_mode,
        "previous_output": _previous_output(out, sidecar),
    }
    bin2llvm = None
    database_open = False
    try:
        idapro.open_database(input_binary, True)
        database_open = True
        ida_auto.auto_wait()
        _init_target_state()
        result["ida_version"] = _ida_version()
        result["source"] = _source_info()
        bin2llvm = BIN2LLVMController(target_mode=target_mode)
        result["target"] = {"triple": str(bin2llvm.m.triple), "datalayout": str(bin2llvm.m.data_layout)}
        result["memory_contract"] = bin2llvm.memory_contract
        bin2llvm.initialize()
        source = result["source"]
        header = [
            f"ida2llvm lifter={LIFTER_ID} ida={result['ida_version']} target_mode={target_mode} "
            f"memory_contract={bin2llvm.memory_contract}",
            f"source proc={source.get('proc')} bits={source['bits']} endian={source['endian']}",
        ]
        bin2llvm.begin_stream_to_file(tmp, header)
        bin2llvm.insertAllFunctions()
        bin2llvm.finish_stream_to_file(fsync=True)
        # 校验前释放 IDA 数据库和 lifter 的模块/缓存：llvm-as 对大模块的峰值内存与 lifter 相当，
        # 避免两者同时驻留（逐函数记录先保存下来）
        snapshot = (dict(lift_stats), Counter(unsupported_reasons), list(function_records), list(unresolved_data_items))
        bin2llvm = None
        idapro.close_database(False)
        database_open = False
        _reset_lift_state()
        gc.collect()
        _release_heap()
        result["rss_before_verify_kb"] = _current_rss_kb()
        lift_stats.update(snapshot[0])
        unsupported_reasons.update(snapshot[1])
        function_records.extend(snapshot[2])
        unresolved_data_items.extend(snapshot[3])
        if lift_stats["defined"] == 0:
            result["status"] = "no_functions_lifted"
        else:
            verify_started = time.time()
            ok, message = verifier(tmp)
            result["verify_s"] = round(time.time() - verify_started, 3)
            if not ok:
                result["status"] = "verify_failed"
                result["error"] = message
            else:
                result["output"] = {"path": out, "size": os.path.getsize(tmp),
                                    "sha256": artifact_state.sha256_file(tmp)}
                result["status"] = "ok"
                os.replace(tmp, out)
                artifact_state.fsync_dir(os.path.dirname(out))
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"[:1000]
        logging.error(f"Failed to lift binary: {e}")
        logging.error(traceback.format_exc())
    finally:
        if bin2llvm is not None:
            with suppress(Exception):
                bin2llvm.close_stream()
        with suppress(FileNotFoundError):
            os.remove(tmp)
        if database_open:
            with suppress(Exception):
                # 不保存：lift 不应修改输入数据库，且保证同一 .i64 重复运行结果一致
                idapro.close_database(False)
        approx_reasons = Counter(tag for rec in function_records for tag in rec.get("approx", ()))
        result.update(
            elapsed_s=round(time.time() - started, 3),
            counts=dict(lift_stats),
            unsupported_reasons=dict(sorted(unsupported_reasons.items())),
            approx_reasons=dict(sorted(approx_reasons.items())),
            unresolved_data=list(unresolved_data_items),
            functions=list(function_records),
        )
        try:
            artifact_state.write_json_atomic(sidecar, result)
        except OSError as e:
            # 新 .ll 已发布但 sidecar 没写成：旧 sidecar（若有）的 sha256 与之不符，下游会拒绝
            logging.error(f"Failed to write {sidecar}: {e}")
            result["status"] = "error"
    return result


def _exit_code(result: dict) -> int:
    return {"ok": EXIT_OK, "no_functions_lifted": EXIT_NO_FUNCTIONS}.get(result["status"], EXIT_ERROR)


def lift_binary_to_llvm(
    input_binary: str,
    output_llvm_ir: str,
    target_mode: str = "source",
    verbose: bool = False
) -> bool:
    """
    将二进制文件转换为 LLVM IR（供其他 Python 文件调用）。

    参数:
        input_binary (str): 输入的二进制 / IDA 数据库路径
        output_llvm_ir (str): 输出的 LLVM IR 文件路径 (.ll)；同目录写 <output>.lift.json 状态文件
        target_mode (str): "source"（默认，源 binary 的 triple/data layout）或 "host"（旧行为兼容模式，
            输出宿主 layout 而类型大小按源指针宽度计算，不做内存语义转换）
        verbose (bool): 是否打印详细日志（默认 False）

    返回值:
        bool: 至少一个函数成功提升且输出已原子发布时返回 True；否则 False（不抛异常）
    """
    if target_mode not in TARGET_MODES:
        logging.error(f"Invalid target_mode '{target_mode}'. Must be one of {TARGET_MODES}")
        return False

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    result = run_lift(input_binary, output_llvm_ir, target_mode)
    if result["status"] == "ok":
        logging.info(f"Successfully lifted binary to LLVM IR: {output_llvm_ir}")
    return result["status"] == "ok"


app = typer.Typer(
    add_completion=False,
    help="IDA2LLVM - Convert binary to LLVM IR using IDA Pro 9+ idalib",
)


@app.command()
def main(
    file: str = typer.Option(
        ..., "-f", "--file", help="Binary file to be analyzed"
    ),
    output: str = typer.Option(
        ..., "-o", "--output", help="Output file for LLVM IR (.ll)"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging"
    ),
    target: str = typer.Option(
        "source", "--target",
        help="source (default): source binary triple/data layout; host: legacy host-layout compatibility "
             "mode without memory-semantics conversion",
    ),
):
    """
    作为独立应用程序运行（使用 IDA Pro 9+ idalib）。
    """
    if target == "ida":
        target = "source"
    if target not in TARGET_MODES:
        raise typer.BadParameter(f"must be one of: {', '.join(TARGET_MODES)}")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    result = run_lift(file, output, target)
    print(
        f"lift stats: status={result['status']} defined={lift_stats['defined']} failed={lift_stats['failed']} "
        f"approx={lift_stats['approx']} unresolved_data={lift_stats['unresolved_data']} "
        f"unsupported={','.join(f'{k}:{v}' for k, v in sorted(unsupported_reasons.items())) or 'none'} "
        f"no_decompile={lift_stats['no_decompile']}",
        file=sys.stderr,
    )
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
    raise typer.Exit(_exit_code(result))


if __name__ == "__main__":
    app()
