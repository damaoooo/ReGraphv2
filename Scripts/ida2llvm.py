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
import struct
import numpy as np
import traceback
import sys
import typer
import re
import io
import zlib
import llvmlite.binding as llvm
from llvmlite import ir

from contextlib import suppress

# Log 输出到文件 ida2llvm.log
logging.basicConfig(
    filename='ida2llvm.log',
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
lift_stats = {"defined": 0, "failed": 0, "no_decompile": 0}
_CACHE_MISS = object()

# 线程局部存储段大小（Windows FS段标准大小）
FS_SEGMENT_SIZE = 0x10000  # 64KB

# 浮点数提取失败时的默认值
DEFAULT_FLOAT_VALUE = 1.0
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
LIFTED_INTRINSIC_NAMES = {
    "sadd_overflow",
    "__OFSUB__",
    "_mm_cvtsi128_si32",
    "_BitScanReverse",
    "__FYL2X__",
    "__FYL2P__",
    "fabs",
    "fabsf",
    "fabsl",
    "memcpy",
    "_byteswap_ulong",
    "_byteswap_uint64",
    "memset",
    "abs64",
    "__PAIR64__",
    "__PAIR128__",
    "__PAIR32__",
    "__PAIR16__",
    "_BitScanReverse64",
    "_BitScanForward64",
    "__halt",
    "is_mul_ok",
    "va_start",
    "va_arg",
    "va_end",
    "_QWORD",
    "__ROL8__",
    "__ROL4__",
    "__ROR4__",
    "__ROR8__",
}
LIFTED_INTRINSIC_PREFIXES = (
    "__readfs",
    "__writefs",
    "sys_",
    "_InterlockedCompareExchange",
    "_InterlockedExchange",
)
LLVM_TERMINATOR_PREFIXES = (
    "br ",
    "ret ",
    "switch ",
    "indirectbr ",
    "invoke ",
    "resume ",
    "unreachable",
    "cleanupret ",
    "catchret ",
    "catchswitch ",
)
LLVM_DEFINE_RE = re.compile(r"define\s+(.+?)\s+@")
LLVM_LABEL_RE = re.compile(r"^[A-Za-z$._-][A-Za-z0-9$._-]*:$")
LLVM_INT_RE = re.compile(r"i\d+$")


def _reset_lift_state():
    """清理一次 binary lift 过程中的全局缓存。"""
    global target_data
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
    target_data = None


def _init_target_state():
    """在 open_database 之后读取当前 binary 的指针宽度和字节序。"""
    global ptrsize, is_big_endian, byteorder, target_data
    ptrsize = 64 if ida_ida.inf_is_64bit() else 32
    is_big_endian = bool(ida_ida.inf_is_be())
    byteorder = "big" if is_big_endian else "little"
    target_data = None
    llvm_type_cache.clear()
    abi_size_cache.clear()
    type_size_cache.clear()


def _get_target_data():
    """复用 llvmlite target data，避免热路径重复构造。"""
    global target_data
    if target_data is None:
        layout = "E" if is_big_endian else "e"
        if ptrsize == 32:
            layout += "-p:32:32"
        target_data = llvm.create_target_data(layout)
    return target_data


def _get_abi_size(typ: ir.Type) -> int:
    key = id(typ)
    entry = abi_size_cache.get(key)
    if entry is None:
        # 同时持有 typ，防止对象被回收后 id 被复用导致取到错误的 size。
        entry = (typ, typ.get_abi_size(_get_target_data()))
        abi_size_cache[key] = entry
    return entry[1]


def _type_size(typ: ir.Type) -> int:
    """按被分析 binary 的指针宽度计算类型字节大小（结构体都是 packed 的）。"""
    if isinstance(typ, ir.IntType):
        return (typ.width + 7) // 8
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


def _get_register_storage(mop: ida_hexrays.mop_t, blk: ir.Block, builder: ir.IRBuilder):
    """
    为 Hex-Rays 的寄存器操作数分配一个稳定的栈槽。

    这里不试图完整模拟物理寄存器别名关系，只提供一个可读可写的 SSA 外存储，
    避免寄存器型操作数直接返回 None 导致后续算术/位运算崩溃。
    """
    func = blk.parent
    reg_name = f"reg_{_sanitize_symbol_name(mop.dstr())}"
    reg_width = ptrsize if mop.size <= 0 else mop.size * 8

    if reg_name not in func.lvars:
        reg_type = ir.IntType(reg_width)
        with builder.goto_entry_block():
            reg_ptr = builder.alloca(reg_type, name=reg_name)
            builder.store(ir.Constant(reg_type, 0), reg_ptr)
        func.lvars[reg_name] = reg_ptr

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
        # 防止创建过大的整数类型（限制为128位）
        if size_bits > 128 or size_bits <= 0:
            # logging.warning(f"Invalid type size {tif.get_size()} bytes, using pointer size instead")
            return ir.IntType(ptrsize)
        return ir.IntType(size_bits)
        
    else:
        if width != -1:
            # 防止创建过大的数组（限制为1MB）
            if width > 1024 * 1024 or width <= 0:
                # logging.warning(f"Invalid array width {width} bytes, using pointer type instead")
                return ir.IntType(ptrsize)
            return ir.ArrayType(ir.IntType(8), width)
        else:
            return ir.IntType(ptrsize)

def _sized_member_type(llvm_type: ir.Type, size: int) -> ir.Type:
    """成员的 LLVM 类型大小和 IDA 不一致时，用同样字节数的整数/字节数组替代。"""
    if _type_size(llvm_type) == size:
        return llvm_type
    if size in (1, 2, 4, 8, 16):
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

    context = ir.context.global_context
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
    if _get_func_cached(ea) != None and _get_segtype_cached(ea) & ida_segment.SEG_XTRN:
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
                ptr = builder.alloca(ir.IntType(8).as_pointer(), name = "ArgList")
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
        with suppress(AttributeError):
            del res.block_by_ea
        with suppress(AttributeError):
            del res.lvars
        for blk in res.blocks:
            with suppress(AttributeError):
                del blk.next_blk
        return res

    return _constant_from_bytes(typ, _read_bytes(ea, _type_size(typ)), 0)


def _read_bytes(ea: int, size: int) -> bytes:
    """一次性读取一段数据；不可读的部分补 0。"""
    if size <= 0:
        return b""
    data = ida_bytes.get_bytes(ea, size) or b""
    if len(data) < size:
        data += b"\x00" * (size - len(data))
    return data


def _constant_from_bytes(typ: ir.Type, data: bytes, off: int):
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
        return ir.Constant(typ, None)
    if isinstance(typ, ir.ArrayType):
        element = typ.element
        sub_size = _type_size(element)
        if isinstance(element, ir.IntType) and element.width % 8 == 0:
            nbytes = element.width // 8
            values = [
                int.from_bytes(data[pos:pos + nbytes], byteorder)
                for pos in range(off, off + sub_size * typ.count, sub_size)
            ]
            return ir.Constant(typ, values)
        if isinstance(element, ir.PointerType):
            return ir.Constant(typ, [ir.Constant(element, None)] * typ.count)
        return ir.Constant.literal_array([
            _constant_from_bytes(element, data, pos)
            for pos in range(off, off + sub_size * typ.count, sub_size)
        ])
    if isinstance(typ, (ir.LiteralStructType, ir.IdentifiedStructType)):
        elements = []
        for el in typ.elements:
            elements.append(_constant_from_bytes(el, data, off))
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


def _is_lifted_intrinsic_name(func_name: str) -> bool:
    return func_name in LIFTED_INTRINSIC_NAMES or func_name.startswith(LIFTED_INTRINSIC_PREFIXES)


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
    with suppress(KeyError):
        existing = module.get_global(func_name)
        if _is_lifted_intrinsic_name(func_name):
            return existing

    if func_name == "sadd_overflow":
        typ = ir.LiteralStructType((ir.IntType(64), ir.IntType(1)))
        return  module.declare_intrinsic('sadd_overflow', fnty=ir.FunctionType(typ.as_pointer(), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "__OFSUB__":
        return  module.declare_intrinsic('__OFSUB__', fnty=ir.FunctionType(ir.IntType(1), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "_mm_cvtsi128_si32":
        return  module.declare_intrinsic('_mm_cvtsi128_si32', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(128)]))

    elif func_name == "_BitScanReverse":
        return  module.declare_intrinsic('_BitScanReverse', fnty=ir.FunctionType(i8ptr, [ir.IntType(32), ir.IntType(32)]))

    elif func_name == "__FYL2X__":
        return  module.declare_intrinsic('__FYL2X__', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]))

    elif func_name == "__FYL2P__":
        return  module.declare_intrinsic('__FYL2P__', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]))
       
    elif func_name == "fabs":
        return  module.declare_intrinsic('fabs', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]))

    elif func_name == "fabsf":
        return  module.declare_intrinsic('fabsf', fnty=ir.FunctionType(ir.FloatType(), [ir.FloatType()]))

    elif func_name == "fabsl":
        return  module.declare_intrinsic('fabs', fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()]))

    elif func_name == "memcpy":
        return  module.declare_intrinsic('memcpy', fnty=ir.FunctionType(i8ptr, [i8ptr, i8ptr, ir.IntType(ptrsize)]))

    elif func_name == "_byteswap_ulong":
        return  module.declare_intrinsic('_byteswap_ulong', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(32)]))

    elif func_name == "_byteswap_uint64":
        return  module.declare_intrinsic('_byteswap_uint64', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)]))

    elif func_name == "memset":
        return  module.declare_intrinsic('memset', fnty=ir.FunctionType(i8ptr, [i8ptr, ir.IntType(32), ir.IntType(ptrsize)]))

    elif func_name == "abs64":
        return  module.declare_intrinsic('abs64', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)]))

    elif func_name == "__PAIR64__":
        return  module.declare_intrinsic('__PAIR64__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(32), ir.IntType(32)]))

    elif func_name == "__PAIR128__":
        return  module.declare_intrinsic('__PAIR128__', fnty=ir.FunctionType(ir.IntType(128), [ir.IntType(64), ir.IntType(64)]))

    elif func_name == "__PAIR32__":
        return  module.declare_intrinsic('__PAIR32__', fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(16), ir.IntType(16)]))

    elif func_name == "__PAIR16__":
        return  module.declare_intrinsic('__PAIR16__', fnty=ir.FunctionType(ir.IntType(16), [ir.IntType(8), ir.IntType(8)]))

    elif func_name == "_BitScanReverse64":
        return  module.declare_intrinsic('_BitScanReverse64', fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]))

    elif func_name == "_BitScanForward64":
        return  module.declare_intrinsic('_BitScanForward64', fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]))


    elif func_name == "__halt":
        fty = ir.FunctionType(ir.VoidType(), [])
        f = ir.Function(module, fty, "__halt")
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        builder.asm(fty, "hlt", "", (), True)
        builder.ret_void()
        return f

    elif func_name == "is_mul_ok":
        is_mul_ok = module.declare_intrinsic('is_mul_ok', fnty=ir.FunctionType(ir.IntType(8), [ir.IntType(64), ir.IntType(64)]))
        return is_mul_ok

    elif func_name == "va_start":
        va_start = module.declare_intrinsic('va_start', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
        return va_start

    elif func_name == "va_arg":
        va_arg = module.declare_intrinsic('va_arg', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer()]))
        return va_arg

    elif func_name == "va_end":
        va_arg = module.declare_intrinsic('va_end', fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]))
        return va_arg

    elif func_name == "_QWORD":
        fQWORD = module.declare_intrinsic('IDA_QWORD', fnty=ir.FunctionType(ir.IntType(8).as_pointer(), []))
        return fQWORD

    elif func_name == "__ROL8__":
        f = module.declare_intrinsic('__ROL8__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f

    elif func_name == "__ROL4__":
        f = module.declare_intrinsic('__ROL4__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f
        
    elif func_name == "__ROR4__":
        f = module.declare_intrinsic('__ROR4__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
        return f

    elif func_name == "__ROR8__":
        f = module.declare_intrinsic('__ROR8__', fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]))
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

    elif func_name.startswith("sys_"):
        fty = ir.FunctionType(ir.IntType(64), [], var_arg=True)
        f = ir.Function(module, fty, func_name)
        return f

    elif func_name.startswith("_InterlockedCompareExchange") or func_name.startswith("_InterlockedExchange"):
        fty = ir.FunctionType(ir.IntType(64), [], var_arg=True)
        f = ir.Function(module, fty, func_name)
        return f

    else:
        raise NotImplementedError(f"不支持的内置函数: {func_name}")  

def _retype_function(func: ir.Function, ftype: ir.FunctionType):
    """把一个尚无函数体的 ir.Function 改成新的函数类型。"""
    func.ftype = ftype
    func.type = ftype.as_pointer()
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
    with suppress(NotImplementedError):
        return lift_intrinsic_function(module, func_name)

    func_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, func_name)
    if func_ea != ida_idaapi.BADADDR and _get_segtype_cached(func_ea) & ida_segment.SEG_XTRN:
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
            if knowntyp is not None:
                d = typecast(d, knowntyp, builder)
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
        reg_ptr = _get_register_storage(mop, blk, builder)
        reg_ptr = _cast_slot(reg_ptr, mop.size if mop.size > 0 else -1, knowntyp, builder)
        return reg_ptr if dest else builder.load(reg_ptr)
    elif t == ida_hexrays.mop_S: # 栈变量
        name = "stack"
        func = blk.parent
        if name not in func.lvars:
            with builder.goto_entry_block():
                func.lvars[name] = builder.alloca(ir.IntType(ptrsize), name = name)
        llvm_arg = get_offset_to(builder, func.lvars[name], mop.s.off)
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
            if tif.is_func() or tif.is_funcptr() or is_func_start:
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
            g = ir.GlobalVariable(module, g_cmt.type, name = name)
            g.initializer = g_cmt

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
        with suppress(NotImplementedError):
            return lift_intrinsic_function(blk.parent.parent, mop.helper)
        return None
    elif t == ida_hexrays.mop_str: # 字符串常量
        str_csnt = mop.cstr
        strType = ir.ArrayType(ir.IntType(8), len(str_csnt))
        g = ir.GlobalVariable(blk.parent.parent, strType, name=f"cstr_{len(blk.parent.parent.globals)}")
        g.initializer = ir.Constant(strType, bytearray(str_csnt.encode("utf-8")))
        g.linkage = "private"
        g.global_constant = True
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
        # IDA获取浮点数值在某些情况下可能会崩溃
        try:
            fp = mop.fpc.fnum.float
        except (AttributeError, ValueError, SystemError) as e:
            logging.debug(f"Failed to extract float value: {e}, using default {DEFAULT_FLOAT_VALUE}")
            fp = DEFAULT_FLOAT_VALUE
        typ = float_type(mop.size)
        return ir.Constant(typ, fp)
    elif t == ida_hexrays.mop_p:
        f = lift_intrinsic_function(blk.parent.parent, f"__PAIR{mop.size*8}__")
        l = lift_mop(mop.pair.hop, blk, builder, dest)
        r = lift_mop(mop.pair.lop, blk, builder, dest)
        l = typecast(l, ir.IntType(mop.size*4), builder)
        r = typecast(r, ir.IntType(mop.size*4), builder)
        return builder.call(f, (l, r))
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
    argtypes = []
    for arg in ftif.args:
        argtypes.append(lift_tif(arg.type))

    rettype = lift_tif(ftif.return_type)
    if isinstance(rettype, ir.VoidType):
        rettype = i8ptr
    return module.declare_intrinsic(func_name, fnty=ir.FunctionType(rettype, argtypes))

def float_type(size: int) -> ir.Type:
    """
    根据给定的大小返回适当的LLVM浮点类型。
    
    注意：llvmlite不支持long double，所以大小 > 8 的默认为double。
    
    :param size: 字节大小（4为float，8为double）
    :return: LLVM FloatType 或 DoubleType
    """
    return ir.FloatType() if size == 4 else ir.DoubleType()

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


def _branch_target(builder, ida_insn, d, next_blk):
    target_blk = _resolve_block_target(builder.block.parent, ida_insn.d)
    if target_blk is not None:
        return target_blk
    if not isinstance(d, ir.Block) and next_blk is not None:
        return next_blk
    return d


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
_SKIP_L_OPS = {ida_hexrays.m_ldx, ida_hexrays.m_icall, ida_hexrays.m_nop, ida_hexrays.m_und, ida_hexrays.m_ext}
_SKIP_R_OPS = {ida_hexrays.m_stx, ida_hexrays.m_ijmp, ida_hexrays.m_nop, ida_hexrays.m_und, ida_hexrays.m_ext}
# 这些指令的 d 操作数是值（地址/调用信息），而不是写入目标
_D_AS_VALUE_OPS = {ida_hexrays.m_call, ida_hexrays.m_icall, ida_hexrays.m_stx}
_SKIP_D_OPS = {ida_hexrays.m_nop, ida_hexrays.m_und, ida_hexrays.m_ext}

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
_NO_OP_OPS = {ida_hexrays.m_nop, ida_hexrays.m_ijmp, ida_hexrays.m_push, ida_hexrays.m_pop, ida_hexrays.m_und, ida_hexrays.m_ext}


def _lift_call(ida_insn, l, d, blk, builder, result_type):
    rets, args = d
    if not isinstance(l.type, ir.PointerType) or not isinstance(l.type.pointee, ir.FunctionType):
        call_ret_type = result_type if result_type is not None else i8ptr
        new_func_type = ir.FunctionType(call_ret_type, [arg.type for arg in args], var_arg=False).as_pointer()
        l = typecast(l, new_func_type, builder)
    else:
        ftype = l.type.pointee
        if (
            result_type is not None
            and not isinstance(result_type, ir.VoidType)
            and isinstance(ftype.return_type, ir.VoidType)
        ):
            # 被调函数声明为 void 但调用点使用了返回值：只在调用点转换类型，不改函数定义
            ftype = ir.FunctionType(result_type, ftype.args, var_arg=ftype.var_arg)
            l = typecast(l, ftype.as_pointer(), builder)

        params = ftype.args
        for i, param_type in enumerate(params):
            if i >= len(args):
                # 调用点缺少的实参
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
    for dst in rets:
        _store_as(ret, dst, blk, builder)
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

    # 为未知的内置函数创建声明
    if l is None and op == ida_hexrays.m_call and l_mop.t == ida_hexrays.mop_h:
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
        typ = float_type(d_mop.size) if ida_insn.is_fpinsn() else ir.IntType(d_mop.size * 8)
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
        if next_blk is not None:
            return builder.branch(next_blk)
        return
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
        blocks = blk.parent.basic_blocks
        default = r["default"] if "default" in r else r[next(iter(r))]
        switch = builder.switch(l, blocks[default])
        for value, target in r.items():
            if isinstance(value, int):
                switch.add_case(value, blocks[target])
        return switch
    elif op == ida_hexrays.m_icall:  # icall  {l=sel, r=off} d    间接调用
        rets, args = d
        ret_type = result_type if result_type is not None else i8ptr
        ftype = ir.FunctionType(ret_type, [arg.type for arg in args])
        f = typecast(r, ftype.as_pointer(), builder)
        ret = builder.call(f, args)
        for dst in rets:
            _store_as(ret, dst, blk, builder)
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
    def __init__(self, target_mode: str = "host"):
        """使用空的LLVM模块初始化控制器。"""
        self.m = ir.Module()
        self._name_cache = {}
        self._func_cache = {}
        self._function_eas = []
        self._stream_file = None
        self._streamed_functions = set()
        self._streamed_types = set()
        if target_mode == "host":
            self._set_module_target_from_host()
        else:
            self._set_module_target_from_ida()

    def _get_name(self, ea):
        return _get_name_cached(ea)

    def _get_func(self, ea):
        return _get_func_cached(ea)

    def _set_module_target_from_host(self):
        """Sets module target triple and data layout based on host LLVM."""
        try:
            # Try to initialize LLVM targets if available
            try:
                # These calls may raise deprecation warnings but are necessary for proper initialization
                if hasattr(llvm, 'initialize_all_targets'):
                    llvm.initialize_all_targets()
                if hasattr(llvm, 'initialize_all_asmprinters'):
                    llvm.initialize_all_asmprinters()
            except:
                # Initialization may fail in some environments, continue anyway
                pass
            
            triple = llvm.get_default_triple()
            target = llvm.Target.from_triple(triple)
            target_machine = target.create_target_machine()
            self.m.triple = triple
            self.m.data_layout = str(target_machine.target_data)
        except Exception as exc:
            # If host mode fails, fall back to IDA mode
            logging.debug(f"Host mode target setup failed: {exc}, falling back to IDA mode")
            self._set_module_target_from_ida()

    def _set_module_target_from_ida(self):
        """Sets module target triple based on IDA's analysis information."""
        try:
            proc = ida_ida.inf_get_procname()
            is_64 = ida_ida.inf_is_64bit()
            arch = "unknown"

            if proc == "metapc":
                arch = "x86_64" if is_64 else "i386"
            elif proc in ("arm", "ARM"):
                arch = "aarch64" if is_64 else "arm"
            elif proc in ("aarch64", "arm64"):
                arch = "aarch64"
            elif proc == "mips":
                arch = "mips64" if is_64 else "mips"
            elif proc in ("ppc", "ppc64"):
                arch = "powerpc64" if is_64 else "powerpc"
            elif proc == "riscv":
                arch = "riscv64" if is_64 else "riscv32"

            os_name = "unknown"
            with suppress(Exception):
                ostype = ida_ida.inf_get_ostype()
                if hasattr(ida_ida, "OSTYPE_WIN") and ostype == ida_ida.OSTYPE_WIN:
                    os_name = "windows"
                elif hasattr(ida_ida, "OSTYPE_LINUX") and ostype == ida_ida.OSTYPE_LINUX:
                    os_name = "linux"
                elif hasattr(ida_ida, "OSTYPE_MACOS") and ostype == ida_ida.OSTYPE_MACOS:
                    os_name = "darwin"

            self.m.triple = f"{arch}-unknown-{os_name}"
            self.m.data_layout = ""
        except Exception as exc:
            logging.warning(f"Failed to set module target from IDA: {exc}")

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
        for attr in ("lvars", "block_by_ea"):
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
        try:
            pfunc = ptext.get(ea)
            if pfunc is None or getattr(pfunc, "type", None) is None:
                with suppress(KeyError):
                    del ptext[ea]
                pfunc = self._decompile_function(ea)
            if pfunc is None or getattr(pfunc, "type", None) is None:
                lift_stats["no_decompile"] += 1
                return

            typ = pfunc.type
            function_signature_cache[ea] = typ
            func_name = self._get_name(ea)
            lifted = lift_function(self.m, func_name, False, ea, typ)
            if isinstance(lifted, ir.Function) and not lifted.is_declaration:
                lift_stats["defined"] += 1
            self._stream_function_if_enabled(lifted)
        except Exception as e:
            lift_stats["failed"] += 1
            logging.debug(f"在 {hex(ea)} 处提升函数失败: {e}")
            self._drop_partial_definition(ea)
        finally:
            with suppress(KeyError):
                del ptext[ea]
            self._trim_ptext()
            # 释放 Hex-Rays 内部缓存的 cfunc，不影响后续反编译结果，显著降低峰值内存
            ida_hexrays.mark_cfunc_dirty(ea, False)

    def create_global(self, ea, width, str_dict):
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
            return

        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, ea):
            if not ENABLE_DATA_TYPE_GUESS:
                return
            ida_typeinf.guess_tinfo(tif, ea)

        # 如果数据项是外部函数，创建声明
        if tif.is_func() or tif.is_funcptr():
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
            or _get_segtype_cached(ea) & ida_segment.SEG_XTRN):
                # 返回函数声明
                g = lift_function(self.m, name, True, ea, tif)

        # 其他数据项，可能是 int/float/数组/结构体
        else:
            typ = lift_tif(tif, width)
            g_cmt = lift_from_address(self.m, ea, typ)
            g = ir.GlobalVariable(self.m, typ, name = name)
            g.initializer = g_cmt

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

        # 步骤3：为非执行段中的所有数据项创建全局变量
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
                self.create_global(head, end - head, str_dict)

    def begin_stream_to_file(self, filename):
        self._stream_file = open(filename, 'w')
        _write_module_header(self.m, self._stream_file)

    def _stream_function_if_enabled(self, func):
        if self._stream_file is None:
            return
        if not isinstance(func, ir.Function) or func.is_declaration:
            return
        if func.name in self._streamed_functions:
            return

        self._stream_new_type_declarations()
        if all(_block_ends_with_terminator(blk) for blk in func.blocks):
            # 每个块都已正确终止时，文本 sanitizer 不会做任何修改，直接写出
            self._stream_file.write(str(func))
        else:
            _write_sanitized_llvm_ir(str(func), self._stream_file)
        self._stream_file.write("\n")
        self._streamed_functions.add(func.name)
        func.blocks.clear()
        func._streamed_definition = True

    def _stream_new_type_declarations(self):
        if self._stream_file is None:
            return
        for type_name, typ in self.m.get_identified_types().items():
            if type_name in self._streamed_types:
                continue
            _write_sanitized_llvm_ir(typ.get_declaration(), self._stream_file)
            self._streamed_types.add(type_name)

    def finish_stream_to_file(self):
        if self._stream_file is None:
            return
        try:
            _write_module_remainder(self.m, self._stream_file, self._streamed_types)
        finally:
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
            _write_module_sanitized(self.m, f)


def _is_llvm_terminator(line: str) -> bool:
    return line.strip().startswith(LLVM_TERMINATOR_PREFIXES)


def _is_llvm_label(line: str) -> bool:
    stripped = line.strip()
    if not stripped.endswith(":"):
        return False
    return stripped.startswith('"') or LLVM_LABEL_RE.match(stripped) is not None


def _llvm_label_ref(label_line: str) -> str:
    label = label_line.strip()[:-1]
    return f"%{label}"


def _default_ret_for_type(ret_type: str) -> str:
    ret_type = ret_type.strip()
    if ret_type == "void":
        return "  ret void"
    if ret_type in ("float", "double"):
        return f"  ret {ret_type} 0.000000e+00"
    if ret_type.endswith("*"):
        return f"  ret {ret_type} null"
    if LLVM_INT_RE.match(ret_type):
        return f"  ret {ret_type} 0"
    return f"  ret {ret_type} zeroinitializer"


def _sanitize_llvm_ir(text: str) -> str:
    out = io.StringIO()
    _write_sanitized_llvm_ir(text, out)
    return out.getvalue()


def _write_module_header(module: ir.Module, output):
    output.write(f'; ModuleID = "{module.name}"\n')
    output.write(f'target triple = "{module.triple}"\n')
    output.write(f'target datalayout = "{module.data_layout}"\n\n')


def _write_module_remainder(module: ir.Module, output, skip_type_names=None):
    skip_type_names = skip_type_names or set()
    for type_name, typ in module.get_identified_types().items():
        if type_name in skip_type_names:
            continue
        _write_sanitized_llvm_ir(typ.get_declaration(), output)

    for global_value in module.globals.values():
        if isinstance(global_value, ir.Function) and getattr(global_value, "_streamed_definition", False):
            continue
        _write_sanitized_llvm_ir(str(global_value), output)

    for metadata_line in module._get_metadata_lines():
        _write_sanitized_llvm_ir(metadata_line, output)


def _write_module_sanitized(module: ir.Module, output):
    _write_module_header(module, output)
    _write_module_remainder(module, output)


def _write_sanitized_llvm_ir(text: str, output):
    in_func = False
    ret_type = None
    prev_nonempty = None

    for raw_line in io.StringIO(text):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        define_match = LLVM_DEFINE_RE.match(stripped)
        if define_match is not None:
            in_func = True
            ret_type = define_match.group(1)
            prev_nonempty = stripped
            output.write(line)
            output.write("\n")
            continue

        if in_func and _is_llvm_label(line):
            if (
                prev_nonempty
                and not prev_nonempty.endswith("{")
                and not _is_llvm_terminator(prev_nonempty)
            ):
                output.write(f"  br label {_llvm_label_ref(line)}\n")
            output.write(line)
            output.write("\n")
            prev_nonempty = stripped
            continue

        if in_func and stripped == "}":
            if prev_nonempty and not _is_llvm_terminator(prev_nonempty):
                output.write(_default_ret_for_type(ret_type or "void"))
                output.write("\n")
            output.write(line)
            output.write("\n")
            in_func = False
            ret_type = None
            prev_nonempty = None
            continue

        output.write(line)
        output.write("\n")
        if in_func and stripped:
            prev_nonempty = stripped


# ============================================================================
# 公开接口：供其他 Python 文件调用的函数
# ============================================================================

def lift_binary_to_llvm(
    input_binary: str,
    output_llvm_ir: str,
    target_mode: str = "host",
    verbose: bool = False
) -> bool:
    """
    将二进制文件转换为 LLVM IR。
    
    这是一个简洁的函数接口，允许其他 Python 文件像调用普通函数一样
    调用 IDA Lift 功能。
    
    参数:
        input_binary (str): 输入的二进制文件路径
        output_llvm_ir (str): 输出的 LLVM IR 文件路径 (.ll)
        target_mode (str): 目标架构设置方式，可选值：
            - "host": 使用当前主机的目标三元组（默认）
            - "ida": 使用 IDA Pro 检测到的架构
        verbose (bool): 是否打印详细日志（默认 False）
    
    返回值:
        bool: 成功返回 True，失败返回 False
    
    异常处理:
        所有异常都被捕获，函数不会抛出异常，而是返回 False
    
    使用示例:
        >>> from Scripts.ida2llvm import lift_binary_to_llvm
        >>> success = lift_binary_to_llvm(
        ...     input_binary="/path/to/binary",
        ...     output_llvm_ir="/path/to/output.ll"
        ... )
        >>> if success:
        ...     print("转换成功！")
        ... else:
        ...     print("转换失败！")
    """
    if target_mode not in ("host", "ida"):
        logging.error(f"Invalid target_mode '{target_mode}'. Must be 'host' or 'ida'")
        return False
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Clear global caches to avoid stale entries across multiple binaries.
        _reset_lift_state()

        # 打开 IDA 数据库
        idapro.open_database(input_binary, True)
        ida_auto.auto_wait()
        _init_target_state()
        
        # 创建控制器并执行提升
        bin2llvm = BIN2LLVMController(target_mode=target_mode)
        bin2llvm.initialize()
        bin2llvm.begin_stream_to_file(output_llvm_ir)
        bin2llvm.insertAllFunctions()
        bin2llvm.finish_stream_to_file()
        
        logging.info(f"Successfully lifted binary to LLVM IR: {output_llvm_ir}")
        return True
        
    except Exception as e:

        logging.error(f"Failed to lift binary: {e}", exc_info=verbose)
        logging.error("".join(traceback.format_exc()))
        return False
        
    finally:
        # 关闭 IDA 数据库
        try:
            # 不保存：lift 不应修改输入数据库，且保证同一 .i64 重复运行结果一致
            idapro.close_database(False)
        except:
            pass


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
        "host", "--target", help="Target triple source: host (default) or ida"
    ),
):
    """
    作为独立应用程序运行（使用 IDA Pro 9+ idalib）。
    """
    if target not in ("host", "ida"):
        raise typer.BadParameter("must be one of: host, ida")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        _reset_lift_state()

        idapro.open_database(file, True)
        ida_auto.auto_wait()
        _init_target_state()
        bin2llvm = BIN2LLVMController(target_mode=target)
        bin2llvm.initialize()
        bin2llvm.begin_stream_to_file(output)
        bin2llvm.insertAllFunctions()
        bin2llvm.finish_stream_to_file()
        print(
            f"lift stats: defined={lift_stats['defined']} failed={lift_stats['failed']} "
            f"no_decompile={lift_stats['no_decompile']}",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        idapro.close_database(False)


if __name__ == "__main__":
    app()
