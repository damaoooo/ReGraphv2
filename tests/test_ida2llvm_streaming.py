"""Streaming must release function bodies without breaking later references."""
import gc
import io
import types
import weakref

from llvmlite import binding, ir
from test_ida2llvm_microcode import L


def body(module):
    fn = ir.Function(module, ir.FunctionType(ir.IntType(32), [ir.IntType(32)]), 'f')
    block = fn.append_basic_block('entry')
    b = ir.IRBuilder(block)
    slot = b.alloca(ir.IntType(32))
    fn.lvars = {'x': slot}
    fn.block_by_ea = {123: block}
    b.store(fn.args[0], slot)
    b.ret(b.load(slot))
    return fn, weakref.ref(block), weakref.ref(slot)


def test_release_body_roots_and_scope(L):
    module = ir.Module()
    fn, block, slot = body(module)
    L._release_function_body(fn)
    gc.collect()
    assert block() is None and slot() is None
    assert fn.is_declaration
    assert fn.scope._useset == {'', fn.args[0].name}
    assert module.globals['f'] is fn
    binding.parse_assembly(str(module)).verify()


def test_streamed_definition_and_later_call(L):
    module = ir.Module()
    fn, block, slot = body(module)
    output = io.StringIO()
    controller = types.SimpleNamespace(
        _stream_file=output, _streamed_functions=set(),
        _stream_new_type_declarations=lambda: None,
    )
    L.BIN2LLVMController._stream_function_if_enabled(controller, fn)
    gc.collect()
    assert block() is None and slot() is None
    assert fn._streamed_definition
    caller = ir.Function(module, fn.function_type, 'caller')
    b = ir.IRBuilder(caller.append_basic_block())
    b.ret(b.call(fn, caller.args))
    binding.parse_assembly(output.getvalue() + str(caller)).verify()
