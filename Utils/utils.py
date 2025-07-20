import os
import sys
import contextlib

DEFAULT_DDG_SO_PATH = '/home/damaoooo/Downloads/regraphv2/GraphBuilder/ddg_exporter/build/libDDGPrinter.so'
DEFAULT_PURIFY_SO_PATH = '/home/damaoooo/Downloads/regraphv2/GraphBuilder/meta_remover/build/libStripAllMetadataPass.so'
DEFAULT_CFG_SO_PATH = '/home/damaoooo/Downloads/regraphv2/GraphBuilder/cfg_exporter/build/libMyCFGPrinterPass.so'
DEFAULT_TOKENIZER_PATH = '/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json'

@contextlib.contextmanager
def suppress_stderr():
    """
    一个上下文管理器，用于临时抑制对 stderr 的写入。
    它通过在操作系统级别重定向文件描述符来实现，因此对 C 扩展也有效。
    """
    # os.devnull 是一个跨平台的空设备路径 (在 Unix 上是 '/dev/null')
    devnull = open(os.devnull, 'w')
    # 获取 stderr 的原始文件描述符 (通常是 2)
    stderr_fd = sys.stderr.fileno()
    # 复制一份原始的 stderr 文件描述符，以便之后恢复
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        # 将 stderr 的文件描述符重定向到空设备
        os.dup2(devnull.fileno(), stderr_fd)
        
        # yield 关键字是上下文管理器的核心，您的代码将在这里执行
        yield
    finally:
        # 无论 try 块中发生什么（即使是异常），finally 块都会执行
        
        # 将 stderr 恢复到原始状态
        os.dup2(saved_stderr_fd, stderr_fd)
        # 关闭我们保存的描述符副本
        os.close(saved_stderr_fd)
        # 关闭我们打开的空设备文件
        devnull.close()
        

def should_skip_this_file(llvm_str: str):
    """
    判断是否应该跳过这个LLVM IR文件
    跳过条件：文件中所有的函数定义的函数体都不超过3行
    
    Args:
        llvm_str: LLVM IR代码字符串
        
    Returns:
        bool: True表示应该跳过，False表示不应该跳过
    """
    lines = llvm_str.split('\n')
    i = 0
    has_function = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 查找函数定义开始
        if line.startswith('define ') and '{' in line:
            has_function = True
            function_body_lines = 0
            i += 1
            
            # 统计函数体行数
            brace_count = 1  # 已经有一个开括号
            while i < len(lines) and brace_count > 0:
                current_line = lines[i].strip()
                
                # 跳过空行和注释
                if current_line and not current_line.startswith(';'):
                    function_body_lines += 1
                
                # 统计大括号
                brace_count += current_line.count('{')
                brace_count -= current_line.count('}')
                
                i += 1
            
            # 如果函数体超过3行，则不跳过
            if function_body_lines > 3:
                return False
        else:
            i += 1
    
    # 如果没有函数定义，或者所有函数体都不超过3行，则跳过
    return has_function
