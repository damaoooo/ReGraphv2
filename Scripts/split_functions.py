"""Two-level function extraction; standard-library-only command-line driver."""
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CHUNK_SIZE = 128
MODULE_ID = "; ModuleID = 'regraph-function'\n"
DEFAULT_EXTRACTOR = Path(__file__).resolve().parents[1] / 'LLVMTools' / 'build' / 'bin' / 'split_funcs'


def configured_extractor_path() -> Path:
    return Path(os.environ.get('REGRAPH_SPLIT_FUNCS', DEFAULT_EXTRACTOR)).resolve()


def run_command(arguments, **kwargs):
    return subprocess.run(list(map(str, arguments)), check=True, text=True, **kwargs)


def resolve_extraction_backend() -> Path | None:
    """Only missing tools fall back; broken or mismatched builds fail explicitly."""
    extractor = configured_extractor_path()
    if not extractor.exists():
        return None
    compiled_version = run_command([extractor, '--version'], capture_output=True).stdout
    cli_version = run_command(['llvm-extract', '--version'], capture_output=True).stdout
    compiled_match = re.search(r'\b\d+\.\d+\.\d+\b', compiled_version)
    cli_match = re.search(r'\b\d+\.\d+\.\d+\b', cli_version)
    if not compiled_match or not cli_match or compiled_match[0] != cli_match[0]:
        raise RuntimeError('split_funcs and llvm-extract LLVM versions differ; rebuild split_funcs')
    return extractor


def list_defined_functions(bitcode_path: Path) -> list[str]:
    listing = run_command(['llvm-nm', '--defined-only', '--format=posix', bitcode_path],
                          capture_output=True).stdout
    names = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f'Unsupported llvm-nm symbol row: {line!r}')
        name, symbol_type, _, _ = fields
        if symbol_type not in ('T', 't', 'W', 'w'):
            continue
        if any(character.isspace() for character in name) or '"' in name:
            raise ValueError(f'Unsupported function name: {name!r}')
        names.append(name)
    return names


def extract_functions(input_path: Path, output_directory: Path) -> None:
    extractor = resolve_extraction_backend()
    if extractor is None:
        print('Warning: split_funcs unavailable; using two-level llvm-extract fallback.', file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix='regraph-extract-') as temporary_directory:
        workspace = Path(temporary_directory)
        if input_path.suffix == '.ll':
            bitcode_path = workspace / 'input.bc'
            run_command(['llvm-as', input_path, '-o', bitcode_path])
        elif input_path.suffix == '.bc':
            bitcode_path = input_path
        else:
            raise ValueError('Input must be a .ll or .bc file')
        names = list_defined_functions(bitcode_path)
        if not names:
            print(f'Warning: no defined functions in {input_path}', file=sys.stderr)
            return
        output_directory.mkdir(parents=True, exist_ok=True)
        filenames = {name: hashlib.sha1(name.encode()).hexdigest() + '.ll' for name in names}
        for start in range(0, len(names), CHUNK_SIZE):
            group = names[start:start + CHUNK_SIZE]
            chunk_path = workspace / 'chunk.bc'
            run_command(['llvm-extract', *[f'--func={name}' for name in group],
                         bitcode_path, '-o', chunk_path])
            if extractor is not None:
                manifest = ''.join(f'{name}\t{filenames[name]}\n' for name in group)
                run_command([extractor, chunk_path, output_directory], input=manifest)
            else:
                for name in group:
                    output_path = output_directory / filenames[name]
                    run_command(['llvm-extract', '-S', f'--func={name}', chunk_path, '-o', output_path])
                    text = output_path.read_text()
                    first_line, separator, remainder = text.partition('\n')
                    if not separator or not first_line.startswith('; ModuleID = '):
                        raise RuntimeError(f'Missing ModuleID header: {output_path}')
                    output_path.write_text(MODULE_ID + remainder)
        with (output_directory / 'function_map.csv').open('w', newline='') as output:
            output.write('OriginalFunctionName,HashedFileName\n')
            writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator='\n')
            writer.writerows((name, filenames[name]) for name in names)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: split_functions.py input.ll|input.bc output_directory')
    try:
        extract_functions(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
