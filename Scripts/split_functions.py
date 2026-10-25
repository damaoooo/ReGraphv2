"""Two-level function extraction; standard-library-only command-line driver.

usage: split_functions.py INPUT.ll|INPUT.bc OUTPUT_DIR (--lift-state X.ll.lift.json | --no-segment-filter)
                          [--select X.ll.select.json]

Functions the lifter recorded in EXCLUDED_SEGMENTS are not extracted: PLT stubs only forward to an
imported symbol, and .init/.fini hold compiler-generated startup/teardown code. Only the samples leave
them out; the .ll/.bc modules stay as lifted, since other functions there still call these definitions.
A selection file (Scripts/select_functions.py) further restricts the samples to the functions it lists.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CHUNK_SIZE = 128
MODULE_ID = "; ModuleID = 'regraph-function'\n"
DEFAULT_EXTRACTOR = Path(__file__).resolve().parents[1] / 'LLVMTools' / 'build' / 'bin' / 'split_funcs'
# IDA segment names whose functions are not samples (an explicit list, not general thunk detection):
# PLT stub sections (.MIPS.stubs is the MIPS lazy-binding equivalent) and the compiler's .init/.fini code
EXCLUDED_SEGMENTS = frozenset({'.plt', '.plt.got', '.plt.sec', '.iplt', '.MIPS.stubs', '.init', '.fini'})


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


def lift_state_path(module_path: Path) -> Path:
    """`X.ll` / `X.bc` -> `X.ll.lift.json`, the state file Task 1 publishes next to its output."""
    return module_path.with_suffix('.ll.lift.json')


def _file_sha256(path: Path) -> str:
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_lift_state(state_path: Path, module_path: Path) -> dict:
    """Load a Task 1 lift state and check that it was written for the module's source `X.ll`.

    The binding is by content: a successful lift whose recorded output checksum is that of `X.ll`, which
    must exist. That `X.bc` was optimized from this `X.ll` is vouched for by Task 2's own completion
    record and resume check, not here; callers only add that the lifted functions are defined in X.bc.
    """
    try:
        state = json.loads(state_path.read_text())
        status, recorded = state.get('status'), state.get('output') or {}
        recorded_sha256 = recorded.get('sha256')
    except AttributeError as error:
        raise ValueError(f'{state_path} is not a Task 1 lift state') from error
    if status != 'ok':
        raise ValueError(f'{state_path}: lift status is {status!r}, not ok')
    source = module_path.with_suffix('.ll')
    if not source.exists():
        raise ValueError(f'{source} is missing; cannot tell whether {state_path} describes {module_path}')
    if _file_sha256(source) != recorded_sha256:
        raise ValueError(f'{state_path} was not written for {source}')
    return state


def load_excluded_function_names(state_path: Path, module_path: Path, defined: list[str]) -> set[str]:
    """LLVM names of the functions the verified lift state places in EXCLUDED_SEGMENTS.

    The lifted functions must all be defined in the module (the lifter may add helper definitions of its
    own), and each needs a segment: unknown is not `.text`.
    """
    state = verify_lift_state(state_path, module_path)
    try:
        return _excluded_function_names(state['functions'], state_path, module_path, set(defined))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'{state_path} is not a Task 1 lift state ({type(error).__name__}: {error})') from error


def _excluded_function_names(functions: list, state_path: Path, module_path: Path, defined: set[str]) -> set[str]:
    lifted = [record for record in functions if record.get('status') == 'lifted']
    undefined = [record['llvm_name'] for record in lifted if record['llvm_name'] not in defined]
    if undefined:
        raise ValueError(f'{state_path} does not describe {module_path}: '
                         f'{len(undefined)} lifted functions are not defined there, e.g. {undefined[0]}')
    unknown = [record['llvm_name'] for record in lifted if not record.get('segment')]
    if unknown:
        raise ValueError(f'{state_path} records no segment for {len(unknown)} functions, e.g. {unknown[0]}; '
                         're-run Task 1 with the current lifter')
    return {record['llvm_name'] for record in lifted if record['segment'] in EXCLUDED_SEGMENTS}


def load_selection(path: Path) -> set[str]:
    """LLVM names listed in a select_functions.py selection file."""
    names = json.loads(path.read_text()).get('functions')
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f'Invalid function selection: {path}')
    return set(names)


def extract_functions(input_path: Path, output_directory: Path, lift_state: Path | None,
                      selection: set[str] | None = None) -> None:
    """Write one `<sha1(name)>.ll` per sample function and `function_map.csv` into an empty directory.

    `lift_state` is the Task 1 state used to leave out EXCLUDED_SEGMENTS; None keeps every function.
    `selection`, if given, keeps only the functions it names.
    """
    if output_directory.exists() and any(output_directory.iterdir()):
        # Files of an earlier run (e.g. functions excluded now) would otherwise stay next to the new set
        raise ValueError(f'output directory {output_directory} is not empty')
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
        defined = list_defined_functions(bitcode_path)
        excluded = load_excluded_function_names(lift_state, input_path, defined) if lift_state else set()
        names = [name for name in defined if name not in excluded]
        if excluded:
            print(f'Skipped {len(defined) - len(names)} functions in excluded segments in {input_path}',
                  file=sys.stderr)
        if selection is not None:
            unselected = len(names)
            names = [name for name in names if name in selection]
            print(f'Skipped {unselected - len(names)} unselected functions in {input_path}', file=sys.stderr)
        if not names:
            print(f'Warning: no functions to extract in {input_path}', file=sys.stderr)
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split an LLVM module into one file per function.')
    parser.add_argument('input', type=Path, help='input .ll or .bc module')
    parser.add_argument('output', type=Path, help='empty or missing output directory')
    segment_filter = parser.add_mutually_exclusive_group(required=True)
    segment_filter.add_argument('--lift-state', type=Path,
                                help='Task 1 state file of the module; leaves out EXCLUDED_SEGMENTS')
    segment_filter.add_argument('--no-segment-filter', action='store_true',
                                help='keep every function (no lift state, e.g. the Ghidra backend)')
    parser.add_argument('--select', type=Path,
                        help='selection file (select_functions.py); extract only the functions it lists')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    try:
        extract_functions(arguments.input.resolve(), arguments.output.resolve(),
                          arguments.lift_state.resolve() if arguments.lift_state else None,
                          load_selection(arguments.select) if arguments.select else None)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
