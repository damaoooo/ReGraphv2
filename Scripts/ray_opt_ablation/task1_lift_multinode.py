#!/usr/bin/env python3
"""Multi-node shard runner for ReGraph task1 lifting on Shaheen.

This runner is intentionally self-contained so it can live under
Scripts/ray_opt_ablation while reusing Scripts/ida2llvm.py.  Use the
submit_task1_lift_multinode.sh wrapper to launch one Slurm array task per
node.  /scratch is shared on Shaheen, so shards can write to the same output
directory without a sync step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IDA2LLVM = REPO_ROOT / 'Scripts' / 'ida2llvm.py'
DEFAULT_INPUT = Path('/scratch/zhoul0e/Dataset-1')
DEFAULT_OUTPUT = Path('/scratch/zhoul0e/Dataset-1-lift')
DEFAULT_IDA = Path('/scratch/zhoul0e/ida-pro-9.3')
DEFAULT_RELL = Path('/scratch/zhoul0e/miniconda3/envs/ReLL/bin/python')
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / 'slurm_logs' / 'task1_multinode'
SKIP_INPUT_SUFFIXES = (
    '.i64', '.idb', '.id0', '.id1', '.id2', '.til', '.nam', '.asm',
    '.ll', '.bc', '.c', '.cpp', '.h', '.hpp', '.txt', '.log', '.md', '.py', '.sh',
)


@dataclass(frozen=True)
class LiftTask:
    input_path: Path
    output_path: Path


@dataclass
class LiftResult:
    ok: bool
    input_path: str
    output_path: str
    returncode: int | None
    duration: float
    timed_out: bool = False
    stdout_tail: str = ''
    stderr_tail: str = ''


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {'1', 'true', 'yes', 'on'}


def slurm_array_count() -> int:
    if os.environ.get('SLURM_ARRAY_TASK_COUNT'):
        return int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    if os.environ.get('SLURM_ARRAY_TASK_MIN') and os.environ.get('SLURM_ARRAY_TASK_MAX'):
        return int(os.environ['SLURM_ARRAY_TASK_MAX']) - int(os.environ['SLURM_ARRAY_TASK_MIN']) + 1
    return 1


def slurm_array_index() -> int:
    if os.environ.get('SLURM_ARRAY_TASK_ID'):
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', '0'))
        return task_id - task_min
    return 0


def existing_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_candidate_binary(path: Path) -> bool:
    if not path.is_file() or path.name.startswith('.'):
        return False
    return not path.name.endswith(SKIP_INPUT_SUFFIXES)


def output_for(input_root: Path, output_root: Path, input_file: Path) -> Path:
    rel = input_file.relative_to(input_root)
    return output_root / input_root.name / rel.parent / f'{input_file.name}.ll'


def scan_tasks(input_root: Path, output_root: Path) -> list[LiftTask]:
    files = sorted((p for p in input_root.rglob('*') if is_candidate_binary(p)), key=lambda p: p.relative_to(input_root).as_posix())
    return [LiftTask(p, output_for(input_root, output_root, p)) for p in files]


def shard_tasks(tasks: list[LiftTask], shard_index: int, num_shards: int) -> list[LiftTask]:
    return [task for index, task in enumerate(tasks) if index % num_shards == shard_index]


def build_env(home: Path, ida_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env['HOME'] = str(home)
    env['IDADIR'] = str(ida_dir)
    env.setdefault('REGRAPH_IDA_HOME', str(home))
    return env


def tail(text: str | bytes | None, limit: int = 4000) -> str:
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode(errors='replace')
    return text[-limit:]


def lift_one(task: LiftTask, python_bin: str, ida_dir: Path, home: Path, timeout_seconds: int) -> LiftResult:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cmd = [python_bin, str(IDA2LLVM), '-f', str(task.input_path), '-o', str(task.output_path), '-v']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=build_env(home, ida_dir), timeout=timeout_seconds)
        ok = result.returncode == 0 and existing_nonempty(task.output_path)
        if not ok and task.output_path.exists():
            task.output_path.unlink(missing_ok=True)
        return LiftResult(ok, str(task.input_path), str(task.output_path), result.returncode, time.monotonic() - start, False, tail(result.stdout), tail(result.stderr))
    except subprocess.TimeoutExpired as exc:
        if task.output_path.exists():
            task.output_path.unlink(missing_ok=True)
        return LiftResult(False, str(task.input_path), str(task.output_path), None, time.monotonic() - start, True, tail(exc.stdout), tail(exc.stderr))


def write_failure(log, result: LiftResult) -> None:
    print('=' * 60, file=log)
    print(f'Input: {result.input_path}', file=log)
    print(f'Output: {result.output_path}', file=log)
    print(f'Duration: {result.duration:.1f}s', file=log)
    if result.timed_out:
        print('Result: timeout', file=log)
    else:
        print(f'Return code: {result.returncode}', file=log)
        if result.returncode is not None and result.returncode < 0:
            print(f'Signal: {-result.returncode}', file=log)
    if result.stdout_tail:
        print('--- stdout tail ---', file=log)
        print(result.stdout_tail, file=log)
    if result.stderr_tail:
        print('--- stderr tail ---', file=log)
        print(result.stderr_tail, file=log)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description='Run one shard of task1_lift on a Shaheen node.')
    parser.add_argument('--input-path', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--num-shards', type=positive_int, default=slurm_array_count())
    parser.add_argument('--shard-index', type=int, default=slurm_array_index())
    parser.add_argument('--workers', type=positive_int, default=int(os.environ.get('REGRAPH_TASK1_WORKERS', os.environ.get('SLURM_CPUS_PER_TASK', '384'))))
    parser.add_argument('--timeout-seconds', type=positive_int, default=int(os.environ.get('REGRAPH_TASK1_TIMEOUT_SECONDS', '300')))
    parser.add_argument('--python', default=os.environ.get('REGRAPH_RELL_PYTHON', str(DEFAULT_RELL)))
    parser.add_argument('--ida-dir', type=Path, default=Path(os.environ.get('IDADIR', str(DEFAULT_IDA))))
    parser.add_argument('--home', type=Path, default=Path(os.environ.get('REGRAPH_IDA_HOME', '/scratch/zhoul0e')))
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--resume', action='store_true', default=parse_bool_env('REGRAPH_TASK1_RESUME', True))
    parser.add_argument('--allow-failures', action='store_true', default=parse_bool_env('REGRAPH_TASK1_ALLOW_FAILURES', True))
    parser.add_argument('--max-files', type=int, default=0, help='debug limit after sharding; 0 means unlimited')
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f'shard-index must be in [0, {args.num_shards - 1}]')
    if not args.input_path.is_dir():
        raise SystemExit(f'input path does not exist: {args.input_path}')
    if not IDA2LLVM.is_file():
        raise SystemExit(f'ida2llvm.py not found: {IDA2LLVM}')

    args.log_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID', 'local')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', str(args.shard_index))
    log_path = args.log_dir / f'task1_multinode_{job}_{task_id}.log'
    failed_path = args.log_dir / f'task1_multinode_failed_{job}_{task_id}.txt'

    all_tasks = scan_tasks(args.input_path, args.output)
    shard = shard_tasks(all_tasks, args.shard_index, args.num_shards)
    before_resume = len(shard)
    if args.resume:
        shard = [task for task in shard if not existing_nonempty(task.output_path)]
    if args.max_files > 0:
        shard = shard[:args.max_files]

    print('Task1 multi-node shard runner')
    print(f'Input: {args.input_path}')
    print(f'Output root: {args.output / args.input_path.name}')
    print(f'Shard: {args.shard_index}/{args.num_shards}')
    print(f'Total candidate files: {len(all_tasks)}')
    print(f'Shard candidate files: {before_resume}')
    print(f'To process after resume/max-files: {len(shard)}')
    print(f'Workers: {args.workers}')
    print(f'Timeout per file: {args.timeout_seconds}s')
    print(f'Log: {log_path}')
    print(f'Failed list: {failed_path}')
    sys.stdout.flush()

    success_count = 0
    failed: list[LiftResult] = []
    start = time.monotonic()
    with open(log_path, 'a') as log:
        print(f'=== shard {args.shard_index}/{args.num_shards} start ===', file=log)
        print(f'tasks={len(shard)} workers={args.workers} timeout={args.timeout_seconds}', file=log)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(lift_one, task, args.python, args.ida_dir, args.home, args.timeout_seconds): task for task in shard}
            for completed, future in enumerate(as_completed(future_to_task), 1):
                result = future.result()
                if result.ok:
                    success_count += 1
                else:
                    failed.append(result)
                    write_failure(log, result)
                    log.flush()
                if completed == len(shard) or completed % 50 == 0:
                    elapsed = time.monotonic() - start
                    print(f'progress {completed}/{len(shard)} success={success_count} failed={len(failed)} elapsed={elapsed:.1f}s')
                    sys.stdout.flush()
        print(f'=== shard done success={success_count} failed={len(failed)} ===', file=log)

    if failed:
        with open(failed_path, 'w') as failed_file:
            for result in failed:
                print(result.input_path, file=failed_file)

    print('Final summary')
    print(f'Success: {success_count}')
    print(f'Failed: {len(failed)}')
    if failed:
        print(f'Failed list: {failed_path}')
    return 0 if args.allow_failures or not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
