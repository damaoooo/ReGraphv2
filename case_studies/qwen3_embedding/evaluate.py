import logging
import pickle
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
# --- 核心修正：引入 AutoTokenizer ---
from transformers import AutoTokenizer, set_seed
from datasets import load_from_disk
from numba import njit, prange, types
from numba.typed import Dict as NumbaDict
import torch


# --- 设置 Rich 和 Typer ---
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

# --- GPU加速支持 ---
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    console.print("[green]✓ GPU加速已启用 (PyTorch)[/green]")
else:
    console.print("[yellow]⚠ 未检测到可用GPU，将使用CPU计算[/yellow]")

MRR_CUTOFFS = (10, 30)


@njit
def _seed_numba_rng(seed: int) -> None:
    """为 numba 内部使用的随机数生成器设置种子。"""
    np.random.seed(seed)


def _sanitize_positive_map(positive_map, total_size: int):
    """清洗 positive_map，移除越界/自指向/重复正样本。"""
    sanitized = {}
    stats = {
        "invalid_anchor": 0,
        "invalid_positive": 0,
        "self_positive": 0,
        "duplicate_positive": 0,
        "empty_anchor": 0,
    }

    for raw_anchor_idx, positives in positive_map.items():
        try:
            anchor_idx = int(raw_anchor_idx)
        except (TypeError, ValueError):
            stats["invalid_anchor"] += 1
            continue

        if anchor_idx < 0 or anchor_idx >= total_size:
            stats["invalid_anchor"] += 1
            continue

        seen = set()
        cleaned = []
        for raw_pos_idx in positives:
            try:
                pos_idx = int(raw_pos_idx)
            except (TypeError, ValueError):
                stats["invalid_positive"] += 1
                continue

            if pos_idx < 0 or pos_idx >= total_size:
                stats["invalid_positive"] += 1
                continue
            if pos_idx == anchor_idx:
                stats["self_positive"] += 1
                continue
            if pos_idx in seen:
                stats["duplicate_positive"] += 1
                continue

            seen.add(pos_idx)
            cleaned.append(pos_idx)

        if cleaned:
            sanitized[anchor_idx] = cleaned
        else:
            stats["empty_anchor"] += 1

    return sanitized, stats



def generate_embeddings_with_tei(
    dataset,
    batch_size: int,
    instruction: str,
    tei_endpoint: str,
    tokenizer,
    max_length: int,
    tei_workers: int = 8,
    tei_timeout: int = 60,
    tei_max_retries: int = 8,
    tei_retry_base_delay: float = 2.0,
) -> np.ndarray:
    import concurrent.futures
    import threading

    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def iter_batches():
        for batch_index, start in enumerate(range(0, len(dataset), batch_size)):
            yield batch_index, dataset[start : start + batch_size]["text"]

    def process_one_batch(batch_texts):
        session = get_session()
        instructed_texts = [instruction + text for text in batch_texts]
        truncated_inputs = tokenizer(
            instructed_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        final_texts_to_send = tokenizer.batch_decode(truncated_inputs["input_ids"], skip_special_tokens=True)
        payload = {"inputs": final_texts_to_send}

        for attempt in range(1, tei_max_retries + 1):
            try:
                response = session.post(f"{tei_endpoint}/embed", json=payload, timeout=tei_timeout)
                response.raise_for_status()
                return np.array(response.json(), dtype=np.float32)
            except requests.exceptions.RequestException as exc:
                if attempt >= tei_max_retries:
                    console.print(f"[bold red]错误: 一个TEI批次在重试后仍失败: {exc}[/bold red]")
                    raise

                sleep_seconds = min(tei_retry_base_delay * (2 ** (attempt - 1)), 60.0) + random.random()
                console.print(
                    f"[yellow]TEI批次请求失败，第 {attempt}/{tei_max_retries} 次重试将在 "
                    f"{sleep_seconds:.1f}s 后进行: {exc}[/yellow]"
                )
                time.sleep(sleep_seconds)

    total_size = len(dataset)
    total_batches = (total_size + batch_size - 1) // batch_size
    description = f"生成嵌入(Workers: {tei_workers})"
    all_embeddings = None
    next_write_offset = 0

    def append_batch(batch_emb: np.ndarray) -> int:
        nonlocal all_embeddings, next_write_offset
        if all_embeddings is None:
            all_embeddings = np.empty((total_size, batch_emb.shape[1]), dtype=np.float32)
        end_offset = next_write_offset + len(batch_emb)
        all_embeddings[next_write_offset:end_offset] = batch_emb
        next_write_offset = end_offset
        return next_write_offset

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[speed]:.1f} 函数/秒"),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=total_size, speed=0.0)

            if tei_workers == 1:
                for _, batch_texts in iter_batches():
                    batch_emb = process_one_batch(batch_texts)
                    processed_count = append_batch(batch_emb)
                    elapsed = progress.tasks[0].elapsed or 0.001
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    progress.update(task, advance=len(batch_emb), speed=speed)
            else:
                batch_iterator = iter_batches()
                next_expected_batch = 0
                completed_batches = {}
                max_in_flight = max(tei_workers * 2, 1)

                with concurrent.futures.ThreadPoolExecutor(max_workers=tei_workers) as executor:
                    pending = {}

                    def submit_one_batch() -> bool:
                        try:
                            batch_index, batch_texts = next(batch_iterator)
                        except StopIteration:
                            return False
                        pending[executor.submit(process_one_batch, batch_texts)] = batch_index
                        return True

                    for _ in range(min(max_in_flight, total_batches)):
                        if not submit_one_batch():
                            break

                    while pending:
                        done, _ = concurrent.futures.wait(
                            tuple(pending.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            batch_index = pending.pop(future)
                            completed_batches[batch_index] = future.result()
                            submit_one_batch()

                        while next_expected_batch in completed_batches:
                            batch_emb = completed_batches.pop(next_expected_batch)
                            processed_count = append_batch(batch_emb)
                            elapsed = progress.tasks[0].elapsed or 0.001
                            speed = processed_count / elapsed if elapsed > 0 else 0
                            progress.update(task, advance=len(batch_emb), speed=speed)
                            next_expected_batch += 1

    except Exception:
        console.print("[bold red]嵌入向量生成过程中发生错误，程序已终止。[/bold red]")
        raise typer.Exit(code=1)

    if all_embeddings is None:
        return np.empty((0, 0), dtype=np.float32)
    return all_embeddings


@njit
def _floyd_sample(pop_size: int, sample_size: int) -> np.ndarray:
    """Floyd算法：在O(k)时间内均匀采样k个不重复整数。"""
    selected = NumbaDict.empty(key_type=types.int64, value_type=types.boolean)
    for j in range(pop_size - sample_size, pop_size):
        t = np.random.randint(0, j + 1)
        if t in selected:
            selected[j] = True
        else:
            selected[t] = True
    out = np.empty(sample_size, dtype=np.int64)
    idx = 0
    for key in selected.keys():
        out[idx] = key
        idx += 1
    return out


@njit
def _map_exclusions(compressed: np.ndarray, exclude_arr: np.ndarray) -> np.ndarray:
    """把压缩空间索引映射回真实索引，保证排除表不被选中。"""
    mapped = compressed.copy()
    while True:
        shift = np.searchsorted(exclude_arr, mapped, side="right")
        new_mapped = compressed + shift
        if np.all(new_mapped == mapped):
            return new_mapped
        mapped = new_mapped


@njit
def _sample_excluding(total_size: int, exclude_arr: np.ndarray, sample_size: int) -> np.ndarray:
    """在排除表之外做均匀无放回采样，返回真实索引。"""
    eligible_size = total_size - exclude_arr.size
    if eligible_size < sample_size:
        return np.empty(0, dtype=np.int64)
    compressed = _floyd_sample(eligible_size, sample_size)
    return _map_exclusions(compressed, exclude_arr)


@njit(parallel=True)
def _build_pools_parallel(anchor_batch: np.ndarray, pos_flat: np.ndarray, pos_offsets: np.ndarray, total_size: int, pool_size: int) -> np.ndarray:
    """并行构建每个锚点的采样池（CPU多核）。"""
    batch_size = anchor_batch.size
    pools = np.full((batch_size, pool_size + 1), -1, dtype=np.int64)
    for i in prange(batch_size):
        anchor_idx = anchor_batch[i]
        start = pos_offsets[anchor_idx]
        end = pos_offsets[anchor_idx + 1]
        pos_len = end - start
        if pos_len <= 0:
            continue
        rand_idx = np.random.randint(start, end)
        positive_anchor_idx = pos_flat[rand_idx]
        exclude_arr = np.empty(pos_len + 1, dtype=np.int64)
        exclude_arr[:pos_len] = pos_flat[start:end]
        exclude_arr[pos_len] = anchor_idx
        exclude_arr.sort()

        mapped = _sample_excluding(total_size, exclude_arr, pool_size)
        if mapped.size != pool_size:
            continue

        pools[i, 0] = positive_anchor_idx
        pools[i, 1:] = mapped

    return pools


def process_anchor_batch_gpu(all_embeddings, anchor_batch, pos_flat, pos_offsets, pool_sizes, k_values: List[int], use_gpu: bool = True):
    """
    处理锚点批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for pool_size in pool_sizes:
        recalls[pool_size] = {}
        for k in k_values:
            recalls[pool_size][k] = [0, 0]  # 每次都创建新的列表

    mrr_stats = {cutoff: [0.0, 0] for cutoff in MRR_CUTOFFS}
    mrr_pool_stats = {pool_size: [0.0, 0] for pool_size in pool_sizes}
    max_pool_size = max(pool_sizes)
    pool_size = max_pool_size - 1
    total_size = len(all_embeddings)
    anchor_batch_arr = np.asarray(anchor_batch, dtype=np.int64)
    pools = _build_pools_parallel(anchor_batch_arr, pos_flat, pos_offsets, total_size, pool_size)
    valid_mask = pools[:, 0] >= 0
    if not np.any(valid_mask):
        return recalls, mrr_stats, mrr_pool_stats

    pools = pools[valid_mask]
    anchor_batch_arr = anchor_batch_arr[valid_mask]

    if use_gpu:
        device = all_embeddings.device
        anchor_idx = torch.from_numpy(anchor_batch_arr).to(device=device, dtype=torch.long)
        pool_idx = torch.from_numpy(pools).to(device=device, dtype=torch.long)
        with torch.inference_mode():
            anchors = all_embeddings.index_select(0, anchor_idx)
            embedding_pools = all_embeddings.index_select(0, pool_idx.view(-1)).view(pool_idx.shape[0], pool_idx.shape[1], -1)
            anchor_emb = anchors.unsqueeze(1)
            similarities = torch.bmm(embedding_pools, anchor_emb.transpose(1, 2)).squeeze(-1)
    else:
        anchors = all_embeddings[anchor_batch_arr]
        embedding_pools = all_embeddings[pools]
        anchor_emb = anchors[:, np.newaxis, :]
        similarities = np.matmul(embedding_pools, np.transpose(anchor_emb, (0, 2, 1))).squeeze(-1)

    # 计算MRR（在最大pool中取排名，超过阈值则记为0）
    mrr_sim_slice = similarities[:, :similarities.shape[1]]
    pos_scores = mrr_sim_slice[:, 0:1]
    if use_gpu:
        count_greater = (mrr_sim_slice > pos_scores).sum(dim=1)
        ranks = count_greater + 1
    else:
        count_greater = (mrr_sim_slice > pos_scores).sum(axis=1)
        ranks = count_greater + 1
    for cutoff in MRR_CUTOFFS:
        mrr_cutoff = min(cutoff, mrr_sim_slice.shape[1])
        if use_gpu:
            mrr_scores = torch.where(ranks <= mrr_cutoff, 1.0 / ranks.to(dtype=torch.float32), torch.zeros_like(ranks, dtype=torch.float32))
            mrr_stats[cutoff][0] += float(mrr_scores.sum().item())
            mrr_stats[cutoff][1] += int(mrr_scores.numel())
        else:
            mrr_scores = np.where(
                ranks <= mrr_cutoff,
                1.0 / ranks.astype(np.float32),
                0.0,
            )
            mrr_stats[cutoff][0] += float(mrr_scores.sum())
            mrr_stats[cutoff][1] += int(mrr_scores.size)

    # 计算Recall@K（不做全量排序，直接比较正样本得分排名）
    for pool_size in pool_sizes:
        sim_slice = similarities[:, :pool_size]
        pos_scores = sim_slice[:, 0:1]
        if use_gpu:
            count_greater = (sim_slice > pos_scores).sum(dim=1)
            ranks = count_greater + 1
            mrr_pool_scores = 1.0 / ranks.to(dtype=torch.float32)
            mrr_pool_stats[pool_size][0] += float(mrr_pool_scores.sum().item())
            mrr_pool_stats[pool_size][1] += int(mrr_pool_scores.numel())
        else:
            count_greater = (sim_slice > pos_scores).sum(axis=1)
            ranks = count_greater + 1
            mrr_pool_scores = 1.0 / ranks.astype(np.float32)
            mrr_pool_stats[pool_size][0] += float(mrr_pool_scores.sum())
            mrr_pool_stats[pool_size][1] += int(mrr_pool_scores.size)
        for k in k_values:
            if use_gpu:
                success = (count_greater < k).sum().item()
                total = int(count_greater.numel())
            else:
                success = int((count_greater < k).sum())
                total = int(count_greater.size)
            assert success <= total, f"Success count {success} cannot be greater than total {total}."
            recalls[pool_size][k][0] += success
            recalls[pool_size][k][1] += total

    return recalls, mrr_stats, mrr_pool_stats


@app.command()
def main(
    validation_dataset_pool_path: Path = typer.Argument(..., help="验证集数据池的路径。", exists=True, dir_okay=True),
    validation_positive_map_path: Path = typer.Argument(..., help="验证集正样本映射.pkl文件路径。", exists=True, file_okay=True),
    tei_endpoint: str = typer.Option("http://127.0.0.1:8080", help="Text Embedding Inference (TEI) 服务器的URL。"),
    ks_str: str = typer.Option("1,5,10,15,20,25,30,35,40,45,50", "--ks", "-k", help="要评估的K值，以逗号分隔。"),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="发送到TEI服务器的批量大小。"),
    max_length: int = typer.Option(2048, "--max-length", help="发送到TEI前，将'指令+文本'整体截断到的最大token长度。"),
    tei_workers: int = typer.Option(8, "--tei-workers", help="并发发送到TEI的worker数量。"),
    tei_timeout: int = typer.Option(60, "--tei-timeout", help="单个TEI请求的超时时间（秒）。"),
    tei_max_retries: int = typer.Option(8, "--tei-max-retries", help="单个TEI批次的最大重试次数。"),
    tei_retry_base_delay: float = typer.Option(2.0, "--tei-retry-base-delay", help="TEI重试的指数退避起始秒数。"),
    eval_samples: int = typer.Option(187256, "--eval-samples", "-n", help="用于评估的随机锚点样本数量。"),
    embeddings_path: Path = typer.Option(None, "--embeddings-path", "-e", help="用于保存/加载嵌入向量Numpy文件的路径。"),
    seed: int = typer.Option(42, "--seed", "-s", help="用于负采样的随机种子。"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="是否使用GPU加速计算。"),
    gpu_batch_size: int = typer.Option(512, "--gpu-batch-size", help="GPU批量处理的锚点数量。"),
):
    """
    在验证集上评估模型的函数检索性能 (Recall@K)，使用TEI服务器加速嵌入生成，GPU加速相似度计算。
    """
    console.rule(f"[bold blue]开始使用TEI进行模型评估[/bold blue]")
    set_seed(seed)
    _seed_numba_rng(seed)

    if tei_workers <= 0:
        console.print("[bold red]错误: --tei-workers 必须大于 0。[/bold red]")
        raise typer.Exit(code=1)
    if tei_timeout <= 0:
        console.print("[bold red]错误: --tei-timeout 必须大于 0。[/bold red]")
        raise typer.Exit(code=1)
    if tei_max_retries <= 0:
        console.print("[bold red]错误: --tei-max-retries 必须大于 0。[/bold red]")
        raise typer.Exit(code=1)
    if tei_retry_base_delay < 0:
        console.print("[bold red]错误: --tei-retry-base-delay 不能为负数。[/bold red]")
        raise typer.Exit(code=1)

    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但PyTorch不可用，将回退到CPU计算[/yellow]")
        use_gpu = False

    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")

    # --- 1. 生成或加载所有嵌入向量 ---
    if embeddings_path and embeddings_path.exists():
        logging.info(f"正在从 [cyan]{embeddings_path}[/cyan] 加载已缓存的嵌入向量...")
        all_embeddings = np.load(embeddings_path)
        logging.info(f"嵌入向量加载完毕，形状为: [green]{all_embeddings.shape}[/green]")
    else:
        # 只有在需要重新生成 embeddings 时，才加载验证集和 tokenizer
        logging.info("正在加载验证集和Tokenizer以生成嵌入向量...")
        logging.info(
            f"TEI请求参数: workers={tei_workers}, timeout={tei_timeout}s, "
            f"max_retries={tei_max_retries}, retry_base_delay={tei_retry_base_delay}s"
        )
        validation_dataset = load_from_disk(str(validation_dataset_pool_path))
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
        instruction = "Represent this LLVM IR for searching for similar functions:"
        all_embeddings = generate_embeddings_with_tei(
            validation_dataset,
            batch_size,
            instruction,
            tei_endpoint,
            tokenizer,
            max_length,
            tei_workers=tei_workers,
            tei_timeout=tei_timeout,
            tei_max_retries=tei_max_retries,
            tei_retry_base_delay=tei_retry_base_delay,
        )
        logging.info(f"嵌入向量生成完毕，形状为: [green]{all_embeddings.shape}[/green]")

        if embeddings_path:
            logging.info(f"正在将新生成的嵌入向量缓存到 [cyan]{embeddings_path}[/cyan]...")
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, all_embeddings)
            logging.info("缓存完成。")

    total_size = len(all_embeddings)

    with open(validation_positive_map_path, 'rb') as f:
        positive_map = pickle.load(f)
    positive_map, positive_map_stats = _sanitize_positive_map(positive_map, total_size)
    sanitized_count = sum(positive_map_stats.values())
    if sanitized_count > 0:
        logging.warning(
            "positive_map 已清洗: "
            f"invalid_anchor={positive_map_stats['invalid_anchor']}, "
            f"invalid_positive={positive_map_stats['invalid_positive']}, "
            f"self_positive={positive_map_stats['self_positive']}, "
            f"duplicate_positive={positive_map_stats['duplicate_positive']}, "
            f"empty_anchor={positive_map_stats['empty_anchor']}"
        )

    # --- 3. GPU内存预处理 ---
    if use_gpu:
        logging.info("正在将嵌入向量转移到GPU (bfloat16)...")
        all_embeddings_gpu = torch.as_tensor(all_embeddings, dtype=torch.bfloat16, device="cuda")
        logging.info(f"GPU内存使用: {all_embeddings_gpu.numel() * all_embeddings_gpu.element_size() / (1024**3):.2f} GB")
    else:
        all_embeddings_gpu = None


    # --- 4. 设置评估参数 ---
    requested_pool_sizes = sorted(set([2**i for i in range(1, 14)] + [100, 10000]))
    try:
        k_values = sorted({int(k.strip()) for k in ks_str.split(',') if k.strip()})
    except ValueError as exc:
        console.print(f"[bold red]错误: --ks 参数格式不合法: {ks_str}[/bold red]")
        raise typer.Exit(code=1) from exc
    if not k_values or any(k <= 0 for k in k_values):
        console.print(f"[bold red]错误: --ks 必须是正整数列表，当前值为: {ks_str}[/bold red]")
        raise typer.Exit(code=1)
    results = {}

    total_size = len(all_embeddings)
    pos_offsets = np.zeros(total_size + 1, dtype=np.int64)
    pos_flat_list = []
    for idx in range(total_size):
        positives = positive_map.get(idx, [])
        pos_offsets[idx + 1] = pos_offsets[idx] + len(positives)
        pos_flat_list.extend(positives)
    pos_flat = np.asarray(pos_flat_list, dtype=np.int64)

    all_possible_anchors = list(positive_map.keys())
    if not all_possible_anchors:
        console.print("[bold red]错误: 清洗后的 positive_map 中没有可评估的锚点。[/bold red]")
        raise typer.Exit(code=1)

    if eval_samples > 0 and eval_samples < len(all_possible_anchors):
        logging.info(f"将从 {len(all_possible_anchors):,} 个可能的锚点中随机采样 [yellow]{eval_samples:,}[/yellow] 个进行评估...")
        anchors_to_evaluate = random.sample(all_possible_anchors, eval_samples)
    else:
        logging.info(f"将评估所有 {len(all_possible_anchors):,} 个锚点...")
        anchors_to_evaluate = all_possible_anchors

    max_feasible_pool_size = min(total_size - len(positive_map[anchor_idx]) for anchor_idx in anchors_to_evaluate)
    pool_sizes = [pool_size for pool_size in requested_pool_sizes if pool_size <= max_feasible_pool_size]
    dropped_pool_sizes = [pool_size for pool_size in requested_pool_sizes if pool_size > max_feasible_pool_size]
    if dropped_pool_sizes:
        logging.warning(
            "以下 pool size 超出当前评估样本可支持上限，已跳过: "
            f"{dropped_pool_sizes} (max_feasible_pool_size={max_feasible_pool_size})"
        )
    if not pool_sizes:
        console.print(
            "[bold red]错误: 当前评估样本不足以构建任意有效检索池。"
            f" max_feasible_pool_size={max_feasible_pool_size}[/bold red]"
        )
        raise typer.Exit(code=1)


    # --- 5. 对不同的池大小进行评估 ---
    logging.info("开始对不同池大小进行批量GPU加速评估...")

    temp_results = {}
    for pool_size in pool_sizes:
        temp_results[pool_size] = {k: [0, 0] for k in k_values}
    total_mrr = {cutoff: [0.0, 0] for cutoff in MRR_CUTOFFS}
    total_mrr_by_pool = {pool_size: [0.0, 0] for pool_size in pool_sizes}


    # 使用自定义进度条显示评估速度
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[speed]:.1f} 锚点/秒"),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "正在评估...",
            total=len(anchors_to_evaluate),
            speed=0.0
        )

        processed_anchors = 0
        for i in range(0, len(anchors_to_evaluate), gpu_batch_size):
            anchor_batch = anchors_to_evaluate[i:i + gpu_batch_size]
            result, batch_mrr, batch_mrr_by_pool = process_anchor_batch_gpu(
                all_embeddings_gpu if use_gpu else all_embeddings,
                anchor_batch,
                pos_flat,
                pos_offsets,
                pool_sizes,
                k_values,
                use_gpu=use_gpu
            )
            # 累加结果
            for pool_size in pool_sizes:
                for k in k_values:
                    temp_results[pool_size][k][0] += result[pool_size][k][0]
                    temp_results[pool_size][k][1] += result[pool_size][k][1]
            for cutoff in MRR_CUTOFFS:
                total_mrr[cutoff][0] += batch_mrr[cutoff][0]
                total_mrr[cutoff][1] += batch_mrr[cutoff][1]
            for pool_size in pool_sizes:
                total_mrr_by_pool[pool_size][0] += batch_mrr_by_pool[pool_size][0]
                total_mrr_by_pool[pool_size][1] += batch_mrr_by_pool[pool_size][1]

            # 更新进度和速度
            processed_anchors += len(anchor_batch)
            elapsed = progress.tasks[0].elapsed or 0.001
            speed = processed_anchors / elapsed if elapsed > 0 else 0
            progress.update(task, advance=len(anchor_batch), speed=speed)

    # 将结果转换为百分比

    for pool_size in pool_sizes:
        results[pool_size] = {f"Recall@{k}": temp_results[pool_size][k][0] / temp_results[pool_size][k][1] if temp_results[pool_size][k][1] > 0 else 0 for k in k_values}

    # --- 6. 打印结果 ---
    console.rule("[bold green]评估结果[/bold green]")
    table = Table(title="Recall@K 在不同大小的检索池中的表现")
    table.add_column("Pool Size", justify="right", style="cyan")
    for k in k_values:
        table.add_column(f"Recall@{k}", justify="right", style="magenta")

    for pool_size, recalls in results.items():
        row_data = [f"{pool_size:,}"] + [f"{recalls[f'Recall@{k}']:.4f}" for k in k_values]
        table.add_row(*row_data)

    console.print(table)
    for cutoff in MRR_CUTOFFS:
        mrr_value = total_mrr[cutoff][0] / total_mrr[cutoff][1] if total_mrr[cutoff][1] > 0 else 0
        console.print(f"[bold green]MRR@{cutoff}: {mrr_value:.4f}[/bold green]")

    mrr_pool_table = Table(title="MRR@P 在不同大小检索池中的表现")
    mrr_pool_table.add_column("Pool Size", justify="right", style="cyan")
    mrr_pool_table.add_column("MRR@P", justify="right", style="green")
    for pool_size in pool_sizes:
        mrr_p = total_mrr_by_pool[pool_size][0] / total_mrr_by_pool[pool_size][1] if total_mrr_by_pool[pool_size][1] > 0 else 0
        mrr_pool_table.add_row(f"{pool_size:,}", f"{mrr_p:.4f}")
    console.print(mrr_pool_table)


if __name__ == "__main__":
    app()
