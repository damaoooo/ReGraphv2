import logging
import pickle
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from transformers import set_seed
from datasets import Dataset as HFDataset, load_from_disk
from Tokenizer.ir_tokenizer import load_tokenizer

from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding
from Model.graph_utils import build_batched_span_graph_tensors
from Pretrain.pretrain_model import MoCoPretrainModel
from Pretrain.pretrain_config import PretrainConfig
import torch
from numba import njit, prange, types
from numba.typed import Dict as NumbaDict


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
            # 没有正样本，跳过该样本
            continue
        else:
            rand_idx = np.random.randint(start, end)
            positive_anchor_idx = pos_flat[rand_idx]
            exclude_arr = np.empty(pos_len + 1, dtype=np.int64)
            exclude_arr[:pos_len] = pos_flat[start:end]
            exclude_arr[pos_len] = anchor_idx
            exclude_arr.sort()

        mapped = _sample_excluding(total_size, exclude_arr, pool_size)
        if mapped.size != pool_size:
            # 候选池不足时，跳过该样本
            continue

        pools[i, 0] = positive_anchor_idx
        pools[i, 1:] = mapped

    return pools


    
    
class FunctionDataCollator:
    
    def __init__(self, tokenizer, max_length=2048, use_cfg: bool = True, use_ddg: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cfg = use_cfg
        self.use_ddg = use_ddg
        
    def __call__(self, batch: List):
        pad_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, pad_to_multiple_of=8)
        truncated_input_ids = []

        has_input_ids = len(batch) > 0 and 'input_ids' in batch[0]
        if has_input_ids:
            input_ids = [batch_item['input_ids'] for batch_item in batch]
        else:
            input_ids = [self.tokenizer(batch_item['text'], truncation=True, max_length=self.max_length)['input_ids'] for batch_item in batch]
        for input_id in input_ids:
            if len(input_id) > self.max_length:
                truncated_input_ids.append(input_id[:self.max_length-1] + [self.tokenizer.eos_token_id])
            else:
                truncated_input_ids.append(input_id)
        
        input_ids = pad_collator({'input_ids': truncated_input_ids})
        input_ids, attention_mask = input_ids['input_ids'], input_ids['attention_mask']
        cfg_graphs_raw = [batch_item['cfg_graph'] for batch_item in batch] if self.use_cfg else None

        seq_lengths = [int(mask.sum().item()) for mask in attention_mask]

        ddg_tensors = {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
        if self.use_ddg:
            ddg_graphs_input = [batch_item['ddg_graph'] for batch_item in batch]
            ddg_tensors = build_batched_span_graph_tensors(ddg_graphs_input, seq_lengths, include_edge_attr=False)

        cfg_tensors = {"node_spans": None, "node_batch": None, "edge_index": None, "edge_attr": None}
        if self.use_cfg:
            cfg_tensors = build_batched_span_graph_tensors(cfg_graphs_raw, seq_lengths, include_edge_attr=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ddg_node_spans": ddg_tensors["node_spans"],
            "ddg_node_batch": ddg_tensors["node_batch"],
            "ddg_edge_index": ddg_tensors["edge_index"],
            "cfg_node_spans": cfg_tensors["node_spans"],
            "cfg_node_batch": cfg_tensors["node_batch"],
            "cfg_edge_index": cfg_tensors["edge_index"],
            "cfg_edge_attr": cfg_tensors["edge_attr"],
        }
    
def get_model(
    model_path,
    max_seq_length=2048,
    embedding_size=768,
    use_cfg: bool = True,
    use_ddg: bool = True,
    tokenizer_path: str = None,
):
    """加载MoCo模型并返回encoder_q用于推理
    
    Args:
        model_path: checkpoint目录路径
        max_seq_length: 最大序列长度
        embedding_size: 对比学习嵌入向量维度（通常等于hidden_size）
    
    Returns:
        encoder_q: 用于推理的query encoder
    """
    import os
    
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = PretrainConfig.from_pretrained(model_path)
    else:
        config = PretrainConfig(max_seq_length=max_seq_length)
    config.max_seq_length = max_seq_length
    config.use_cfg = use_cfg
    config.use_ddg = use_ddg
    if tokenizer_path is not None:
        config.tokenizer_path = tokenizer_path
    
    # 2. 设置vocab_size（从tokenizer获取）
    # 这里需要确保与训练时使用的tokenizer一致
    tokenizer = load_tokenizer(config.tokenizer_path)
    config.vocab_size = len(tokenizer.get_vocab())
    
    # 3. 设置embedding_size（如果config中没有，手动添加）
    if not hasattr(config, 'embedding_size'):
        config.embedding_size = embedding_size
    
    # 4. 实例化MoCo模型
    moco_model = MoCoPretrainModel(config)
    
    # 5. 加载权重文件
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        console.print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        alt_checkpoint_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(alt_checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path} or {alt_checkpoint_path}"
            )
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed."
            ) from exc
        console.print(f"Loading checkpoint from: {alt_checkpoint_path}")
        state_dict = load_safetensors_file(alt_checkpoint_path, device="cpu")

    moco_model.load_state_dict(state_dict)
    console.print("[green]✓ Model loaded successfully[/green]")
    
    # 6. 在evaluation阶段只使用encoder_q进行推理
    return moco_model.encoder_q

def _load_dataset(dataset_source: Union[Path, HFDataset]) -> HFDataset:
    if isinstance(dataset_source, HFDataset):
        return dataset_source
    return load_from_disk(str(dataset_source))


def get_dataloader(
    dataset_source: Union[Path, HFDataset],
    tokenizer,
    batch_size: int = 64,
    max_length: int = 2048,
    use_cfg: bool = True,
    use_ddg: bool = True,
) -> DataLoader:
    dataset = _load_dataset(dataset_source)
    collator = FunctionDataCollator(
        tokenizer,
        max_length=max_length,
        use_cfg=use_cfg,
        use_ddg=use_ddg,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,  # 禁用多进程避免数据缓冲
        pin_memory=False,  # 禁用pin_memory减少内存占用
        persistent_workers=False,
        shuffle=False,
    )
    return dataloader
    


def generate_embeddings_with_model(
    dataset_source: Union[Path, HFDataset],
    batch_size: int,
    tokenizer,
    model_path: str,
    max_length: int = 2048,
    embedding_size: int = 768,
    use_cfg: bool = True,
    use_ddg: bool = True,
) -> np.ndarray:
    """
    使用本地模型为整个数据集生成嵌入向量。
    使用CUDA和bf16精度进行推理。
    使用memmap避免内存爆炸。
    """
    import psutil
    import os
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    
    # 加载和设置模型（encoder_q用于推理）
    model = get_model(
        model_path=model_path,
        max_seq_length=max_length,
        embedding_size=embedding_size,
        use_cfg=use_cfg,
        use_ddg=use_ddg,
    )
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 使用bfloat16精度 - 不手动转换模型，而是使用autocast
    use_bf16 = device.type == "cuda"
    
    # 创建DataLoader
    dataloader = get_dataloader(
        dataset_source,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        use_cfg=use_cfg,
        use_ddg=use_ddg,
    )

    # 计算总样本数和embedding维度，提前创建memmap
    total_samples = len(dataloader.dataset)
    embedding_dim = embedding_size
    
    # 使用memmap替代list，避免内存峰值翻倍
    import tempfile
    temp_memmap = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    temp_memmap_path = temp_memmap.name
    temp_memmap.close()
    
    all_embeddings = np.memmap(temp_memmap_path, dtype=np.float32, mode='w+', shape=(total_samples, embedding_dim))
    failed_batches = []  # 记录失败的batch信息
    
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
    import time
    
    def get_memory_usage():
        """获取当前进程的内存占用(MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[speed]:.1f} 函数/秒"),
        TextColumn("[yellow]{task.fields[memory]:.0f}MB"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("正在生成嵌入向量...", total=total_samples, speed=0.0, memory=0.0)
        
        start_time = time.time()
        processed_count = 0
        batch_idx = 0
        current_idx = 0  # 当前memmap写入位置
        graph_tensor_keys = (
            "ddg_node_spans",
            "ddg_node_batch",
            "ddg_edge_index",
            "cfg_node_spans",
            "cfg_node_batch",
            "cfg_edge_index",
            "cfg_edge_attr",
        )
        
        for batch in dataloader:
            batch_idx += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_inputs = {
                key: batch[key].to(device) if batch.get(key) is not None else None
                for key in graph_tensor_keys
            }
            
            try:
                with torch.inference_mode():
                    if use_bf16:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(
                                input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                return_dict=True,
                                **graph_inputs,
                            )
                            embeddings = outputs['embedding']
                    else:
                        outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            return_dict=True,
                            **graph_inputs,
                        )
                        embeddings = outputs['embedding']
                    
                    # 转换为CPU numpy数组
                    batch_embeddings = embeddings.cpu().float().numpy()
                    batch_size_actual = len(batch_embeddings)
                    
                    # 直接写入memmap，不保留在内存中
                    all_embeddings[current_idx:current_idx + batch_size_actual] = batch_embeddings
                    current_idx += batch_size_actual
                    
                    # 更新进度条和速度统计
                    processed_count += batch_size_actual
                    elapsed_time = time.time() - start_time
                    speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    memory_usage = get_memory_usage()
                    progress.update(task, advance=batch_size_actual, speed=speed, memory=memory_usage)
                    
                    del input_ids, attention_mask, graph_inputs
                    del outputs, embeddings, batch_embeddings, batch
                    
            except Exception as e:
                batch_size_current = len(input_ids)
                failed_batches.append({
                    'batch_idx': batch_idx,
                    'batch_size': batch_size_current,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                console.print(f"[bold yellow]⚠ Batch {batch_idx} 推理失败，已跳过 ({batch_size_current}个样本)[/bold yellow]")
                console.print(f"[yellow]  错误类型: {type(e).__name__}[/yellow]")
                console.print(f"[yellow]  错误信息: {str(e)}[/yellow]")
                # 清理失败batch的变量
                del input_ids, attention_mask, graph_inputs, batch

    # 输出失败统计信息
    if failed_batches:
        console.print(f"\n[bold yellow]⚠ 评估完成！共有 {len(failed_batches)} 个batch推理失败[/bold yellow]")
        console.print(f"[yellow]失败的样本总数: {sum(b['batch_size'] for b in failed_batches)}[/yellow]")
        console.print("\n[bold]失败batch详情:[/bold]")
        for failure in failed_batches:
            console.print(f"  Batch #{failure['batch_idx']}: {failure['error_type']} - {failure['error'][:100]}")
    else:
        console.print(f"\n[bold green]✓ 评估完成！所有batch推理成功[/bold green]")

    # embedding 生成完成后模型不再使用，显式释放以降低后续阶段显存峰值。
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    if current_idx == 0:
        console.print("[bold red]ERROR: 没有成功的batch，无法返回结果[/bold red]")
        # 清理临时文件
        os.unlink(temp_memmap_path)
        raise typer.Exit(code=1)
    
    # 将memmap内容转为普通numpy数组并返回（会自动清理临时文件）
    result = np.array(all_embeddings[:current_idx], dtype=np.float32)
    del all_embeddings
    os.unlink(temp_memmap_path)
    
    return result


def _build_eval_subset(
    dataset: HFDataset,
    positive_map: dict,
    eval_samples: int,
    pool_samples: int,
) -> tuple[HFDataset, dict, List[int]]:
    all_possible_anchors = list(positive_map.keys())
    if not all_possible_anchors:
        return dataset, positive_map, []

    if eval_samples > 0 and eval_samples < len(all_possible_anchors):
        anchors_to_evaluate = random.sample(all_possible_anchors, eval_samples)
    else:
        anchors_to_evaluate = all_possible_anchors

    if pool_samples <= 0 or pool_samples >= len(dataset):
        return dataset, positive_map, anchors_to_evaluate

    selected_global_indices = set(anchors_to_evaluate)
    for anchor_idx in anchors_to_evaluate:
        positives = positive_map.get(anchor_idx, [])
        if positives:
            selected_global_indices.add(random.choice(positives))

    remaining_budget = pool_samples - len(selected_global_indices)
    if remaining_budget > 0:
        all_indices = list(range(len(dataset)))
        excluded = selected_global_indices
        candidates = [idx for idx in all_indices if idx not in excluded]
        if remaining_budget < len(candidates):
            selected_global_indices.update(random.sample(candidates, remaining_budget))
        else:
            selected_global_indices.update(candidates)

    selected_indices = sorted(selected_global_indices)
    subset = dataset.select(selected_indices)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    subset_positive_map = {}
    for old_anchor in anchors_to_evaluate:
        if old_anchor not in old_to_new:
            continue
        new_anchor = old_to_new[old_anchor]
        mapped_positives = [
            old_to_new[pos_idx]
            for pos_idx in positive_map.get(old_anchor, [])
            if pos_idx in old_to_new
        ]
        if mapped_positives:
            subset_positive_map[new_anchor] = mapped_positives

    subset_anchors = []
    for idx in anchors_to_evaluate:
        if idx not in old_to_new:
            continue
        new_idx = old_to_new[idx]
        if new_idx in subset_positive_map:
            subset_anchors.append(new_idx)
    return subset, subset_positive_map, subset_anchors


def process_anchor_batch_gpu(all_embeddings, anchor_batch, pos_flat, pos_offsets, pool_sizes, k_values: List[int], use_gpu: bool = True) -> dict:
    """
    处理锚点批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for pool_size in pool_sizes:
        recalls[pool_size] = {
            "recall": {k: [0, 0] for k in k_values},
        }
    mrr_stats = {10: [0.0, 0], 30: [0.0, 0]}
    
    max_pool_size = max(pool_sizes)
    pool_size = max_pool_size - 1
    batch_size = len(anchor_batch)
    total_size = len(all_embeddings)
    # 并行构建采样池（CPU多核）
    anchor_batch_arr = np.asarray(anchor_batch, dtype=np.int64)
    pools = _build_pools_parallel(anchor_batch_arr, pos_flat, pos_offsets, total_size, pool_size)
    valid_mask = pools[:, 0] >= 0
    if not np.any(valid_mask):
        return recalls

    pools = pools[valid_mask]
    anchor_batch_arr = anchor_batch_arr[valid_mask]

    if use_gpu:
        device = all_embeddings.device
        anchor_idx_tensor = torch.as_tensor(anchor_batch_arr, device=device, dtype=torch.long)
        anchors = all_embeddings.index_select(0, anchor_idx_tensor)
    else:
        anchors = all_embeddings[anchor_batch_arr]
    if use_gpu:
        # 将索引放到GPU上，直接在GPU内存中gather，避免CPU->GPU大拷贝
        pool_idx_tensor = torch.as_tensor(pools, device=device, dtype=torch.long)
        embedding_pools = all_embeddings[pool_idx_tensor]
        anchor_emb = anchors.unsqueeze(1)
        similarities = torch.bmm(anchor_emb, embedding_pools.transpose(1, 2)).squeeze(1)
    else:
        embedding_pools = all_embeddings[pools]  # size: (batch_size, pool_size, embedding_dim)
        anchor_emb = anchors[:, np.newaxis, :]
        anchor_emb_t = torch.as_tensor(anchor_emb)
        embedding_pools_t = torch.as_tensor(embedding_pools)
        similarities = torch.bmm(anchor_emb_t, embedding_pools_t.transpose(1, 2)).squeeze(1)


        
    # 计算MRR（在最大pool中取排名，超过阈值则记为0）
    max_pool_size = similarities.shape[1]
    mrr_sim_slice = similarities[:, :max_pool_size]
    mrr_pos_scores = mrr_sim_slice[:, 0].unsqueeze(1)
    mrr_count_greater = (mrr_sim_slice > mrr_pos_scores).sum(dim=1)
    mrr_ranks = mrr_count_greater + 1
    for cutoff in (10, 30):
        mrr_cutoff = min(cutoff, max_pool_size)
        mrr_scores = torch.where(
            mrr_ranks <= mrr_cutoff,
            1.0 / mrr_ranks.float(),
            torch.zeros_like(mrr_ranks, dtype=torch.float),
        )
        mrr_stats[cutoff][0] += mrr_scores.sum().item()
        mrr_stats[cutoff][1] += mrr_scores.numel()

    # 计算Recall@K（不做全量排序，直接比较正样本得分排名）
    for pool_size in pool_sizes:
        sim_slice = similarities[:, :pool_size]
        pos_scores = sim_slice[:, 0].unsqueeze(1)
        # 统计比正样本更大的个数，个数 < k 即表示正样本进入Top-K
        count_greater = (sim_slice > pos_scores).sum(dim=1)
        for k in k_values:
            success = (count_greater < k).sum().item()
            total = count_greater.numel()
            assert success <= total, f"Success count {success} cannot be greater than total {total}."
            recalls[pool_size]["recall"][k][0] += success
            recalls[pool_size]["recall"][k][1] += total

    return recalls, mrr_stats


@app.command()
def main(
    validation_dataset_pool_path: Path = typer.Argument(..., help="验证集数据池的路径。", exists=True, dir_okay=True),
    validation_positive_map_path: Path = typer.Argument(..., help="验证集正样本映射.pkl文件路径。", exists=True, file_okay=True),
    model_path: Path = typer.Argument(..., help="训练好的模型路径。", exists=True, dir_okay=True),
    ks_str: str = typer.Option("1,5,10,15,20,25,30,35,40,45,50", "--ks", "-k", help="要评估的K值，以逗号分隔。"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="模型推理的批量大小。"),
    max_length: int = typer.Option(2048, "--max-length", help="将'文本'整体截断到的最大token长度。"),
    eval_samples: int = typer.Option(0, "--eval-samples", "-n", help="用于评估的随机锚点样本数量。"),
    pool_samples: int = typer.Option(0, "--pool-samples", help="快速评估时使用的小检索池大小。0 表示全量检索池。"),
    embeddings_path: Path = typer.Option(None, "--embeddings-path", "-e", help="用于保存/加载嵌入向量Numpy文件的路径。"),
    seed: int = typer.Option(42, "--seed", "-s", help="用于负采样的随机种子。"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="是否使用GPU加速计算。"),
    gpu_batch_size: int = typer.Option(512, "--gpu-batch-size", help="GPU批量处理的锚点数量。"),
    tokenizer_path: Path = typer.Option(None, "--tokenizer-path", help="Tokenizer文件路径，如果与模型路径不同。"),
    use_cfg: bool = typer.Option(True, "--cfg/--no-cfg", help="是否启用 CFG 图分支。"),
    use_ddg: bool = typer.Option(True, "--ddg/--no-ddg", help="是否启用 DDG 图分支。"),
    use_bf16: bool = typer.Option(False, "--bf16", help="将嵌入以BF16加载到GPU，减少显存与带宽开销。"),
):
    """
    在验证集上评估模型的函数检索性能 (Recall@K)，使用本地模型生成嵌入，GPU加速相似度计算。
    """
    console.rule(f"[bold blue]开始使用本地模型进行评估[/bold blue]")
    set_seed(seed)
    console.print(f"[blue]Graph branches: cfg={use_cfg}, ddg={use_ddg}[/blue]")
    
    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但PyTorch CUDA不可用，将回退到CPU计算[/yellow]")
        use_gpu = False
    
    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")
    
    # --- 1. 加载数据和Tokenizer ---
    logging.info("正在加载数据和Tokenizer...")
    with open(validation_positive_map_path, 'rb') as f:
        positive_map = pickle.load(f)
    dataset = load_from_disk(str(validation_dataset_pool_path))
    
    # 加载Tokenizer
    if tokenizer_path:
        tokenizer = load_tokenizer(str(tokenizer_path))
    else:
        # 使用默认的tokenizer路径（假设与测试代码中相同）
        default_tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
        tokenizer = load_tokenizer(default_tokenizer_path)


    dataset_for_eval, positive_map_for_eval, anchors_to_evaluate = _build_eval_subset(
        dataset=dataset,
        positive_map=positive_map,
        eval_samples=eval_samples,
        pool_samples=pool_samples,
    )

    if pool_samples > 0:
        logging.info(
            f"快速评估模式: 检索池从 {len(dataset):,} 缩小到 {len(dataset_for_eval):,}，"
            f"锚点数 {len(anchors_to_evaluate):,}"
        )
        if embeddings_path:
            logging.warning("启用了 --pool-samples 时，请确保 embeddings_path 不与全量评估缓存混用。")
    else:
        logging.info(f"全量评估模式: 检索池 {len(dataset_for_eval):,}，锚点数 {len(anchors_to_evaluate):,}")

    # --- 2. 生成或加载所有嵌入向量 ---
    if embeddings_path and embeddings_path.exists():
        logging.info(f"正在从 [cyan]{embeddings_path}[/cyan] 加载已缓存的嵌入向量...")
        all_embeddings = np.load(embeddings_path, mmap_mode='r')  # 使用mmap_mode避免全部加载到内存
        logging.info(f"嵌入向量加载完毕，形状为: [green]{all_embeddings.shape}[/green]")
    else:
        all_embeddings = generate_embeddings_with_model(
            dataset_source=dataset_for_eval,
            batch_size=batch_size, 
            tokenizer=tokenizer, 
            model_path=str(model_path),
            max_length=max_length,
            use_cfg=use_cfg,
            use_ddg=use_ddg,
        )
        logging.info(f"嵌入向量生成完毕，形状为: [green]{all_embeddings.shape}[/green]")
        
        if embeddings_path:
            logging.info(f"正在将新生成的嵌入向量缓存到 [cyan]{embeddings_path}[/cyan]...")
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, all_embeddings)
            logging.info("缓存完成。")
            # 缓存后重新加载为memmap，释放内存
            del all_embeddings
            all_embeddings = np.load(str(embeddings_path), mmap_mode='r')
            logging.info(f"已将嵌入向量切换为内存映射模式 (mmap)")

    # --- 3. GPU内存预处理 ---
    if use_gpu:
        logging.info("正在将嵌入向量转移到GPU...")
        # 一次性把全部嵌入加载到GPU，避免每个batch的CPU->GPU拷贝
        # memmap在mmap_mode='r'下是只读的，先拷贝为可写数组再转成Tensor
        target_dtype = torch.bfloat16 if use_bf16 else torch.float32
        all_embeddings_gpu = torch.from_numpy(np.array(all_embeddings, copy=True)).to("cuda", dtype=target_dtype, non_blocking=False)
        dtype_label = "BF16" if use_bf16 else "FP32"
        logging.info(f"[green]✓ 已将全部嵌入加载到GPU ({dtype_label}): {all_embeddings_gpu.nbytes / (1024**3):.2f} GB[/green]")
    else:
        all_embeddings_gpu = None


    # --- 4. 设置评估参数 ---
    pool_sizes = [2**i for i in range(1, 14)] + [100, 10000]
    # Sort it
    pool_sizes = sorted(pool_sizes)
    k_values = sorted([int(k.strip()) for k in ks_str.split(',')])
    max_k = max(k_values)
    results = {}
    
    total_size = len(all_embeddings)
    # 预处理正样本映射为扁平数组，便于numba并行访问
    pos_offsets = np.zeros(total_size + 1, dtype=np.int64)
    pos_flat_list = []
    for idx in range(total_size):
        positives = positive_map_for_eval.get(idx, [])
        pos_offsets[idx + 1] = pos_offsets[idx] + len(positives)
        pos_flat_list.extend(positives)
    pos_flat = np.asarray(pos_flat_list, dtype=np.int64)


    # --- 5. 对不同的池大小进行评估 ---
    logging.info("开始对不同池大小进行批量GPU加速评估...")
    
    temp_results = {}
    for pool_size in pool_sizes:
        temp_results[pool_size] = {
            "recall": {k: [0, 0] for k in k_values},
        }
    total_mrr = {10: [0.0, 0], 30: [0.0, 0]}
    
    
    for i in track(range(0, len(anchors_to_evaluate), gpu_batch_size), description="正在评估..."):
        anchor_batch = anchors_to_evaluate[i:i + gpu_batch_size]
        result, batch_mrr = process_anchor_batch_gpu(
            all_embeddings_gpu if use_gpu else all_embeddings,
            anchor_batch,
            pos_flat,
            pos_offsets,
            pool_sizes,
            k_values,
            use_gpu=use_gpu,
        )
        # 累加结果
        for pool_size in pool_sizes:
            for k in k_values:
                temp_results[pool_size]["recall"][k][0] += result[pool_size]["recall"][k][0]
                temp_results[pool_size]["recall"][k][1] += result[pool_size]["recall"][k][1]
        for cutoff in (10, 30):
            total_mrr[cutoff][0] += batch_mrr[cutoff][0]
            total_mrr[cutoff][1] += batch_mrr[cutoff][1]
                
    # 将结果转换为百分比
    
    for pool_size in pool_sizes:
        recalls_result = {
            f"Recall@{k}": temp_results[pool_size]["recall"][k][0] / temp_results[pool_size]["recall"][k][1]
            if temp_results[pool_size]["recall"][k][1] > 0
            else 0
            for k in k_values
        }
        results[pool_size] = recalls_result

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
    mrr10 = total_mrr[10][0] / total_mrr[10][1] if total_mrr[10][1] > 0 else 0
    mrr30 = total_mrr[30][0] / total_mrr[30][1] if total_mrr[30][1] > 0 else 0
    console.print(f"[bold green]MRR@10: {mrr10:.4f}[/bold green]")
    console.print(f"[bold green]MRR@30: {mrr30:.4f}[/bold green]")


if __name__ == "__main__":
    app()
