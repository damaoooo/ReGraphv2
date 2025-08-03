import logging
import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
import requests
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
# --- 核心修正：引入 AutoTokenizer ---
from transformers import AutoTokenizer, set_seed, AutoModel
from datasets import load_from_disk
from Tokenizer.ir_tokenizer import load_tokenizer

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding
import torch.nn.functional as F
from Pretrain.pretrain_model import BinDebertaV2ModelForPretrain
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
try:
    import cupy as cp
    GPU_AVAILABLE = True
    console.print("[green]✓ GPU加速已启用 (CuPy)[/green]")
except ImportError:
    GPU_AVAILABLE = False
    console.print("[yellow]⚠ 未安装CuPy，将使用CPU计算[/yellow]")
    
    
class FunctioNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class FunctionDataCollator:
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: List):
        pad_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, pad_to_multiple_of=8)
        truncated_input_ids = []
        input_ids = [batch_item['input_ids'] for batch_item in batch]
        for input_id in input_ids:
            if len(input_id) > self.max_length:
                truncated_input_ids.append(input_id[:self.max_length-1] + [self.tokenizer.eos_token_id])
            else:
                truncated_input_ids.append(input_id)
        
        input_ids = pad_collator({'input_ids': truncated_input_ids})
        input_ids, attention_mask = input_ids['input_ids'], input_ids['attention_mask']
        cfg_graphs = [batch_item['cfg_graph'] for batch_item in batch]
        ddg_graphs = [batch_item['ddg_graph'] for batch_item in batch]
        
        # Pad graphs
        cfg_graphs = self.pad_graph(cfg_graphs, feature_length=5)
        ddg_graphs = self.pad_graph(ddg_graphs)
        
        cfg_graphs = torch.tensor(cfg_graphs, dtype=torch.long)
        ddg_graphs = torch.tensor(ddg_graphs, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cfg_graphs": cfg_graphs,
            "ddg_graphs": ddg_graphs,
        }

    def pad_graph(self, graph, feature_length: int = 4):
        max_edges = max(len(g) for g in graph)
        padded_graphs = []
        for g in graph:
            padding_needed = max_edges - len(g)
            padded_graph = g + [[-1] * feature_length] * padding_needed
            padded_graphs.append(padded_graph)
        return padded_graphs
    
def get_model(model_path):
    model = BinDebertaV2ModelForPretrain.from_pretrained(model_path, trust_remote_code=True)
    return model

def get_dataloader(dataset_path: Path, tokenizer, batch_size: int = 64, max_length: int = 2048) -> DataLoader:
    dataset = load_from_disk(str(dataset_path))
    collator = FunctionDataCollator(tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 每次只处理一个样本
        collate_fn=collator,
        shuffle=False,
    )
    return dataloader
    


def generate_embeddings_with_model(dataset_path: Path, batch_size: int, tokenizer, model_path: str, max_length: int = 2048) -> np.ndarray:
    """
    使用本地模型为整个数据集生成嵌入向量。
    使用CUDA和bf16精度进行推理。
    """
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    
    # 加载和设置模型
    model = get_model(model_path=model_path)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    
    console.print(f"Model device: {next(model.parameters()).device}")
    console.print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # 创建DataLoader
    dataloader = get_dataloader(dataset_path, tokenizer, batch_size=batch_size, max_length=max_length)

    all_embeddings = []
    
    for batch in track(dataloader, description="正在通过本地模型生成嵌入向量..."):
        # 将所有输入数据移动到CUDA
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cfg_graphs = batch['cfg_graphs'].to(device)
        ddg_graphs = batch['ddg_graphs'].to(device)
        
        try:
            with torch.no_grad():
                # 使用模型生成嵌入
                outputs = model.bindeberta(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    cfg_adj_list=cfg_graphs if len(cfg_graphs.shape) > 2 else None,
                    ddg_adj_list=ddg_graphs if len(ddg_graphs.shape) > 2 else None,
                ).last_hidden_state[:, 0, :]  # 取CLS token的输出
                
                # L2归一化
                embeddings = F.normalize(outputs, p=2, dim=-1)
                
                # 转换为CPU numpy数组
                batch_embeddings = embeddings.cpu().float().numpy()
                all_embeddings.append(batch_embeddings)
                
        except Exception as e:
            console.print(f"[bold red]错误: 模型推理失败: {e}[/bold red]")
            raise typer.Exit(code=1)

    return np.vstack(all_embeddings)


def process_anchor_batch_gpu(all_embeddings, anchor_batch, positive_map, pool_sizes, k_values: List[int], use_gpu: bool = True) -> dict:
    """
    处理锚点批次，计算与所有嵌入向量的相似度，并返回Recall@K结果。
    使用GPU加速计算相似度。
    """
    recalls = {}
    for pool_size in pool_sizes:
        recalls[pool_size] = {}
        for k in k_values:
            recalls[pool_size][k] = [0, 0]  # 每次都创建新的列表
    
    anchors = all_embeddings[anchor_batch]
    max_pool_size = max(pool_sizes)
    pool_size = max_pool_size - 1
    pools = []
    batch_size = len(anchor_batch)
    
    for i in range(batch_size):
        anchor_idx = anchor_batch[i]
        positive = positive_map[anchor_idx]
        positive_anchor_idx = random.choice(positive)
        
        while True:
            # Sample the random Pool
            candidate_indices = np.random.choice(len(all_embeddings), size=pool_size, replace=False)
            candidate_indices_set = set(candidate_indices)
            if positive_anchor_idx in candidate_indices_set or anchor_idx in candidate_indices_set:
                continue
            else:
                batch_pool = np.concatenate(([positive_anchor_idx], candidate_indices))
                pools.append(batch_pool)
                break

    pools = np.array(pools)
    embedding_pools = all_embeddings[pools] # size: (batch_size, pool_size, embedding_dim)
    anchor_emb = anchors[:, np.newaxis, :]  # size: (batch_size, 1, embedding_dim)
    

    anchor_emb_gpu = cp.asarray(anchor_emb)
    embedding_pools_gpu = cp.asarray(embedding_pools)
    similarities = cp.einsum('bij,bkj->bik', anchor_emb_gpu, embedding_pools_gpu)
    similarities = cp.squeeze(similarities, axis=1)  # size: (batch_size, pool_size)


        
    # 计算Recall@K
    for pool_size in pool_sizes:
        top_indices = cp.argsort(cp.argsort(-similarities[:, :pool_size], axis=1), axis=1)[:, 0] + 1
        for k in k_values:
            success, total = 0, 0
            success = (top_indices <= k).sum()
            total = len(top_indices)
            assert success <= total, f"Success count {success} cannot be greater than total {total}."
            recalls[pool_size][k][0] += success
            recalls[pool_size][k][1] += total

    return recalls


@app.command()
def main(
    validation_dataset_pool_path: Path = typer.Argument(..., help="验证集数据池的路径。", exists=True, dir_okay=True),
    validation_positive_map_path: Path = typer.Argument(..., help="验证集正样本映射.pkl文件路径。", exists=True, file_okay=True),
    model_path: Path = typer.Argument(..., help="训练好的模型路径。", exists=True, dir_okay=True),
    ks_str: str = typer.Option("1,5,10,15,20,25,30,35,40,45,50", "--ks", "-k", help="要评估的K值，以逗号分隔。"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="模型推理的批量大小。"),
    max_length: int = typer.Option(2048, "--max-length", help="将'文本'整体截断到的最大token长度。"),
    eval_samples: int = typer.Option(0, "--eval-samples", "-n", help="用于评估的随机锚点样本数量。"),
    embeddings_path: Path = typer.Option(None, "--embeddings-path", "-e", help="用于保存/加载嵌入向量Numpy文件的路径。"),
    seed: int = typer.Option(42, "--seed", "-s", help="用于负采样的随机种子。"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="是否使用GPU加速计算。"),
    gpu_batch_size: int = typer.Option(512, "--gpu-batch-size", help="GPU批量处理的锚点数量。"),
    tokenizer_path: Path = typer.Option(None, "--tokenizer-path", help="Tokenizer文件路径，如果与模型路径不同。"),
):
    """
    在验证集上评估模型的函数检索性能 (Recall@K)，使用本地模型生成嵌入，GPU加速相似度计算。
    """
    console.rule(f"[bold blue]开始使用本地模型进行评估[/bold blue]")
    set_seed(seed)
    
    # GPU可用性检查
    if use_gpu and not GPU_AVAILABLE:
        console.print("[yellow]⚠ 请求使用GPU但CuPy不可用，将回退到CPU计算[/yellow]")
        use_gpu = False
    
    if use_gpu:
        console.print(f"[green]🚀 将使用GPU加速，批量大小: {gpu_batch_size}[/green]")
    else:
        console.print("[blue]💻 使用CPU计算[/blue]")
    
    # --- 1. 加载数据和Tokenizer ---
    logging.info("正在加载数据和Tokenizer...")
    with open(validation_positive_map_path, 'rb') as f:
        positive_map = pickle.load(f)
    
    # 加载Tokenizer
    if tokenizer_path:
        tokenizer = load_tokenizer(str(tokenizer_path))
    else:
        # 使用默认的tokenizer路径（假设与测试代码中相同）
        default_tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/output_tokenizer/llvm_ir_bpe.json"
        tokenizer = load_tokenizer(default_tokenizer_path)


    # --- 2. 生成或加载所有嵌入向量 ---
    if embeddings_path and embeddings_path.exists():
        logging.info(f"正在从 [cyan]{embeddings_path}[/cyan] 加载已缓存的嵌入向量...")
        all_embeddings = np.load(embeddings_path)
        logging.info(f"嵌入向量加载完毕，形状为: [green]{all_embeddings.shape}[/green]")
    else:
        all_embeddings = generate_embeddings_with_model(
            dataset_path=validation_dataset_pool_path, 
            batch_size=batch_size, 
            tokenizer=tokenizer, 
            model_path=str(model_path),
            max_length=max_length
        )
        logging.info(f"嵌入向量生成完毕，形状为: [green]{all_embeddings.shape}[/green]")
        
        if embeddings_path:
            logging.info(f"正在将新生成的嵌入向量缓存到 [cyan]{embeddings_path}[/cyan]...")
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, all_embeddings)
            logging.info("缓存完成。")

    # --- 3. GPU内存预处理 ---
    if use_gpu:
        logging.info("正在将嵌入向量转移到GPU...")
        all_embeddings_gpu = cp.asarray(all_embeddings)
        logging.info(f"GPU内存使用: {all_embeddings_gpu.nbytes / (1024**3):.2f} GB")
    else:
        all_embeddings_gpu = None


    # --- 4. 设置评估参数 ---
    pool_sizes = [2**i for i in range(1, 14)] + [100, 10000]
    # Sort it
    pool_sizes = sorted(pool_sizes)
    k_values = sorted([int(k.strip()) for k in ks_str.split(',')])
    max_k = max(k_values)
    results = {}
    
    all_possible_anchors = list(positive_map.keys())
    if eval_samples > 0 and eval_samples < len(all_possible_anchors):
        logging.info(f"将从 {len(all_possible_anchors):,} 个可能的锚点中随机采样 [yellow]{eval_samples:,}[/yellow] 个进行评估...")
        anchors_to_evaluate = random.sample(all_possible_anchors, eval_samples)
    else:
        logging.info(f"将评估所有 {len(all_possible_anchors):,} 个锚点...")
        anchors_to_evaluate = all_possible_anchors


    # --- 5. 对不同的池大小进行评估 ---
    logging.info("开始对不同池大小进行批量GPU加速评估...")
    
    temp_results = {}
    for pool_size in pool_sizes:
        temp_results[pool_size] = {k: [0, 0] for k in k_values}
    
    
    for i in track(range(0, len(anchors_to_evaluate), gpu_batch_size), description="正在评估..."):
        anchor_batch = anchors_to_evaluate[i:i + gpu_batch_size]
        result = process_anchor_batch_gpu(
            all_embeddings_gpu if use_gpu else all_embeddings,
            anchor_batch,
            positive_map,
            pool_sizes,
            k_values,
            use_gpu=use_gpu
        )
        # 累加结果
        for pool_size in pool_sizes:
            for k in k_values:
                temp_results[pool_size][k][0] += result[pool_size][k][0]
                temp_results[pool_size][k][1] += result[pool_size][k][1]
                
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


if __name__ == "__main__":
    app()
