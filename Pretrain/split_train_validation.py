import json
import random
import networkx as nx
from datasets import load_dataset
import datasets
from tqdm import tqdm
import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import combinations
import os
from typing import Dict, List, Any, Tuple
import pickle
import typer


OPT_LEVELS = {"O0", "O1", "O2", "O3", "Os", "Oz"}


def parse_binary_folder_name(folder_name: str) -> Tuple[str, str]:
    """Parse `<arch>-<compiler>-<ver>-<opt>_<binary>[_functions]` style folder names."""
    stem = folder_name[:-10] if folder_name.endswith("_functions") else folder_name

    parts = stem.split("-")
    if len(parts) < 4:
        return "unknown", stem

    trailing = "-".join(parts[3:])
    if "_" in trailing:
        maybe_opt, binary_name = trailing.split("_", 1)
        if maybe_opt in OPT_LEVELS and binary_name:
            return maybe_opt, binary_name

    return "unknown", trailing or stem


def split_dataset(original_dataset: datasets.Dataset, positive_map: Dict[str, List[int]], function_info_df: pd.DataFrame, ratio: float = 0.9) -> None:

    print("原始数据集总大小:", len(original_dataset))
    
    # Add columns to the dataset
    indices_column = np.arange(len(original_dataset))
    original_dataset = original_dataset.add_column("original_idx", column=indices_column)

    # 过滤掉没有 function name 的数据
    print("过滤掉没有 function name 的数据...")
    
    # 获取有 function name 的索引集合
    valid_indices = set(function_info_df['original_idx'].tolist())
    print(f"原始数据集大小: {len(original_dataset)}, 有 function name 的数据: {len(valid_indices)}")
    
    # 创建一个新的索引列表，只包含有 function name 的数据
    valid_indices_list = sorted(list(valid_indices))
    
    # 过滤原始数据集，只保留有 function name 的数据
    original_dataset = original_dataset.select(valid_indices_list)
    print(f"过滤后数据集大小: {len(original_dataset)}")
    
    # 重新创建索引映射，因为数据集已经被过滤
    new_indices_column = np.arange(len(original_dataset))
    original_dataset = original_dataset.remove_columns(["original_idx"])
    original_dataset = original_dataset.add_column("original_idx", new_indices_column)
    
    # 创建新的索引到原始索引的映射
    new_to_old_idx_map = {new_idx: old_idx for new_idx, old_idx in enumerate(valid_indices_list)}
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices_list)}
    
    # 添加 function name 信息到数据集
    print("添加 function name 信息到数据集...")
    
    # 创建一个映射：old_original_idx -> function_name
    idx_to_function_name = {}
    idx_to_opt_level = {}
    idx_to_origin_binary_name = {}
    
    for _, row in function_info_df.iterrows():
        original_idx = row['original_idx']
        idx_to_function_name[original_idx] = row['function_name']
        idx_to_opt_level[original_idx] = row['opt_level']
        idx_to_origin_binary_name[original_idx] = row['origin_binary_name']
    
    # 为过滤后的数据集创建 function name 列表
    function_names = []
    opt_levels = []
    origin_binary_names = []
    
    for new_idx in range(len(original_dataset)):
        old_idx = new_to_old_idx_map[new_idx]
        function_names.append(idx_to_function_name[old_idx])
        opt_levels.append(idx_to_opt_level[old_idx])
        origin_binary_names.append(idx_to_origin_binary_name[old_idx])
    
    # 将这些信息添加到数据集
    original_dataset = original_dataset.add_column("function_name", function_names)
    original_dataset = original_dataset.add_column("opt_level", opt_levels)
    original_dataset = original_dataset.add_column("origin_binary_name", origin_binary_names)
    
    # 更新 positive_map，将旧索引映射到新索引
    print("更新 positive_map 以适应新的索引...")
    updated_positive_map = {}
    for old_anchor, old_positives in positive_map.items():
        if old_anchor in old_to_new_idx_map:
            new_anchor = old_to_new_idx_map[old_anchor]
            new_positives = [old_to_new_idx_map[old_pos] for old_pos in old_positives if old_pos in old_to_new_idx_map]
            if new_positives:  # 只有当还有正样本时才添加
                updated_positive_map[new_anchor] = new_positives
    
    positive_map = updated_positive_map

    # === 2. 构建完整的相似关系图 ===
    print("构建关系图中...")
    G = nx.Graph()
    # 将所有函数作为节点添加到图中
    G.add_nodes_from(range(len(original_dataset)))

    # 根据positive_map添加边
    for anchor_idx_str, positive_indices in tqdm(positive_map.items()):
        anchor_idx = int(anchor_idx_str)
        for positive_idx in positive_indices:
            G.add_edge(anchor_idx, positive_idx)

    # === 3. 找出所有的“朋友圈”（图的连通分量） ===
    print("寻找连通分量（朋友圈）...")
    connected_components = list(nx.connected_components(G))
    print(f"找到了 {len(connected_components)} 个独立的“朋友圈”")

    # === 4. 在“朋友圈”的层面上进行划分 ===
    print("在“朋友圈”层面上进行随机划分...")
    random.seed(42)
    random.shuffle(connected_components)

    # 假设我们按 90% 训练集, 10% 验证集
    num_groups = len(connected_components)
    train_split_idx = int(num_groups * ratio)
    valid_split_idx = int(num_groups)

    train_groups = connected_components[:train_split_idx]
    validation_groups = connected_components[train_split_idx:valid_split_idx]

    # === 5. “解散”朋友圈，得到最终的函数索引列表 ===
    print("生成最终的索引列表...")
    train_indices = [idx for group in train_groups for idx in group]
    validation_indices = [idx for group in validation_groups for idx in group]

    print(f"训练集大小: {len(train_indices)} 个函数")
    print(f"验证集大小: {len(validation_indices)} 个函数")

    # === 6. 使用索引来创建最终的数据集 (可选，也可以直接在训练脚本里用索引) ===
    # datasets库的 .select() 方法可以高效地根据索引创建子集
    # final_train_dataset = original_dataset.select(train_indices)
    # final_validation_dataset = original_dataset.select(validation_indices)

    # 为了方便，可以直接将这些索引列表保存到文件
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
    }
    with open("split_indices.json", "w") as f:
        json.dump(split_indices, f)

    print("数据划分完成，索引已保存到 split_indices.json")

    # 把datasets也分别保存下来
    final_train_dataset = original_dataset.select(train_indices)
    final_validation_dataset = original_dataset.select(validation_indices)

    # final_train_dataset.save_to_disk('train_dataset')
    # final_validation_dataset.save_to_disk('validation_dataset')

    train_indices_set = set(train_indices)
    validation_indices_set = set(validation_indices)

    train_df_index = final_train_dataset.select_columns(['original_idx']).to_pandas()
    train_df_index['new_idx'] = train_df_index.index
    train_old_to_new_map = pd.Series(train_df_index.new_idx.values, index=train_df_index.original_idx).to_dict()

    validation_df_index = final_validation_dataset.select_columns(['original_idx']).to_pandas()
    validation_df_index['new_idx'] = validation_df_index.index
    validation_old_to_new_map = pd.Series(validation_df_index.new_idx.values, index=validation_df_index.original_idx).to_dict()

    def filter_and_translate_map(global_map, index_set, old_to_new_map):
        new_map_with_new_indices = {}
        for anchor_old_idx_str, positive_old_list in tqdm(global_map.items(), desc="Filtering and translating map"):
            anchor_old_idx = int(anchor_old_idx_str)

            if anchor_old_idx in index_set:
                # 过滤正样本列表
                filtered_positives_old = [p_idx for p_idx in positive_old_list if p_idx in index_set]

                if filtered_positives_old:
                    # 【关键翻译步骤】
                    anchor_new_idx = old_to_new_map[anchor_old_idx] 
                    translated_positives_new = [old_to_new_map[p_idx] for p_idx in filtered_positives_old]

                    new_map_with_new_indices[anchor_new_idx] = translated_positives_new
        return new_map_with_new_indices

    # 还需要把positive_map按照训练集和验证集进行划分
    train_positive_map = filter_and_translate_map(positive_map, train_indices_set, train_old_to_new_map)
    validation_positive_map = filter_and_translate_map(positive_map, validation_indices_set, validation_old_to_new_map)

    # 分别保存下来
    with open('train_positive_map.pkl', 'wb') as f:
        pickle.dump(train_positive_map, f)
    with open('validation_positive_map.pkl', 'wb') as f:
        pickle.dump(validation_positive_map, f)
    
    # dataset 里面有些没有正样本的函数，这些函数在positive_map中没有对应的键
    # 所以我们需要在datasets里面去掉这些函数
    train_function_set = set(train_positive_map.keys())
    validation_function_set = set(validation_positive_map.keys())
    
    # Convert every element in the set into int
    train_function_set = {int(x) for x in train_function_set}
    validation_function_set = {int(x) for x in validation_function_set}
    
    train_task_dataset = datasets.Dataset.from_dict({'anchor_idx': train_positive_map})
    validation_task_dataset = datasets.Dataset.from_dict({'anchor_idx': validation_positive_map})

    # 保存最终的任务数据集
    train_task_dataset.save_to_disk('train_task_dataset')
    validation_task_dataset.save_to_disk('validation_task_dataset')
    
    final_train_dataset.save_to_disk('train_dataset_pool')
    final_validation_dataset.save_to_disk('validation_dataset_pool')
        

def build_positive_indices(dataset: datasets.Dataset, base_path: str) -> Tuple[dict, pd.DataFrame]:
    """
    构建正样本索引映射
    :param datasets: 包含函数的datasets.Dataset对象
    :param base_path: 数据集的基础路径
    :return: 正样本索引映射字典和包含function name的DataFrame
    """
    df: pd.DataFrame = dataset.select_columns(['file_path']).to_pandas()
    df['original_idx'] = range(len(df))

    def extract_binary_info_vectorized(file_paths: pd.Series) -> pd.DataFrame:
        """从 ASM/LLVM 文件路径里提取可用于构造正样本的信息。"""

        def parse_single_file_path(file_path: str) -> Dict[str, str]:
            normalized = os.path.normpath(file_path)
            parts = normalized.split(os.sep)

            file_name = parts[-1]
            function_name = os.path.splitext(file_name)[0]

            folder_name = parts[-2] if len(parts) >= 2 else ""
            project_name = parts[-3] if len(parts) >= 3 else ""
            dir_name = os.path.join(project_name, folder_name) if project_name else folder_name

            opt_level, binary_name = parse_binary_folder_name(folder_name)
            origin_binary_name = binary_name

            if project_name:
                origin_binary_name = f"{project_name}/{origin_binary_name}"

            return {
                "file_name": file_name,
                "function_name": function_name,
                "opt_level": opt_level,
                "dir_name": dir_name,
                "origin_binary_name": origin_binary_name,
            }

        parsed_rows = [parse_single_file_path(file_path) for file_path in file_paths]
        return pd.DataFrame(parsed_rows)

    extracted_df = extract_binary_info_vectorized(df['file_path'])
    full_info_df = pd.concat([df.reset_index(drop=True), extracted_df.reset_index(drop=True)], axis=1)
    full_info_df.dropna(subset=['origin_binary_name', 'function_name'], inplace=True)

    print("Grouping by [origin_binary_name, function_name] to find correct positive pairs...")
    final_grouped = full_info_df.groupby(['origin_binary_name', 'function_name'])

    positive_map = defaultdict(list)

    for name, group in tqdm(final_grouped, desc="Generating pairs from correct groups"):
        if len(group) > 1:
            indices = group['original_idx'].tolist()
            
            for idx1, idx2 in combinations(indices, 2):
                positive_map[idx1].append(idx2)
                positive_map[idx2].append(idx1)

    return dict(positive_map), full_info_df


def main(
    dataset_path: str = typer.Argument(..., help="Path to the dataset directory"),
    base_path: str = typer.Option(".", help="Base path of the source dataset tree. Retained for CLI compatibility."),
    train_ratio: float = typer.Option(0.9, help="Training set ratio (0.0-1.0)"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_dir: str = typer.Option(".", help="Output directory for generated files")
):
    """
    Split dataset into train and validation sets while preserving function similarity relationships.
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Change to output directory
    original_cwd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    try:
        # Load dataset
        typer.echo(f"Loading dataset from: {dataset_path}")
        dataset = datasets.load_from_disk(dataset_path)
        
        # Build positive indices
        typer.echo("Building positive indices...")
        positive_map, function_info_df = build_positive_indices(dataset, base_path)
        
        # Split dataset
        typer.echo(f"Splitting dataset with train ratio: {train_ratio}")
        split_dataset(dataset, positive_map, function_info_df, train_ratio)
        
        typer.echo("Dataset splitting completed successfully!")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    typer.run(main)
