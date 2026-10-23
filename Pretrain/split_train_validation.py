import json
import random
import re
from enum import Enum
from datasets import load_dataset
import datasets
from tqdm import tqdm
import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import combinations
import os
from typing import Dict, List, Any, Optional, Tuple
import pickle
import typer


OPT_LEVEL = r'O0|O1|O2|O3|Os|Og|Oz|Ofast'
# Per dataset: how a binary's directory name (without `_functions`) splits into its build variant and its
# origin, the name with that variant removed. Functions that share (origin, function name) are positives;
# a binary whose name does not follow its dataset's convention is left out.
NAMING_CONVENTIONS = {
    # <arch>-<compiler>-<version>-<opt>_<binary>, e.g. arm32-gcc-4.8-O0_clambc
    'dataset1': (re.compile(rf'^(?P<config>[A-Za-z0-9_]+-[A-Za-z]+-[0-9.]+)-(?P<variant>{OPT_LEVEL})_(?P<binary>.+)$'),
                 '{config}_{binary}'),
    # <arch>-<bits>_<project>-<version>-<opt or obfuscation>_<binary>, e.g. x86-64_coreutils-8.32-subobf_join
    'dataset2': (re.compile(r'^(?P<config>[A-Za-z0-9]+-(?:32|64)_.+?-[0-9][0-9.]*)-(?P<variant>[A-Za-z0-9]+)_(?P<binary>.+)$'),
                 '{config}_{binary}'),
    # BinaryCorp: <binary>-<opt>-<md5>, e.g. agensgraph-git-bloom.so-Os-3ad76dc09f4f95595f6989901b7f4dd7
    'binarycorp': (re.compile(rf'^(?P<binary>.+)-(?P<variant>{OPT_LEVEL})-[0-9a-f]{{32}}$'), '{binary}'),
}
Naming = Enum('Naming', {name: name for name in NAMING_CONVENTIONS}, type=str)


def parse_binary_directory(name: str, naming: str) -> Tuple[str, str] | None:
    """(origin, build variant) of a binary directory name, or None if it does not follow the convention."""
    pattern, origin = NAMING_CONVENTIONS[naming]
    match = pattern.match(name)
    return (origin.format(**match.groupdict()), match['variant']) if match else None


def split_dataset(original_dataset: datasets.Dataset, positive_map: Dict[str, List[int]], function_info_df: pd.DataFrame, ratio: float = 0.9, seed: int = 42) -> None:

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
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices_list)}
    
    # 添加 function name 信息到数据集
    print("添加 function name 信息到数据集...")
    
    # 按新索引顺序（即 valid_indices_list 的顺序）取出每个样本的信息
    info_by_old_idx = function_info_df.set_index('original_idx').loc[valid_indices_list]
    function_names = info_by_old_idx['function_name'].tolist()
    opt_levels = info_by_old_idx['opt_level'].tolist()
    origin_binary_names = info_by_old_idx['origin_binary_name'].tolist()
    
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

    # === 2. 找出所有的“朋友圈” ===
    # positive_map pairs every two functions of one (origin, function name) group and nothing across groups,
    # so the groups are exactly the connected components of the positive graph; split whole groups.
    print("按 (origin_binary_name, function_name) 分组（朋友圈）...")
    connected_components = sorted(
        pd.Series(np.arange(len(original_dataset)))
        .groupby([pd.Series(origin_binary_names), pd.Series(function_names)], sort=False)
        .agg(list),
        key=lambda group: group[0])
    print(f"找到了 {len(connected_components)} 个独立的“朋友圈”")

    # === 3. 在“朋友圈”的层面上进行划分 ===
    print("在“朋友圈”层面上进行随机划分...")
    random.Random(seed).shuffle(connected_components)

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
        

def build_positive_indices(dataset: datasets.Dataset, base_path: Optional[str], naming: str) -> Tuple[dict, pd.DataFrame]:
    """
    构建正样本索引映射
    :param datasets: 包含函数的datasets.Dataset对象
    :param base_path: function_map.csv 的根目录；None 表示用每个函数文件所在目录
    :param naming: NAMING_CONVENTIONS 中该数据集的命名规则
    :return: 正样本索引映射字典和包含function name的DataFrame
    """
    example_file = dataset[0]['file_path']
    df: pd.DataFrame = dataset.select_columns(['file_path']).to_pandas()
    df['original_idx'] = range(len(df))

    path_parts = df['file_path'].str.split('/')
    binary_directories = path_parts.str[-2].str.removesuffix('_functions')
    parsed = {name: parse_binary_directory(name, naming) for name in binary_directories.unique()}
    unmatched = sorted(name for name, result in parsed.items() if result is None)
    if unmatched:
        dropped = binary_directories.isin(unmatched).sum()
        print(f"Dropping {dropped} functions of {len(unmatched)} binaries not named per '{naming}', "
              f"e.g. {unmatched[:3]}")
    df['ll_name'] = path_parts.str[-1]
    df['dir_name'] = path_parts.str[-3] + "/" + path_parts.str[-2]
    df['opt_level'] = binary_directories.map(lambda name: parsed[name] and parsed[name][1])
    origins = binary_directories.map(lambda name: parsed[name] and parsed[name][0])
    df['origin_binary_name'] = (path_parts.str[-3] + "/" + origins).where(origins.notna())

    # Binaries named per the convention take part and must be fully mapped; the others were dropped above
    participating = df[df['origin_binary_name'].notna()]
    if participating.empty:
        raise ValueError(f"No binary in the dataset is named per '{naming}'; check --naming")
    if base_path is not None and "Dataset-2-IR" in base_path:
        # Base path add /../
        base_path = os.path.abspath(os.path.join(base_path, ".."))
    processed_groups, missing_maps, unmapped = [], [], []
    for dir_name, group_content in tqdm(participating.groupby('dir_name'), desc="Mapping functions to original names"):
        if base_path is None:
            # Task 3 writes function_map.csv next to the function files it extracts
            csv_path = os.path.join(os.path.dirname(group_content['file_path'].iloc[0]), 'function_map.csv')
        else:
            csv_path = os.path.join(base_path, dir_name, 'function_map.csv')
        if not os.path.isfile(csv_path):
            missing_maps.append(csv_path)
            continue
        # Read names verbatim: by default pandas turns a function named NA or nan into a missing value
        hash_map_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        hash_to_name_map = pd.Series(hash_map_df['OriginalFunctionName'].values, index=hash_map_df['HashedFileName']).to_dict()

        temp_df = group_content.copy()
        temp_df['function_name'] = temp_df['ll_name'].map(hash_to_name_map)
        unmapped += temp_df.loc[temp_df['function_name'].isna(), 'file_path'].tolist()
        processed_groups.append(temp_df)

    if missing_maps or unmapped:
        raise ValueError(
            f"Incomplete function maps: {len(missing_maps)} binaries have no function_map.csv (e.g. {missing_maps[:3]}), "
            f"{len(unmapped)} functions are missing from theirs (e.g. {unmapped[:3]}). Check --base-path, "
            "or rerun Task 3 for those binaries; samples are not dropped silently.")
    full_info_df = pd.concat(processed_groups)

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

    if not positive_map:
        raise ValueError(f"No positive pairs: no function occurs in two build variants of one binary under '{naming}'. "
                         "Check --naming and that the dataset holds several variants per binary.")
    return dict(positive_map), full_info_df


def main(
    dataset_path: str = typer.Argument(..., help="Path to the dataset directory"),
    base_path: Optional[str] = typer.Option(
        None, help="Root holding <project>/<binary>_functions/function_map.csv when the data moved since Task 3 "
                   "(default: each function file's own directory)"),
    train_ratio: float = typer.Option(0.9, help="Training set ratio (0.0-1.0)"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_dir: str = typer.Option(".", help="Output directory for generated files"),
    naming: Naming = typer.Option(..., help="Binary naming convention of the dataset (see NAMING_CONVENTIONS)"),
):
    """
    Split dataset into train and validation sets while preserving function similarity relationships.
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Resolve inputs before changing into the output directory, so relative paths keep their meaning
    dataset_path = os.path.abspath(dataset_path)
    base_path = os.path.abspath(base_path) if base_path else None

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
        positive_map, function_info_df = build_positive_indices(dataset, base_path, naming.value)
        
        # Split dataset
        typer.echo(f"Splitting dataset with train ratio: {train_ratio}")
        split_dataset(dataset, positive_map, function_info_df, train_ratio, random_seed)
        
        typer.echo("Dataset splitting completed successfully!")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    typer.run(main)