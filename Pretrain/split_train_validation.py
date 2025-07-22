import json
import random
import networkx as nx
from datasets import load_dataset
import datasets
from tqdm import tqdm
import pandas as pd

import os
from typing import Dict, List, Any, Tuple


def split_dataset(original_dataset: datasets.Dataset, positive_map: Dict[str, List[int]]) -> None:

    print("原始数据集总大小:", len(original_dataset))

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

    # 假设我们按 90% 训练集, 5% 验证集, 5% 测试集来划分这些“朋友圈”
    num_groups = len(connected_components)
    train_split_idx = int(num_groups * 0.9)
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

    final_train_dataset.save_to_disk('train_dataset')
    final_validation_dataset.save_to_disk('validation_dataset')

    train_indices_set = set(train_indices)
    validation_indices_set = set(validation_indices)

    # 还需要把positive_map按照训练集和验证集进行划分
    train_positive_map = {}
    validation_positive_map = {}
    for k, v in tqdm(positive_map.items(), desc="划分positive_map"):
        if int(k) in train_indices_set:
            train_positive_map[k] = v

        if int(k) in validation_indices_set:
            validation_positive_map[k] = v

    # 分别保存下来
    with open('train_positive_map.json', 'w') as f:
        json.dump(train_positive_map, f)
    with open('validation_positive_map.json', 'w') as f:
        json.dump(validation_positive_map, f)


def build_positive_indices(dataset: datasets.Dataset) -> dict:
    """
    构建正样本索引映射
    :param datasets: 包含函数的datasets.Dataset对象
    :return: 正样本索引映射字典
    """
    example_file = dataset[0]['file_path']
    df: pd.DataFrame = dataset.select_columns(['file_path']).to_pandas()
    df['original_idx'] = range(len(df))
    base_path = "/home/damaoooo/Downloads/regraphv2/IR/BinaryCorp/small_train"

    def extract_binary_info(file_path: str) -> tuple[str, str, str]:
        ll_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        dir_name = os.path.basename(dir_name)
        origin_dir_name = dir_name.replace('_functions', '')
        binary_hash = origin_dir_name.split('-')[-1]
        opt_level = origin_dir_name.split('-')[-2]
        origin_binary_name = '-'.join(origin_dir_name.split('-')[:-2])

        return ll_name, opt_level, dir_name, origin_binary_name

    def extract_binary_info_vectorized(file_paths: pd.Series) -> pd.DataFrame:
        """向量化版本的信息提取"""
        # 使用字符串方法进行批量操作
        ll_names = file_paths.str.split('/').str[-1]
        dir_names = file_paths.str.split('/').str[-2]
        origin_dir_names = dir_names.str.replace('_functions', '')
        
        # 批量分割和提取
        split_parts = origin_dir_names.str.split('-')
        binary_hashes = split_parts.str[-1]
        opt_levels = split_parts.str[-2]
        origin_binary_names = split_parts.apply(lambda x: '-'.join(x[:-2]) if len(x) > 2 else '')
        
        return pd.DataFrame({
            'll_name': ll_names,
            'opt_level': opt_levels,
            'dir_name': dir_names,
            'origin_binary_name': origin_binary_names
        })

    # df[['ll_name', 'opt_level', 'dir_name', 'origin_binary_name']] = df['file_path'].apply(extract_binary_info).apply(pd.Series)
    df[['ll_name', 'opt_level', 'dir_name', 'origin_binary_name']] = extract_binary_info_vectorized(df['file_path'])

    

    grouped_by_dir_name = df.groupby('dir_name')
    processed_groups = []
    for dir_name, group_content in tqdm(grouped_by_dir_name, desc="Mapping functions to original names"):
        csv_path = os.path.join(base_path, dir_name, 'function_map.csv')
        try:
            hash_map_df = pd.read_csv(csv_path)
            hash_to_name_map = pd.Series(hash_map_df['OriginalFunctionName'].values, index=hash_map_df['HashedFileName']).to_dict()
        except FileNotFoundError:
            print(f"File {csv_path} not found, skipping.")
            continue

        temp_df = group_content.copy()
        temp_df['function_name'] = temp_df['ll_name'].map(hash_to_name_map)
        processed_groups.append(temp_df)

    full_info_df = pd.concat(processed_groups)

    final_grouped = full_info_df.groupby('function_name')

    positive_map = {}

    for func_name, group in tqdm(final_grouped, desc="Processing functions"):
        if group['dir_name'].nunique() > 1:
            # Find the original idx of the func_name
            indices = group['original_idx'].tolist()
            for index in indices:
                if index not in positive_map:
                    positive_map[index] = []
                new_indices = indices.copy()
                new_indices.remove(index)  # Remove the current index to avoid self-reference
                positive_map[index].extend(new_indices)
    # 去重
    for key in positive_map:
        positive_map[key] = list(set(positive_map[key]))

    return positive_map
        

if __name__ == "__main__":
    # 运行数据划分
    dataset = datasets.load_from_disk('/home/damaoooo/Downloads/regraphv2/IR/binary_save_truncated')
    positive_map = build_positive_indices(dataset)
    split_dataset(dataset, positive_map)
    print("数据划分和索引构建完成！")