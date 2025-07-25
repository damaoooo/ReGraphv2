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


def split_dataset(original_dataset: datasets.Dataset, positive_map: Dict[str, List[int]]) -> None:

    print("原始数据集总大小:", len(original_dataset))
    
    # Add columns to the dataset
    indices_column = np.arange(len(original_dataset))
    original_dataset = original_dataset.add_column("original_idx", column=indices_column)

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
    
    train_task_dataset = dataset.from_dict({'anchor_idx': train_positive_map})
    validation_task_dataset = dataset.from_dict({'anchor_idx': validation_positive_map})
    
    # 保存最终的任务数据集
    train_task_dataset.save_to_disk('train_task_dataset')
    validation_task_dataset.save_to_disk('validation_task_dataset')
    
    final_train_dataset.save_to_disk('train_dataset_pool')
    final_validation_dataset.save_to_disk('validation_dataset_pool')
        

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

    full_info_df.dropna(subset=['origin_binary_name', 'function_name'], inplace=True)
    # -------------------------------------------------------------------

    # ===================================================================
    # === 核心修正：使用复合键进行分组 ===
    # ===================================================================
    print("Grouping by [origin_binary_name, function_name] to find correct positive pairs...")
    final_grouped = full_info_df.groupby(['origin_binary_name', 'function_name'])

    # 使用defaultdict可以简化代码
    positive_map = defaultdict(list)

    for name, group in tqdm(final_grouped, desc="Generating pairs from correct groups"):
        # 只要一个组里有多个成员（比如不同优化等级），它们就互为正样本
        if len(group) > 1:
            indices = group['original_idx'].tolist()
            
            # 【优化建议】使用itertools.combinations可以更高效、简洁地生成组内所有不重复的配对
            for idx1, idx2 in combinations(indices, 2):
                positive_map[idx1].append(idx2)
                positive_map[idx2].append(idx1)

    # 将defaultdict转换回普通字典
    return dict(positive_map)
        

if __name__ == "__main__":
    # 运行数据划分
    dataset = datasets.load_from_disk('/home/damaoooo/Downloads/regraphv2/IR/binary_save_truncated')
    positive_map = build_positive_indices(dataset)
    split_dataset(dataset, positive_map)
    print("数据划分和索引构建完成！")