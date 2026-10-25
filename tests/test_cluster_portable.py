import json
import pickle
import subprocess
import sys

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from DataProcess.select_samples import select_samples
from Pretrain.split_train_validation import parse_binary_directory
from Scripts.split_functions import load_selection


def test_selection_file(tmp_path):
    path = tmp_path / 'selected.json'
    path.write_text(json.dumps({'functions': ['f', 'g']}))
    assert load_selection(path) == {'f', 'g'}


def test_local_flowchart_selection(tmp_path):
    stats = tmp_path / 'stats'
    lift = tmp_path / 'lift'
    stats.mkdir()
    lift.mkdir()
    (stats / 'binary.flowchart.json').write_text(json.dumps({'functions': [
        {'fva': 16, 'bb_num': 5, 'hashopcodes': 'abc'},
        {'fva': 32, 'bb_num': 1, 'hashopcodes': 'def'}]}))
    (lift / 'binary.ll.lift.json').write_text(json.dumps({'status': 'ok', 'functions': [
        {'ea': '0x10', 'llvm_name': 'f', 'status': 'lifted', 'segment': '.text'},
        {'ea': '0x20', 'llvm_name': 'g', 'status': 'lifted', 'segment': '.text'}]}))
    output = tmp_path / 'selected'
    subprocess.run([sys.executable, 'Scripts/select_functions.py', str(stats), str(lift),
                    str(output), '--workers', '1'], check=True, capture_output=True)
    assert json.loads((output / 'binary.ll.select.json').read_text())['functions'] == ['f']


def test_xm_preserves_old_naming():
    name = 'arm32-gcc-4.8-O0_clambc'
    assert parse_binary_directory(name, 'dataset1-xm') == ('clambc', 'O0')
    assert parse_binary_directory(name, 'dataset1') == ('arm32-gcc-4.8_clambc', 'O0')


def test_sample_selection(tmp_path):
    paths = []
    for level in ['O0', 'O2']:
        directory = tmp_path / 'train' / 'curl' / f'arm32-gcc-4.8-{level}_curl_functions'
        directory.mkdir(parents=True)
        (directory / 'function_map.csv').write_text('OriginalFunctionName,HashedFileName\nf,f.ll\ng,g.ll\n')
        paths += [str(directory / 'f.ll'), str(directory / 'g.ll')]
    frame = pd.DataFrame({'file_path': paths, 'ir_digest': [b'a', b'b', b'c', b'b']})
    assert select_samples({'train': frame}, 'dataset1-xm')['train'].tolist() == [True, False, True, False]


def test_qwen_export_alignment(tmp_path):
    src = tmp_path / 'level' / 'test_final_set'
    src.mkdir(parents=True)
    module = 'test/curl/arm32-gcc-4.8-O0_curl'
    pool = datasets.Dataset.from_dict({
        'file_path': [f'/old/functions/{module}_functions/{n}.ll' for n in ['b', 'a']],
        'function_name': ['b', 'a'], 'original_idx': [0, 1],
        'opt_level': ['O0', 'O0'], 'origin_binary_name': ['curl', 'curl'],
        'input_ids': [[1], [2]],
    })
    pool.save_to_disk(str(src / 'train_dataset_pool'))
    datasets.Dataset.from_dict({'anchor_idx': [1, 0]}).save_to_disk(str(src / 'train_task_dataset'))
    mapping = pickle.dumps({0: [1], 1: [0]})
    (src / 'train_positive_map.pkl').write_bytes(mapping)
    text_path = tmp_path / 'text' / (module + '.parquet')
    text_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'function_name': ['a', 'b'], 'text': ['ret void', 'ret i32 0']}), text_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')), unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path / 'tokenizer')
    command = [sys.executable, '-m', 'DataProcess', 'export-qwen', '--level-dir', str(src.parent),
               '--split', 'test', '--text-roots', str(tmp_path / 'text'), '--qwen-tokenizer',
               str(tmp_path / 'tokenizer'), '--out', str(tmp_path / 'out'), '--workers', '1']
    subprocess.run(command, check=True, capture_output=True, text=True)
    target = tmp_path / 'out' / 'test_final_set'
    result = datasets.load_from_disk(str(target / 'train_dataset_pool'))
    assert list(result['text']) == ['ret i32 0', 'ret void']
    assert list(result['file_path']) == list(pool['file_path'])
    assert (target / 'train_positive_map.pkl').read_bytes() == mapping
    assert list(datasets.load_from_disk(str(target / 'train_task_dataset'))['anchor_idx']) == [1, 0]
    assert subprocess.run(command, capture_output=True).returncode != 0
