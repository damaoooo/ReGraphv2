"""wash truncation: the last kept token is EOS, and graph spans are clipped to the text before it."""
from pathlib import Path
import sys

import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DataProcess.wash import truncate_example

EOS = 3
L = 10  # kept positions 0..9; position 9 becomes EOS, so spans may reach position 8


def sample(length, ddg, cfg):
    return {'file_path': 'f.ll', 'input_ids': [2] + [100] * (length - 2) + [EOS], 'ddg_graph': ddg, 'cfg_graph': cfg}


def test_short_sample_is_unchanged():
    example = sample(L, [[1, 2, 3, 8]], [[1, 4, 5, 8, 0.5]])
    assert truncate_example(dict(example), L, EOS) == example


def test_long_sample_ends_with_eos_and_spans_stop_before_it():
    example = truncate_example(sample(30, ddg=[
        [1, 2, 3, 4],     # inside: kept as is
        [1, 2, 5, 9],     # ends on the EOS slot: clipped to 8
        [1, 2, 6, 20],    # crosses the cut: clipped to 8
        [1, 2, 9, 12],    # starts on the EOS slot: dropped
        [12, 14, 1, 2],   # source entirely past the cut: dropped
    ], cfg=[
        [1, 4, 5, 15, 0.25],   # crossing block keeps its edge and probability
        [5, 15, 16, 25, 1.0],  # target block entirely cut: dropped
    ]), L, EOS)
    assert len(example['input_ids']) == L and example['input_ids'][-1] == EOS
    assert example['ddg_graph'] == [[1, 2, 3, 4], [1, 2, 5, 8], [1, 2, 6, 8]]
    assert example['cfg_graph'] == [[1, 4, 5, 8, 0.25]]


def test_wash_command_applies_the_same_rule(tmp_path):
    from DataProcess.wash import main
    from Tokenizer.ir_tokenizer import load_tokenizer
    from Utils.utils import DEFAULT_TOKENIZER_PATH
    eos = load_tokenizer(DEFAULT_TOKENIZER_PATH).eos_token_id
    raw = sample(30, [[1, 2, 6, 20]], [[1, 4, 5, 15, 0.25]])
    raw['input_ids'][-1] = eos
    datasets.Dataset.from_list([raw, sample(5, [], [])]).save_to_disk(str(tmp_path / 'raw'))
    main(dataset_path=str(tmp_path / 'raw'), output_path=str(tmp_path / 'washed'),
         tokenizer_path=DEFAULT_TOKENIZER_PATH, max_seq_length=L)
    washed = datasets.load_from_disk(str(tmp_path / 'washed'))
    assert washed[0]['input_ids'][-1] == eos and len(washed[0]['input_ids']) == L
    assert washed[0]['ddg_graph'] == [[1, 2, 6, 8]] and washed[0]['cfg_graph'] == [[1, 4, 5, 8, 0.25]]
    assert washed[1]['input_ids'] == sample(5, [], [])['input_ids']
