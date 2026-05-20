"""
Dataset features definition for HuggingFace datasets
"""
import datasets


def get_dataset_features():
    """Define the features structure for HuggingFace dataset"""
    return datasets.Features({
        'binary_name': datasets.Value('string'),
        'function_name': datasets.Value('string'),
        'file_path': datasets.Value('string'),
        'ddg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('int32'))
        ),
        'cfg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('float32'))
        ),
        'input_ids': datasets.Sequence(datasets.Value('int32')),
    })


def get_qwen_text_dataset_features():
    """Define the text-only dataset structure for Qwen/ReLL embedding evaluation."""
    return datasets.Features({
        'binary_name': datasets.Value('string'),
        'function_name': datasets.Value('string'),
        'file_path': datasets.Value('string'),
        'text': datasets.Value('string'),
        'token_len': datasets.Value('int32'),
    })
