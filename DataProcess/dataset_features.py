"""
Dataset features definition for HuggingFace datasets
"""
import datasets


def get_dataset_features():
    """Define the features structure for HuggingFace dataset"""
    return datasets.Features({
        'file_path': datasets.Value('string'),
        'ddg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('int32'))
        ),
        'cfg_graph': datasets.Sequence(
            datasets.Sequence(datasets.Value('float32'))
        ),
        'input_ids': datasets.Sequence(datasets.Value('int32')),
    })
