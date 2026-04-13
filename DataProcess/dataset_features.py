"""
Dataset features definition for Hugging Face datasets
"""
import datasets


def get_dataset_features():
    """Define the features structure for Hugging Face datasets."""
    return datasets.Features(
        {
            "file_path": datasets.Value("string"),
            "input_ids": datasets.Sequence(datasets.Value("int32")),
        }
    )
