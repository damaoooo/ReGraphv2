from typing import Any, Optional

import pandas as pd
from datasets import load_from_disk


class HuggingFaceDatasetPlayer:
    def __init__(self, dataset_path: str = "/home/damaoooo/Downloads/regraphv2/IR/hf_save"):
        """Initialize the dataset player with the path to the saved dataset."""
        self.dataset_path = dataset_path
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        """Load the dataset from disk."""
        try:
            self.dataset = load_from_disk(self.dataset_path)
            print(f"Dataset loaded successfully from {self.dataset_path}")
            self.show_basic_info()
        except Exception as exc:
            print(f"Error loading dataset: {exc}")

    def show_basic_info(self):
        """Display basic information about the dataset."""
        if self.dataset is None:
            print("No dataset loaded.")
            return

        print("\n=== Dataset Basic Information ===")
        print(f"Dataset type: {type(self.dataset)}")

        if hasattr(self.dataset, "keys"):
            print(f"Dataset splits: {list(self.dataset.keys())}")
            for split_name in self.dataset.keys():
                split_data = self.dataset[split_name]
                print(f"  {split_name}: {len(split_data)} examples")
                if len(split_data) > 0:
                    print(f"    Features: {list(split_data.features.keys())}")
        else:
            print(f"Dataset length: {len(self.dataset)}")
            if len(self.dataset) > 0:
                print(f"Features: {list(self.dataset.features.keys())}")

    def _get_split_data(self, split: Optional[str] = None):
        if self.dataset is None:
            return None
        if hasattr(self.dataset, "keys") and split is None:
            split = list(self.dataset.keys())[0]
            return self.dataset[split]
        if split is not None and hasattr(self.dataset, "keys"):
            return self.dataset[split]
        return self.dataset

    def show_sample(self, split: Optional[str] = None, index: int = 0, num_samples: int = 1):
        """Show sample data from the dataset."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return

        try:
            print(
                f"\n=== Sample Data (showing {num_samples} example(s) starting from index {index}) ==="
            )
            for row_index in range(index, min(index + num_samples, len(data))):
                print(f"\nExample {row_index}:")
                example = data[row_index]
                for key, value in example.items():
                    print(f"  {key}: {value}")
        except Exception as exc:
            print(f"Error showing sample: {exc}")

    def browse_by_index(self, index: int, split: Optional[str] = None):
        """Browse a specific example by index."""
        self.show_sample(split=split, index=index, num_samples=1)

    def show_feature_info(self, split: Optional[str] = None):
        """Show detailed information about dataset features."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return

        try:
            print("\n=== Feature Information ===")
            for feature_name, feature_type in data.features.items():
                print(f"{feature_name}: {feature_type}")
        except Exception as exc:
            print(f"Error showing feature info: {exc}")

    def to_pandas(self, split: Optional[str] = None, max_rows: int = 1000) -> Optional[pd.DataFrame]:
        """Convert dataset to pandas DataFrame for easier exploration."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return None

        try:
            subset = data.select(range(min(len(data), max_rows)))
            dataframe = subset.to_pandas()
            print(
                f"Converted to pandas DataFrame with {len(dataframe)} rows and {len(dataframe.columns)} columns"
            )
            return dataframe
        except Exception as exc:
            print(f"Error converting to pandas: {exc}")
            return None

    def search_examples(self, column: str, value: Any, split: Optional[str] = None, max_results: int = 10):
        """Search for examples where a specific column contains a value."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return

        try:
            matches = []
            for index, example in enumerate(data):
                if column in example and example[column] == value:
                    matches.append((index, example))
                    if len(matches) >= max_results:
                        break

            print(f"\n=== Search Results for {column}='{value}' ===")
            print(f"Found {len(matches)} matches (showing up to {max_results}):")
            for match_index, (index, example) in enumerate(matches, start=1):
                print(f"\nMatch {match_index} (index {index}):")
                for key, entry in example.items():
                    print(f"  {key}: {entry}")
        except Exception as exc:
            print(f"Error searching examples: {exc}")

    def analyze_token_lengths(self, split: Optional[str] = None):
        """Analyze input token lengths and missing-token rows."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return None

        try:
            total_examples = len(data)
            empty_examples = 0
            token_lengths = []

            print(f"\n=== Token Analysis for {total_examples} examples ===")

            for example in data:
                input_ids = example.get("input_ids") or []
                if not input_ids:
                    empty_examples += 1
                    continue
                token_lengths.append(len(input_ids))

            min_length = min(token_lengths) if token_lengths else 0
            max_length = max(token_lengths) if token_lengths else 0
            avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0

            print(f"Total examples: {total_examples}")
            print(f"Examples with empty input_ids: {empty_examples} ({empty_examples / total_examples * 100:.2f}%)")
            print(f"Examples with tokens: {len(token_lengths)} ({len(token_lengths) / total_examples * 100:.2f}%)")
            print(f"Min token length: {min_length}")
            print(f"Max token length: {max_length}")
            print(f"Average token length: {avg_length:.2f}")

            return {
                "total": total_examples,
                "empty_examples": empty_examples,
                "non_empty_examples": len(token_lengths),
                "min_token_length": min_length,
                "max_token_length": max_length,
                "avg_token_length": avg_length,
            }
        except Exception as exc:
            print(f"Error analyzing token lengths: {exc}")
            return None

    def show_examples_missing_tokens(
        self,
        split: Optional[str] = None,
        max_examples: int = 5,
    ):
        """Show examples that do not contain token ids."""
        data = self._get_split_data(split)
        if data is None:
            print("No dataset loaded.")
            return

        try:
            missing_examples = []
            for index, example in enumerate(data):
                if not (example.get("input_ids") or []):
                    missing_examples.append((index, example))
                if len(missing_examples) >= max_examples:
                    break

            print(f"\n=== Examples Missing input_ids (showing up to {max_examples}) ===")
            for example_index, (index, example) in enumerate(missing_examples, start=1):
                print(f"\nExample {example_index} (index {index}):")
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:200]}...")
                    else:
                        print(f"  {key}: {value}")
        except Exception as exc:
            print(f"Error showing examples missing input_ids: {exc}")


def main():
    """Main function to demonstrate usage."""
    player = HuggingFaceDatasetPlayer()

    print("\n" + "=" * 50)
    print("DATASET STATISTICS:")
    print("=" * 50)

    stats = player.analyze_token_lengths()
    if stats:
        print(f"\nEmpty examples: {stats['empty_examples']}")
        print(f"Average token length: {stats['avg_token_length']:.2f}")

    player.show_examples_missing_tokens(max_examples=2)

    dataframe = player.to_pandas(max_rows=100)
    if dataframe is not None:
        print(f"\nDataFrame shape: {dataframe.shape}")
        print(f"Column names: {list(dataframe.columns)}")


if __name__ == "__main__":
    main()
