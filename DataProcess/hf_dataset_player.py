from datasets import load_from_disk
import pandas as pd
from typing import Optional, Dict, Any

class HuggingFaceDatasetPlayer:
    def __init__(self, dataset_path: str = "/home/damaoooo/Downloads/regraphv2/IR/hf_save"):
        """
        Initialize the dataset player with the path to the saved dataset.
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the dataset from disk."""
        try:
            self.dataset = load_from_disk(self.dataset_path)
            print(f"Dataset loaded successfully from {self.dataset_path}")
            self.show_basic_info()
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def show_basic_info(self):
        """Display basic information about the dataset."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        print("\n=== Dataset Basic Information ===")
        print(f"Dataset type: {type(self.dataset)}")
        
        if hasattr(self.dataset, 'keys'):
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
    
    def show_sample(self, split: Optional[str] = None, index: int = 0, num_samples: int = 1):
        """Show sample data from the dataset."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                # If dataset has splits but no split specified, use the first one
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            print(f"\n=== Sample Data (showing {num_samples} example(s) starting from index {index}) ===")
            for i in range(index, min(index + num_samples, len(data))):
                print(f"\nExample {i}:")
                example = data[i]
                for key, value in example.items():
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error showing sample: {e}")
    
    def browse_by_index(self, index: int, split: Optional[str] = None):
        """Browse a specific example by index."""
        self.show_sample(split=split, index=index, num_samples=1)
    
    def show_feature_info(self, split: Optional[str] = None):
        """Show detailed information about dataset features."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            print(f"\n=== Feature Information ===")
            for feature_name, feature_type in data.features.items():
                print(f"{feature_name}: {feature_type}")
        except Exception as e:
            print(f"Error showing feature info: {e}")
    
    def to_pandas(self, split: Optional[str] = None, max_rows: int = 1000) -> Optional[pd.DataFrame]:
        """Convert dataset to pandas DataFrame for easier exploration."""
        if self.dataset is None:
            print("No dataset loaded.")
            return None
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            # Limit the number of rows to avoid memory issues
            subset = data.select(range(min(len(data), max_rows)))
            df = subset.to_pandas()
            print(f"Converted to pandas DataFrame with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error converting to pandas: {e}")
            return None
    
    def search_examples(self, column: str, value: Any, split: Optional[str] = None, max_results: int = 10):
        """Search for examples where a specific column contains a value."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            matches = []
            for i, example in enumerate(data):
                if column in example and example[column] == value:
                    matches.append((i, example))
                    if len(matches) >= max_results:
                        break
            
            print(f"\n=== Search Results for {column}='{value}' ===")
            print(f"Found {len(matches)} matches (showing up to {max_results}):")
            for i, (index, example) in enumerate(matches):
                print(f"\nMatch {i+1} (index {index}):")
                for key, val in example.items():
                    print(f"  {key}: {val}")
        except Exception as e:
            print(f"Error searching examples: {e}")

    def analyze_missing_graphs(self, split: Optional[str] = None):
        """Analyze how many examples are missing DDG or CFG."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            total_examples = len(data)
            missing_ddg = 0
            missing_cfg = 0
            missing_both = 0
            has_both = 0
            
            print(f"\n=== Graph Analysis for {total_examples} examples ===")
            
            for i, example in enumerate(data):
                has_ddg = 'ddg_graph' in example and example['ddg_graph'] is not None and len(example['ddg_graph']) > 0
                has_cfg = 'cfg_graph' in example and example['cfg_graph'] is not None and len(example['cfg_graph']) > 0
                
                if not has_ddg and not has_cfg:
                    missing_both += 1
                elif not has_ddg:
                    missing_ddg += 1
                elif not has_cfg:
                    missing_cfg += 1
                else:
                    has_both += 1
            
            print(f"Total examples: {total_examples}")
            print(f"Missing DDG only: {missing_ddg} ({missing_ddg/total_examples*100:.2f}%)")
            print(f"Missing CFG only: {missing_cfg} ({missing_cfg/total_examples*100:.2f}%)")
            print(f"Missing both DDG and CFG: {missing_both} ({missing_both/total_examples*100:.2f}%)")
            print(f"Has both DDG and CFG: {has_both} ({has_both/total_examples*100:.2f}%)")
            print(f"Total missing DDG: {missing_ddg + missing_both} ({(missing_ddg + missing_both)/total_examples*100:.2f}%)")
            print(f"Total missing CFG: {missing_cfg + missing_both} ({(missing_cfg + missing_both)/total_examples*100:.2f}%)")
            
            return {
                'total': total_examples,
                'missing_ddg_only': missing_ddg,
                'missing_cfg_only': missing_cfg,
                'missing_both': missing_both,
                'has_both': has_both,
                'total_missing_ddg': missing_ddg + missing_both,
                'total_missing_cfg': missing_cfg + missing_both
            }
            
        except Exception as e:
            print(f"Error analyzing missing graphs: {e}")
            return None

    def show_examples_missing_graphs(self, missing_type: str = "ddg", split: Optional[str] = None, max_examples: int = 5):
        """Show examples that are missing specific graph types."""
        if self.dataset is None:
            print("No dataset loaded.")
            return
        
        try:
            if hasattr(self.dataset, 'keys') and split is None:
                split = list(self.dataset.keys())[0]
                data = self.dataset[split]
            elif split is not None:
                data = self.dataset[split]
            else:
                data = self.dataset
            
            missing_examples = []
            
            for i, example in enumerate(data):
                has_ddg = 'ddg_graph' in example and example['ddg_graph'] is not None and len(example['ddg_graph']) > 0
                has_cfg = 'cfg_graph' in example and example['cfg_graph'] is not None and len(example['cfg_graph']) > 0
                
                if missing_type.lower() == "ddg" and not has_ddg:
                    missing_examples.append((i, example))
                elif missing_type.lower() == "cfg" and not has_cfg:
                    missing_examples.append((i, example))
                elif missing_type.lower() == "both" and not has_ddg and not has_cfg:
                    missing_examples.append((i, example))
                
                if len(missing_examples) >= max_examples:
                    break
            
            print(f"\n=== Examples Missing {missing_type.upper()} (showing up to {max_examples}) ===")
            for i, (index, example) in enumerate(missing_examples):
                print(f"\nExample {i+1} (index {index}):")
                for key, value in example.items():
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:200]}...")
                    else:
                        print(f"  {key}: {value}")
                        
        except Exception as e:
            print(f"Error showing missing examples: {e}")

def main():
    """Main function to demonstrate usage."""
    # Initialize the dataset player
    player = HuggingFaceDatasetPlayer()
    
    # Analyze missing graphs for the entire dataset
    print("\n" + "="*50)
    print("DATASET STATISTICS:")
    print("="*50)
    
    result = player.analyze_missing_graphs()
    if result:
        print(f"\n📊 SUMMARY:")
        print(f"🔴 Files missing DDG: {result['total_missing_ddg']}")
        print(f"🔴 Files missing CFG: {result['total_missing_cfg']}")
        print(f"🔴 Files missing BOTH: {result['missing_both']}")


    # Analyze missing graphs
    player.analyze_missing_graphs()
    
    # Show examples missing DDG
    player.show_examples_missing_graphs("ddg", max_examples=2)
    
    # Show examples missing CFG
    player.show_examples_missing_graphs("cfg", max_examples=2)
    
    # Convert to pandas for easier exploration
    df = player.to_pandas(max_rows=100)
    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")


if __name__ == "__main__":
    main()
