from rich.progress import Progress
import os
import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from datasets import load_from_disk
import time

# Load the BPE tokenizer
tokenizer_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_tokenizer/llvm_ir_bpe.json"
dataset_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_corpus/ir_corpus"

def count_tokens_batch(examples, tokenizer):
    """Count tokens for a batch of examples using tokenizer"""
    # Assume the text field is called 'text' or 'ir_code' - we'll check the dataset schema
    text_field = None
    for key in examples.keys():
        if 'text' in key.lower() or 'ir' in key.lower() or 'code' in key.lower():
            text_field = key
            break
    
    if text_field is None:
        # Use the first text field we find
        text_field = list(examples.keys())[0]
    
    texts = examples[text_field]
    token_counts = []
    
    for text in texts:
        if text and isinstance(text, str):
            try:
                encoding = tokenizer.encode(text)
                token_counts.append(len(encoding.tokens))
            except:
                token_counts.append(0)
        else:
            token_counts.append(0)
    
    return {"token_count": token_counts}

def process_dataset_with_hf(dataset_path, tokenizer_path, sample_size=None):
    """Process dataset using HuggingFace Datasets with built-in multiprocessing"""
    print(f"Loading dataset from {dataset_path}...")
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Print dataset info to understand the schema
    print("Dataset schema:", dataset.features)
    if len(dataset) > 0:
        print("First example keys:", list(dataset[0].keys()))
    
    # Sample if requested
    if sample_size and len(dataset) > sample_size:
        print(f"Sampling {sample_size} examples from {len(dataset)} total")
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Process the dataset using HF's built-in multiprocessing
    print("Processing dataset with tokenizer...")
    
    def tokenize_function(examples):
        return count_tokens_batch(examples, tokenizer)
    
    # Use HuggingFace's map function with multiprocessing
    with Progress() as progress:
        # Create a progress bar
        num_examples = len(dataset)
        task = progress.add_task("[cyan]Tokenizing...", total=num_examples)
        
        processed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Process 1000 examples at a time
            num_proc=os.cpu_count(),  # Use multiple processes
            desc="Counting tokens"
        )
        
        progress.update(task, completed=num_examples)
    
    # Extract token counts
    token_counts = [count for count in processed_dataset["token_count"] if count > 0]
    
    print(f"Successfully processed {len(token_counts)} examples with valid token counts")
    return token_counts

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading and processing dataset using HuggingFace Datasets...")
    
    # Check dataset size first
    dataset = load_from_disk(dataset_path)
    dataset_size = len(dataset)
    print(f"Dataset contains {dataset_size} examples")
    
    # Add sampling option for very large datasets
    sample_size = None
    if dataset_size > 100000:  # If more than 100k examples
        print("Large dataset detected. Do you want to:")
        print("1. Process all examples (will take some time)")
        print("2. Process a random sample of 50,000 examples")
        print("3. Process a random sample of 10,000 examples")
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "2":
            sample_size = 50000
        elif choice == "3":
            sample_size = 10000
    
    # Process dataset using HuggingFace Datasets
    token_counts = process_dataset_with_hf(dataset_path, tokenizer_path, sample_size)

    elapsed_time = time.time() - start_time
    print(f"Successfully processed {len(token_counts)} examples with valid token counts")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    if len(token_counts) > 0:
        print(f"Examples per second: {len(token_counts) / elapsed_time:.2f}")
        print(f"Average tokens per example: {np.mean(token_counts):.2f}")
        print(f"Max tokens: {np.max(token_counts)}")
        print(f"Min tokens: {np.min(token_counts)}")
    
    if len(token_counts) == 0:
        print("No valid token counts found! Check your dataset schema.")
        exit(1)

    plt.figure(figsize=(10, 6))
    counts, bin_edges = np.histogram(token_counts, bins=500, range=(0, 5000), density=True)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]
    plt.plot(bin_edges[1:], cdf, color='blue', alpha=0.7)
    plt.title('Token Count CDF in IR Files (using BPE Tokenizer)')
    plt.xlabel('Token Count')
    plt.xlim(0, 5000)
    plt.ylabel('CDF')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("token_count_cdf.png", dpi=300)
    plt.show()
    
    percentiles = [0.9, 0.95, 0.98, 0.99]
    token_counts_sorted = np.sort(token_counts)
    for p in percentiles:
        idx = int(np.ceil(p * len(token_counts_sorted))) - 1
        print(f"Token count at {int(p*100)}% percentile: {token_counts_sorted[idx]}")