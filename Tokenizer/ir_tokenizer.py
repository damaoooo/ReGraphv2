import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer as tk
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


BASE_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]


def _load_special_tokens_from_config(tokenizer_path: str) -> List[str]:
    config_path = Path(tokenizer_path).with_name(f"{Path(tokenizer_path).stem}_config.json")
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("special_tokens", [])
    except Exception:
        return []


def load_tokenizer(tokenizer_path: str, additional_special_tokens: Optional[List[str]] = None):
    raw_tokenizer = tk.from_file(tokenizer_path)
    bos_id = raw_tokenizer.token_to_id("<bos>")
    eos_id = raw_tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError(f"Tokenizer at {tokenizer_path} is missing <bos>/<eos> special tokens")

    raw_tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    if additional_special_tokens is None:
        configured_tokens = _load_special_tokens_from_config(tokenizer_path)
        additional_special_tokens = [
            token for token in configured_tokens if token not in BASE_SPECIAL_TOKENS
        ]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<bos>",
        sep_token="<eos>",
        additional_special_tokens=additional_special_tokens or [],
    )

    return tokenizer


def validate_special_tokens(tokenizer, expected_tokens: Optional[List[str]] = None):
    """Validate special tokens loaded in tokenizer."""
    if expected_tokens is None:
        expected_tokens = BASE_SPECIAL_TOKENS + list(tokenizer.additional_special_tokens)

    print("\n=== Special Token Validation ===")
    missing_tokens = []

    for token in expected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
            missing_tokens.append(token)
            print(f"missing {token}: maps to UNK")
        else:
            print(f"ok {token}: ID {token_id}")

    if missing_tokens:
        print(f"\nMissing tokens: {missing_tokens}")
        return False

    print("\nAll special tokens validated successfully.")
    return True


def test_tokenization(tokenizer, sample_text: str = None):
    """Test tokenizer with ASM text."""
    if sample_text is None:
        sample_text = """push rbp
mov rbp, rsp
sub rsp, 20h
cmp eax, [rbp+var_4]
jg 0x1305
call 0x1150"""

    print("\n=== Tokenization Test ===")
    print(f"Input text: {sample_text[:100]}...")

    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"Number of tokens: {len(tokens)}")
    print(f"Token IDs: {token_ids}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"Decoded text: {tokenizer.decode(token_ids)[:100]}...")

    return tokens, token_ids, tokenizer.decode(token_ids)


if __name__ == "__main__":
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/Tokenizer/asm_output_tokenizer/asm_bpe.json"
    tokenizer = load_tokenizer(tokenizer_path)
    validate_special_tokens(tokenizer)
    test_tokenization(tokenizer)
    print(len(tokenizer.get_vocab()))
