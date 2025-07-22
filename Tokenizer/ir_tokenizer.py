from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer as tk
from tokenizers.processors import TemplateProcessing

def load_tokenizer(tokenizer_path: str):
    raw_tokenizer = tk.from_file(tokenizer_path)
    raw_tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", raw_tokenizer.token_to_id("<bos>")),
            ("<eos>", raw_tokenizer.token_to_id("<eos>")),
        ],
    )
    
    additional_special_tokens = [
        "<func>", "<bb>", "<var>", "<const>",
        "<MED_INT>", "<LARGE_INT>", "<HUGE_INT>",
        "<HEX_CONST>", "<FLOAT_CONST>", "<STRING_CONST>"
    ]
    

    # 加载您的 tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        
        # --- 明确定义所有特殊符号 ---
        bos_token='<bos>',
        eos_token='<eos>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>',
        
        # --- 设置别名以兼容BERT/DeBERTa家族的习惯 ---
        cls_token='<bos>',
        sep_token='<eos>',
        
        # --- 确保您的额外特殊符号也被注册 ---
        additional_special_tokens=additional_special_tokens,
    )

    return tokenizer


def validate_special_tokens(tokenizer):
    """验证所有特殊 token 是否正确加载到 tokenizer 中"""
    
    expected_tokens = [
        # 基础 tokens
        '<pad>', '<unk>', '<bos>', '<eos>', '<mask>',
        
        # LLVM IR 结构化 tokens  
        '<func>', '<bb>', '<var>', '<const>',
        
        # 常量类型 tokens
        '<MED_INT>', '<LARGE_INT>', '<HUGE_INT>',
        '<HEX_CONST>', '<FLOAT_CONST>', '<STRING_CONST>'
    ]
    
    print("\n=== Special Token Validation ===")
    missing_tokens = []
    
    for token in expected_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            missing_tokens.append(token)
            print(f"❌ {token}: Not found (maps to UNK)")
        else:
            print(f"✅ {token}: ID {token_id}")
    
    if missing_tokens:
        print(f"\n⚠️  Missing tokens: {missing_tokens}")
        print("These tokens were not found in the tokenizer vocabulary.")
        print("Make sure your BPE training included these tokens.")
        return False
    else:
        print("\n✅ All special tokens validated successfully!")
        return True


def test_tokenization(tokenizer, sample_text: str = None):
    """测试 tokenizer 对包含特殊 token 的文本的处理"""
    
    if sample_text is None:
        # 包含各种特殊 token 的示例文本
        sample_text = '''define i32 @func1(i32 %arg1) {
bb1:
  %var1 = add i32 %arg1, <LARGE_INT>
  %var2 = load i32, i32* <HEX_CONST>
  %var3 = call i32 @func2(<FLOAT_CONST>)
  ret i32 %var1
}'''
    
    print("\n=== Tokenization Test ===")
    print(f"Input text: {sample_text[:100]}...")
    
    # Tokenize
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    
    print(f"Number of tokens: {len(tokens)}")
    print(f"Token IDs: {token_ids}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Check if special tokens are preserved
    special_tokens_found = []
    for token in tokens:
        if token.startswith('<') and token.endswith('>'):
            special_tokens_found.append(token)
    
    if special_tokens_found:
        print(f"Special tokens found: {set(special_tokens_found)}")
    else:
        print("No special tokens found in tokenization")
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded[:100]}...")
    
    return tokens, token_ids, decoded


if __name__ == "__main__":
    # Example usage
    tokenizer_path = "/home/damaoooo/Downloads/regraphv2/DataProcess/output_tokenizer/llvm_ir_bpe.json"  # Replace with your actual path
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Validate special tokens
    validate_special_tokens(tokenizer)
    
    # Test tokenization
    test_tokenization(tokenizer)