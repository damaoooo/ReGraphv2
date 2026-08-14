# Qwen/Qwen3-Embedding-0.6B official on raw Oc LLVM IR CSV len128 hashdedup test_final_set

- Endpoint: `http://127.0.0.1:18082`
- Model ID: `Qwen/Qwen3-Embedding-0.6B official`
- Dataset pool: `/path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_dataset_pool`
- Positive map: `/path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_positive_map.pkl`
- Cache file: `/path/to/qwen-case-study/experiments/qwen3_0p6b_official_ir_raw_csv_len128_text_hashdedup.bin.npy`
- Log file: `/path/to/qwen-case-study/experiments/eval_reports/logs/qwen3_0p6b_official_ir_raw_csv_len128_text_hashdedup.log`
- Command: `conda run -n ml --no-capture-output python evaluate.py /path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_dataset_pool /path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_positive_map.pkl --tei-endpoint http://127.0.0.1:18082 --ks 1,5,10,15,20,25,30,35,40,45,50 --batch-size 128 --max-length 2048 --gpu-batch-size 512 --eval-samples 0 --embeddings-path /path/to/qwen-case-study/experiments/qwen3_0p6b_official_ir_raw_csv_len128_text_hashdedup.bin.npy --seed 42 --gpu`
- Embedding shape: `(208293, 1024)`
- Anchors evaluated: `164,988`
- MRR@10: `0.6685`
- MRR@30: `0.6708`

## Summary

| Setting | Recall@1 @ Pool 10,000 | MRR@P @ Pool 10,000 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-official raw Oc IR | 0.5733 | 0.6718 | 0.6685 | 0.6708 |

## Recall@K

| Pool Size | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | Recall@35 | Recall@40 | Recall@45 | Recall@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9830 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9686 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9535 | 0.9917 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9370 | 0.9810 | 0.9920 | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.9192 | 0.9696 | 0.9816 | 0.9877 | 0.9918 | 0.9956 | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.8989 | 0.9554 | 0.9706 | 0.9777 | 0.9818 | 0.9851 | 0.9876 | 0.9898 | 0.9919 | 0.9938 | 0.9957 |
| 100 | 0.8850 | 0.9450 | 0.9621 | 0.9700 | 0.9752 | 0.9788 | 0.9814 | 0.9835 | 0.9854 | 0.9871 | 0.9885 |
| 128 | 0.8768 | 0.9388 | 0.9565 | 0.9654 | 0.9710 | 0.9749 | 0.9779 | 0.9802 | 0.9820 | 0.9836 | 0.9851 |
| 256 | 0.8517 | 0.9211 | 0.9400 | 0.9505 | 0.9573 | 0.9620 | 0.9656 | 0.9686 | 0.9711 | 0.9731 | 0.9750 |
| 512 | 0.8214 | 0.9028 | 0.9228 | 0.9333 | 0.9409 | 0.9467 | 0.9512 | 0.9547 | 0.9577 | 0.9602 | 0.9623 |
| 1,024 | 0.7839 | 0.8829 | 0.9042 | 0.9157 | 0.9234 | 0.9293 | 0.9337 | 0.9378 | 0.9414 | 0.9444 | 0.9471 |
| 2,048 | 0.7364 | 0.8615 | 0.8849 | 0.8967 | 0.9051 | 0.9112 | 0.9163 | 0.9203 | 0.9237 | 0.9268 | 0.9293 |
| 4,096 | 0.6738 | 0.8370 | 0.8635 | 0.8765 | 0.8859 | 0.8920 | 0.8972 | 0.9018 | 0.9056 | 0.9086 | 0.9114 |
| 8,192 | 0.5975 | 0.8058 | 0.8397 | 0.8549 | 0.8645 | 0.8717 | 0.8769 | 0.8819 | 0.8858 | 0.8893 | 0.8925 |
| 10,000 | 0.5733 | 0.7949 | 0.8318 | 0.8484 | 0.8581 | 0.8657 | 0.8712 | 0.8759 | 0.8800 | 0.8835 | 0.8864 |

## MRR@P

| Pool Size | MRR@P |
| ---: | ---: |
| 2 | 0.9915 |
| 4 | 0.9816 |
| 8 | 0.9700 |
| 16 | 0.9568 |
| 32 | 0.9418 |
| 64 | 0.9247 |
| 100 | 0.9126 |
| 128 | 0.9055 |
| 256 | 0.8843 |
| 512 | 0.8596 |
| 1,024 | 0.8304 |
| 2,048 | 0.7946 |
| 4,096 | 0.7486 |
| 8,192 | 0.6907 |
| 10,000 | 0.6718 |
