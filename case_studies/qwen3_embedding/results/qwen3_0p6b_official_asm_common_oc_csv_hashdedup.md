# Qwen/Qwen3-Embedding-0.6B official on ASM common OC CSV hashdedup test_final_set

- Endpoint: `http://127.0.0.1:8080`
- Model ID: `Qwen/Qwen3-Embedding-0.6B official`
- Dataset pool: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool`
- Positive map: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl`
- Cache file: `/path/to/qwen-case-study/experiments/qwen3_0p6b_official_asm_common_oc_csv_hashdedup.bin.npy`
- Log file: `/path/to/qwen-case-study/experiments/eval_reports/logs/qwen3_0p6b_official_asm_common_oc_csv_hashdedup.log`
- Command: `python evaluate.py /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl --tei-endpoint http://127.0.0.1:8080 --ks 1,5,10,15,20,25,30,35,40,45,50 --batch-size 128 --max-length 2048 --tei-workers 4 --tei-timeout 180 --tei-max-retries 12 --tei-retry-base-delay 2 --gpu-batch-size 512 --eval-samples 0 --embeddings-path /path/to/qwen-case-study/experiments/qwen3_0p6b_official_asm_common_oc_csv_hashdedup.bin.npy --seed 42 --gpu`
- Embedding shape: `(137912, 1024)`
- Anchors evaluated: `107,395`
- MRR@10: `0.5939`
- MRR@30: `0.5973`

## Summary

| Setting | Recall@1 @ Pool 10,000 | MRR@P @ Pool 10,000 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-official ASM common-oc-csv-hashdedup | 0.5069 | 0.5989 | 0.5939 | 0.5973 |

## Recall@K

| Pool Size | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | Recall@35 | Recall@40 | Recall@45 | Recall@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9836 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9662 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9454 | 0.9945 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9214 | 0.9834 | 0.9951 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.8917 | 0.9702 | 0.9845 | 0.9911 | 0.9951 | 0.9981 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.8575 | 0.9522 | 0.9715 | 0.9797 | 0.9849 | 0.9884 | 0.9912 | 0.9935 | 0.9954 | 0.9968 | 0.9981 |
| 100 | 0.8345 | 0.9366 | 0.9614 | 0.9713 | 0.9768 | 0.9809 | 0.9844 | 0.9868 | 0.9888 | 0.9906 | 0.9922 |
| 128 | 0.8218 | 0.9266 | 0.9547 | 0.9659 | 0.9724 | 0.9763 | 0.9798 | 0.9825 | 0.9850 | 0.9869 | 0.9886 |
| 256 | 0.7843 | 0.8937 | 0.9298 | 0.9464 | 0.9560 | 0.9620 | 0.9667 | 0.9703 | 0.9726 | 0.9749 | 0.9767 |
| 512 | 0.7450 | 0.8579 | 0.8959 | 0.9175 | 0.9310 | 0.9403 | 0.9473 | 0.9526 | 0.9564 | 0.9596 | 0.9625 |
| 1,024 | 0.7032 | 0.8229 | 0.8611 | 0.8829 | 0.8977 | 0.9092 | 0.9181 | 0.9257 | 0.9318 | 0.9366 | 0.9407 |
| 2,048 | 0.6557 | 0.7869 | 0.8256 | 0.8475 | 0.8627 | 0.8743 | 0.8834 | 0.8918 | 0.8984 | 0.9043 | 0.9097 |
| 4,096 | 0.5995 | 0.7518 | 0.7896 | 0.8116 | 0.8270 | 0.8385 | 0.8484 | 0.8557 | 0.8632 | 0.8695 | 0.8749 |
| 8,192 | 0.5295 | 0.7158 | 0.7538 | 0.7757 | 0.7908 | 0.8027 | 0.8125 | 0.8201 | 0.8274 | 0.8332 | 0.8389 |
| 10,000 | 0.5069 | 0.7051 | 0.7442 | 0.7654 | 0.7807 | 0.7923 | 0.8021 | 0.8103 | 0.8172 | 0.8232 | 0.8284 |

## MRR@P

| Pool Size | MRR@P |
| ---: | ---: |
| 2 | 0.9918 |
| 4 | 0.9810 |
| 8 | 0.9667 |
| 16 | 0.9490 |
| 32 | 0.9264 |
| 64 | 0.8991 |
| 100 | 0.8795 |
| 128 | 0.8682 |
| 256 | 0.8344 |
| 512 | 0.7981 |
| 1,024 | 0.7601 |
| 2,048 | 0.7186 |
| 4,096 | 0.6719 |
| 8,192 | 0.6166 |
| 10,000 | 0.5989 |
