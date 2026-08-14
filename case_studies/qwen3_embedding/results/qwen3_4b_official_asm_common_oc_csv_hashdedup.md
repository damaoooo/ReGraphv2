# Qwen/Qwen3-Embedding-4B official on ASM common OC CSV hashdedup test_final_set

- Endpoint: `cache-only`
- Model ID: `Qwen/Qwen3-Embedding-4B official`
- Dataset pool: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool`
- Positive map: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl`
- Cache file: `/path/to/qwen-case-study/experiments/qwen3_4b_official_asm_common_oc_csv_hashdedup.bin.npy`
- Log file: `/path/to/qwen-case-study/experiments/eval_reports/logs/qwen3_4b_official_asm_common_oc_csv_hashdedup.log`
- Command: `python evaluate.py /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl --tei-endpoint cache-only --ks 1,5,10,15,20,25,30,35,40,45,50 --batch-size 16 --max-length 2048 --tei-workers 1 --tei-timeout 300 --tei-max-retries 12 --tei-retry-base-delay 2 --gpu-batch-size 128 --eval-samples 0 --embeddings-path /path/to/qwen-case-study/experiments/qwen3_4b_official_asm_common_oc_csv_hashdedup.bin.npy --seed 42 --gpu`
- Embedding shape: `(137912, 2560)`
- Anchors evaluated: `107,395`
- MRR@10: `0.6684`
- MRR@30: `0.6712`

## Summary

| Setting | Recall@1 @ Pool 10,000 | MRR@P @ Pool 10,000 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-official ASM common-oc-csv-hashdedup | 0.5669 | 0.6724 | 0.6684 | 0.6712 |

## Recall@K

| Pool Size | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | Recall@35 | Recall@40 | Recall@45 | Recall@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9959 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9890 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9777 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9634 | 0.9980 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.9436 | 0.9934 | 0.9985 | 0.9993 | 0.9998 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.9204 | 0.9833 | 0.9946 | 0.9973 | 0.9985 | 0.9991 | 0.9994 | 0.9997 | 0.9998 | 0.9999 | 0.9999 |
| 100 | 0.9029 | 0.9742 | 0.9890 | 0.9945 | 0.9967 | 0.9977 | 0.9983 | 0.9989 | 0.9992 | 0.9994 | 0.9995 |
| 128 | 0.8930 | 0.9683 | 0.9853 | 0.9918 | 0.9949 | 0.9965 | 0.9975 | 0.9980 | 0.9985 | 0.9988 | 0.9992 |
| 256 | 0.8628 | 0.9481 | 0.9701 | 0.9802 | 0.9860 | 0.9896 | 0.9921 | 0.9939 | 0.9951 | 0.9960 | 0.9966 |
| 512 | 0.8285 | 0.9241 | 0.9501 | 0.9630 | 0.9709 | 0.9765 | 0.9806 | 0.9840 | 0.9863 | 0.9885 | 0.9899 |
| 1,024 | 0.7885 | 0.8980 | 0.9265 | 0.9416 | 0.9510 | 0.9582 | 0.9633 | 0.9675 | 0.9713 | 0.9744 | 0.9769 |
| 2,048 | 0.7393 | 0.8703 | 0.9004 | 0.9160 | 0.9277 | 0.9356 | 0.9423 | 0.9473 | 0.9514 | 0.9551 | 0.9585 |
| 4,096 | 0.6762 | 0.8413 | 0.8728 | 0.8899 | 0.9010 | 0.9098 | 0.9170 | 0.9230 | 0.9281 | 0.9322 | 0.9359 |
| 8,192 | 0.5944 | 0.8072 | 0.8440 | 0.8617 | 0.8738 | 0.8833 | 0.8905 | 0.8965 | 0.9015 | 0.9060 | 0.9099 |
| 10,000 | 0.5669 | 0.7967 | 0.8350 | 0.8539 | 0.8655 | 0.8751 | 0.8829 | 0.8889 | 0.8940 | 0.8983 | 0.9025 |

## MRR@P

| Pool Size | MRR@P |
| ---: | ---: |
| 2 | 0.9980 |
| 4 | 0.9943 |
| 8 | 0.9877 |
| 16 | 0.9786 |
| 32 | 0.9652 |
| 64 | 0.9479 |
| 100 | 0.9343 |
| 128 | 0.9263 |
| 256 | 0.9013 |
| 512 | 0.8726 |
| 1,024 | 0.8398 |
| 2,048 | 0.8009 |
| 4,096 | 0.7530 |
| 8,192 | 0.6928 |
| 10,000 | 0.6724 |
