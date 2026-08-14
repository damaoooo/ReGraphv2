# Qwen3-Embedding-0.6B ASM QLoRA (original ASM train) on ASM common OC CSV hashdedup test_final_set

- Endpoint: `http://127.0.0.1:18081`
- Model ID: `Qwen3-Embedding-0.6B ASM QLoRA (original ASM train)`
- Dataset pool: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool`
- Positive map: `/path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl`
- Cache file: `/path/to/qwen-case-study/experiments/qwen3_0p6b_asm_ft_original_train_asm_prompt.bin.npy`
- Log file: `/path/to/qwen-case-study/experiments/eval_reports/logs/qwen3_0p6b_asm_ft_original_train_asm_prompt.log`
- Command: `python /path/to/qwen-case-study/evaluate.py /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_dataset_pool /path/to/rell/IR/dataset-1-asm/test_final_set_common_oc_csv_hashdedup_qwen_text/train_positive_map.pkl --tei-endpoint http://127.0.0.1:18081 --ks 1,5,10,15,20,25,30,35,40,45,50 --batch-size 128 --max-length 2048 --gpu-batch-size 512 --eval-samples 0 --embeddings-path /path/to/qwen-case-study/experiments/qwen3_0p6b_asm_ft_original_train_asm_prompt.bin.npy --seed 42 --gpu`
- Embedding shape: `(137912, 1024)`
- Anchors evaluated: `107,395`
- MRR@10: `0.5179`
- MRR@30: `0.5215`

## Summary

| Setting | Recall@1 @ Pool 10,000 | MRR@P @ Pool 10,000 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-ft ASM original train | 0.4192 | 0.5231 | 0.5179 | 0.5215 |

## Recall@K

| Pool Size | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | Recall@35 | Recall@40 | Recall@45 | Recall@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9736 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9461 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9173 | 0.9899 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.8874 | 0.9689 | 0.9907 | 0.9992 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.8562 | 0.9453 | 0.9701 | 0.9832 | 0.9909 | 0.9960 | 0.9992 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.8233 | 0.9194 | 0.9470 | 0.9614 | 0.9709 | 0.9779 | 0.9835 | 0.9876 | 0.9911 | 0.9938 | 0.9960 |
| 100 | 0.8010 | 0.9005 | 0.9307 | 0.9462 | 0.9562 | 0.9638 | 0.9699 | 0.9748 | 0.9789 | 0.9823 | 0.9853 |
| 128 | 0.7876 | 0.8900 | 0.9211 | 0.9373 | 0.9481 | 0.9555 | 0.9616 | 0.9669 | 0.9713 | 0.9750 | 0.9783 |
| 256 | 0.7478 | 0.8587 | 0.8926 | 0.9105 | 0.9222 | 0.9313 | 0.9383 | 0.9437 | 0.9483 | 0.9523 | 0.9557 |
| 512 | 0.7031 | 0.8275 | 0.8618 | 0.8806 | 0.8936 | 0.9036 | 0.9110 | 0.9174 | 0.9227 | 0.9276 | 0.9319 |
| 1,024 | 0.6524 | 0.7943 | 0.8300 | 0.8498 | 0.8629 | 0.8730 | 0.8813 | 0.8882 | 0.8942 | 0.8994 | 0.9041 |
| 2,048 | 0.5911 | 0.7598 | 0.7968 | 0.8176 | 0.8310 | 0.8419 | 0.8504 | 0.8574 | 0.8634 | 0.8688 | 0.8736 |
| 4,096 | 0.5191 | 0.7216 | 0.7626 | 0.7835 | 0.7982 | 0.8093 | 0.8181 | 0.8254 | 0.8317 | 0.8373 | 0.8419 |
| 8,192 | 0.4416 | 0.6708 | 0.7261 | 0.7492 | 0.7644 | 0.7755 | 0.7848 | 0.7923 | 0.7987 | 0.8045 | 0.8096 |
| 10,000 | 0.4192 | 0.6533 | 0.7143 | 0.7382 | 0.7546 | 0.7662 | 0.7752 | 0.7827 | 0.7891 | 0.7949 | 0.8002 |

## MRR@P

| Pool Size | MRR@P |
| ---: | ---: |
| 2 | 0.9868 |
| 4 | 0.9693 |
| 8 | 0.9477 |
| 16 | 0.9233 |
| 32 | 0.8965 |
| 64 | 0.8672 |
| 100 | 0.8471 |
| 128 | 0.8352 |
| 256 | 0.8004 |
| 512 | 0.7621 |
| 1,024 | 0.7193 |
| 2,048 | 0.6694 |
| 4,096 | 0.6107 |
| 8,192 | 0.5436 |
| 10,000 | 0.5231 |
