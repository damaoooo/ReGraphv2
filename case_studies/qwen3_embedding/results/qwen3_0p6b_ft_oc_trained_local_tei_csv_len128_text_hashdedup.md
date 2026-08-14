# Qwen3-Embedding-0.6B OC fine-tuned local TEI qwen CSV len128 text hashdedup on OC test_final_set

- Endpoint: `http://127.0.0.1:8080`
- Model ID: `Qwen3-Embedding-0.6B OC fine-tuned local TEI qwen CSV len128 text hashdedup`
- Dataset pool: `/path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_dataset_pool`
- Positive map: `/path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_positive_map.pkl`
- Cache file: `/path/to/qwen-case-study/experiments/qwen3_0p6b_ft_oc_trained_local_tei_csv_len128_text_hashdedup.bin.npy`
- Log file: `/path/to/qwen-case-study/experiments/eval_reports/logs/qwen3_0p6b_ft_oc_trained_local_tei_csv_len128_text_hashdedup.log`
- Command: `python evaluate.py /path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_dataset_pool /path/to/rell/IR/Dataset-1-Oc-qwen-text-fused/test_final_set_csv_len128_text_hashdedup/train_positive_map.pkl --tei-endpoint http://127.0.0.1:8080 --ks 1,5,10,15,20,25,30,35,40,45,50 --batch-size 128 --max-length 2048 --tei-workers 4 --tei-timeout 180 --tei-max-retries 12 --tei-retry-base-delay 2 --gpu-batch-size 512 --eval-samples 0 --embeddings-path /path/to/qwen-case-study/experiments/qwen3_0p6b_ft_oc_trained_local_tei_csv_len128_text_hashdedup.bin.npy --seed 42 --gpu`
- Embedding shape: `(208293, 1024)`
- Anchors evaluated: `164,988`
- MRR@10: `0.7609`
- MRR@30: `0.7625`

## Summary

| Setting | Recall@1 @ Pool 10,000 | MRR@P @ Pool 10,000 | MRR@10 | MRR@30 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-ft OC qwen-csv-len128-text-hashdedup | 0.6679 | 0.7631 | 0.7609 | 0.7625 |

## Recall@K

| Pool Size | Recall@1 | Recall@5 | Recall@10 | Recall@15 | Recall@20 | Recall@25 | Recall@30 | Recall@35 | Recall@40 | Recall@45 | Recall@50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9968 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9920 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 0.9850 | 0.9995 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 16 | 0.9761 | 0.9976 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32 | 0.9653 | 0.9935 | 0.9979 | 0.9992 | 0.9997 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 64 | 0.9522 | 0.9871 | 0.9940 | 0.9967 | 0.9981 | 0.9987 | 0.9992 | 0.9995 | 0.9997 | 0.9998 | 0.9999 |
| 100 | 0.9426 | 0.9821 | 0.9901 | 0.9937 | 0.9960 | 0.9972 | 0.9980 | 0.9985 | 0.9989 | 0.9992 | 0.9994 |
| 128 | 0.9367 | 0.9786 | 0.9878 | 0.9918 | 0.9942 | 0.9959 | 0.9969 | 0.9976 | 0.9981 | 0.9985 | 0.9988 |
| 256 | 0.9184 | 0.9683 | 0.9795 | 0.9848 | 0.9881 | 0.9903 | 0.9920 | 0.9933 | 0.9945 | 0.9954 | 0.9960 |
| 512 | 0.8954 | 0.9559 | 0.9693 | 0.9757 | 0.9800 | 0.9829 | 0.9851 | 0.9868 | 0.9883 | 0.9894 | 0.9904 |
| 1,024 | 0.8665 | 0.9418 | 0.9574 | 0.9649 | 0.9697 | 0.9731 | 0.9760 | 0.9781 | 0.9804 | 0.9820 | 0.9833 |
| 2,048 | 0.8255 | 0.9266 | 0.9430 | 0.9520 | 0.9578 | 0.9621 | 0.9653 | 0.9677 | 0.9700 | 0.9718 | 0.9733 |
| 4,096 | 0.7695 | 0.9099 | 0.9275 | 0.9369 | 0.9437 | 0.9486 | 0.9525 | 0.9554 | 0.9580 | 0.9602 | 0.9623 |
| 8,192 | 0.6936 | 0.8898 | 0.9114 | 0.9215 | 0.9279 | 0.9328 | 0.9372 | 0.9409 | 0.9439 | 0.9466 | 0.9488 |
| 10,000 | 0.6679 | 0.8821 | 0.9064 | 0.9169 | 0.9237 | 0.9285 | 0.9326 | 0.9361 | 0.9392 | 0.9420 | 0.9445 |

## MRR@P

| Pool Size | MRR@P |
| ---: | ---: |
| 2 | 0.9984 |
| 4 | 0.9957 |
| 8 | 0.9914 |
| 16 | 0.9855 |
| 32 | 0.9776 |
| 64 | 0.9678 |
| 100 | 0.9602 |
| 128 | 0.9556 |
| 256 | 0.9411 |
| 512 | 0.9235 |
| 1,024 | 0.9020 |
| 2,048 | 0.8731 |
| 4,096 | 0.8345 |
| 8,192 | 0.7815 |
| 10,000 | 0.7631 |
