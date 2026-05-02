# 预训练轮数与 checkpoint 配置

## 当前默认配置

- `num_train_epochs = 1`：默认训练 1 个 epoch。
- `max_steps = -1`：不再用固定 step 截断训练。
- `save_steps = 10000`：每 10K step 保存一次常规 checkpoint。
- `eval_steps = None`：有验证集时默认跟随 `save_steps`，也就是每 10K step 验证一次。
- `save_total_limit = None`：默认不删除历史 checkpoint。

训练结束时会额外维护 3 个命名 checkpoint：

- `checkpoint-last`：最后一次训练状态。
- `checkpoint-best-validation-loss`：validation loss 最低的 checkpoint。
- `checkpoint-best-train-loss`：train loss 最低的 checkpoint。

这些目录里会写入 `named_checkpoint_info.json`，记录来源 checkpoint、step 和对应 loss。

## 常用覆盖方式

通常不需要再设置 `max_steps`。如果想多训几轮，优先改 epoch：

```bash
python -m Pretrain.run_pretrain train --set num_train_epochs=2
```

如果只是想调整保存频率：

```bash
python -m Pretrain.run_pretrain train --set save_steps=20000
```

只有在做很短的 smoke test 时，才建议临时设置 `max_steps`：

```bash
python -m Pretrain.run_pretrain train --set max_steps=1000
```
