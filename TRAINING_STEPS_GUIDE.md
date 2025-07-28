# 预训练步数配置建议

## 当前默认配置
- max_steps: 10000 (约等于之前的短时间训练)
- save_steps: 1000 (每1000步保存一次)
- warmup_steps: 1000 (前1000步进行学习率预热)
- logging_steps: 100 (每100步记录一次日志)

## 根据训练时间调整步数的建议

### 短期测试 (1-2小时)
```python
max_steps = 1000
save_steps = 200
warmup_steps = 100
```

### 中期训练 (8-12小时)
```python
max_steps = 5000
save_steps = 500
warmup_steps = 500
```

### 长期训练 (24-48小时)
```python
max_steps = 20000
save_steps = 2000
warmup_steps = 1000
```

### 完整训练 (多天)
```python
max_steps = 50000
save_steps = 5000
warmup_steps = 2000
```

## 如何修改配置

1. 直接在 pretrain_config.py 中修改 max_steps 的值
2. 或者在运行时通过代码覆盖:

```python
from Pretrain.pretrain_config import DEFAULT_CONFIG

# 创建自定义配置
config = DEFAULT_CONFIG
config.max_steps = 5000  # 你想要的步数
config.save_steps = 500  # 调整保存频率

# 使用自定义配置运行
main(config)
```

## 步数与时间的估算
- 每步大约需要 1-3 秒 (取决于硬件和批次大小)
- 1000步 ≈ 30分钟 - 1小时
- 5000步 ≈ 3-5小时
- 10000步 ≈ 6-10小时
