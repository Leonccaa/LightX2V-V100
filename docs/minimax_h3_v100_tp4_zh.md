# MiniMax-H3 四卡 V100 TP4 实现说明

本分支基于 `ModelTC/LightX2V` commit
`60631ff15310d645128226a98f138245bd59eba0`，公开的是代码适配和复现工具，
不包含 H3、LoRA、文本编码器或 VAE 权重。

实现分为三段：

```text
官方 H3 conditioning
  -> safetensors 条件包
  -> 4x V100 FP16 TP4 DiT
  -> 单设备官方 video/audio VAE
  -> H.264 + AAC
```

关键改动：

- 主 DiT 使用 V100 FP16 Tensor Core，text refiner、归一化、残差和防溢出
  路径保留 FP32。
- 每个 TP rank 显式 materialize 自己的 shard，避免 contiguous view 继续持有
  完整源 tensor storage；修复前每 rank 约占 30.6 GiB。
- Attention 使用 PyTorch SDPA，RMSNorm/RoPE 使用原生 PyTorch。
- 支持官方 pruned AdaLN curve checkpoint 的插值和 projection。
- 支持预计算 T2V/I2V/R2V conditioning bundle、逐 block finite gate，以及
  DiT/VAE 分阶段执行。
- R2V 使用 pruned Ref2VA + official base evaluations，不跨模型家族套用
  FL2VA Turbo。

完整转换、conditioning 导出、`torchrun` 和 VAE 解码命令见
[英文复现说明](minimax_h3_v100_tp4.md)，实际速度、显存和容量边界见
[benchmark 报告](minimax_h3_v100_tp4_benchmark.md)。

这是经过本地硬件验收的下游实验分支，不代表 LightX2V 上游正式支持，也不
代表所有主题的生成质量均已验收。模型和权重仍须遵守各自的上游许可证与
访问条件。
