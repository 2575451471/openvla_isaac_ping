# Pose-v2 短试训

本流程使用新版 `~/tensorflow_datasets_pose_v2/ur10e_push_sim/2.0.0/`，
从本机缓存的原始 OpenVLA-7B 权重重新创建 LoRA；不沿用旧微调权重。
Isaac Sim 和推理服务不参与数据检查或训练。

## 数据输入检查

```bash
cd /home/dt/openvla/shenle_projects
bash scripts/temp/run_pose_v2_trial.sh --check
```

检查禁用 GPU，限制最多两个 CPU 逻辑核，遍历全部 3208 个训练动作和
567 个验证动作。归一化统计只来自 22 条训练演示，验证集复用同一统计。
报告还会比对修正 raw 的训练动作统计，检查图像、语言、夹爪标签和动作 token。
成功显示 `TRAINING_PREFLIGHT_PASSED`，报告位于
`datasets/pose_v2_preparation/training/preflight.json`。

## 启动短试训

```bash
bash scripts/temp/run_pose_v2_trial.sh --train
```

此命令会实际占用 GPU。入口检查 CUDA 可用、没有其他 GPU 计算进程，
以及至少 18000 MiB 空闲显存；
这个门槛不能保证没有显存不足，真实峰值以首次试训为准。
默认参数：单 GPU、每次一个样本、梯度累积 4 次、200 次参数更新，
共读取 800 个训练样本。LoRA rank 32、不量化、无图像增强。
训练样本在全部 3208 个动作间打乱，再循环读取，避免短试训仅看到前几条演示。
学习率峰值 `5e-5`，前 20 次更新预热，随后余弦下降到峰值的 10%，
梯度范数上限 1.0。训练和验证保持相同的图像与动作转换方式。

先记录未训练时的验证指标，之后每 50 次更新以及结束时验证一次。
固定取遍布整个验证集的 64 个样本，不更新验证样本的梯度。
指标为教师强制下的 loss、动作 token 准确率及归一化动作 L1，
不能视为机械臂实际任务成功率。

## 结果与下一步

每次运行创建 `runs/ft_pose_v2/trial_时间戳/`，保存配置、数据划分、
训练统计、`metrics.jsonl`、各次验证的 LoRA adapter 与优化器状态。
`best.json` 按验证 loss 选择；若未超过未训练基线，保留 `step: 0`，
不强行把最后一个 adapter 认定为最好。

成功结束显示 `POSE_V2_TRIAL_COMPLETE`。这只是短试训，尚不足以判断收敛。
先检查 loss 是否稳定、验证是否改善，再决定训练量；随后还需要检查
自回归预测能否随不同图片改变，最后回到 Isaac Sim 做闭环任务评估。
adapter 需要另行加载或合并，不能直接当作完整模型目录交给旧推理服务。
当前脚本不提供从优化器状态自动续训的入口，也不会自动合并模型。

旧 `run_ft_pipeline.sh` 仍对应旧数据和参数，请使用这里的专用入口。
原始数据、正式 TFDS 和旧模型保留。

## 环境与检查记录

两个相关 Python 3.10 环境（`openvla`、`rlds_env`）使用
`protobuf==4.21.12` 与 `tensorflow-metadata==1.17.2`，
符合当前 TensorFlow 及其他已安装依赖的声明要求，并已验证真实导入与数据读取。
原有包备份位于 `~/.local/state/openvla-protobuf-fix/`。
此调整解决 `MessageFactory.GetPrototype` 报错；CPU 模式的 CUDA 启动日志仍可能出现。
完整数据输入检查与正式 TFDS 校验已通过，59 项测试通过；
依赖调整后重新检查了全部数据输入、正式 TFDS 和新增 6 项测试。
GPU 试训尚未启动，真实显存峰值和模型改善情况需由试训结果确认。
