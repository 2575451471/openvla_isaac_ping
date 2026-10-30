# HOME 重采与 4bit 重训

2026-10-27：闭环第 40 帧验证确认离开奇异点后 IK 能跑，模型方向不对。
本轮换非奇异 HOME，覆盖 `raw_pose_v2` 与 TFDS `2.0.0`，从原始 OpenVLA 做 4bit LoRA。

## 采数约定

- `HOME_DEG = (0, -88.4, 80.1, 0, 254.2, 0)`，来自 2026-10-28 关节滑条截图
- P1 = 复位后实测末端世界坐标，不加噪声；臂已经在这一点，航点从 P2 开始
- P2 = P1 与 P3 的中点；P3 的高度是原 P3 与原 P4 高度的中点（0.562 m），XY 仍是原 P3；P4 仍用原接近模板，P5–P7 仍相对方块
- 输出：`datasets/raw_pose_v2`（过夜脚本会先清空旧 episode）
- 闭环：xyz 硬约束 3mm；旋转参考模型 RPY，容许 20°，不达标也不停车

姿态预览（脚本自己打开 Isaac、摆好 HOME、截多机位、再退出）：

```bash
/home/dt/openvla/shenle_projects/scripts/temp/preview_home_isaac.py
```

图在 `logs/preview_home_isaac/`。总览是 `overview.png`，单机位是 `quarter.png`、`side.png`、`top.png`、`front.png`、`elbow.png`。

## 无人值守

确认预览姿态后，用 `scripts/temp/overnight_home_v3.sh` 经 `delayed-kill` 启动。
脚本会自动：清空旧 episode → Isaac 采数 → 打标签 → 关 Isaac → 覆盖 TFDS → 4bit 训练 1000 步 → 自删。
闭环评估不在过夜任务里。

旧 `run_ft_pipeline.sh` 不要用。4bit 在 4090 上曾出过问题，以训练日志为准。
