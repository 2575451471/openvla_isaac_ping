# 第二步 B：修正动作 → IK → 仿真回放

前一步直接回放记录的关节角，已确认成功。此步骤从
`datasets/raw_pose_v2/episode_000001/actions.npy` 的 129 步动作生成轨迹，
使用 `scripts/ros/ee_nudge.py` 中的原始 FK/IK 数学函数，不导入 ROS 节点。
只用第一帧记录关节角初始化；后续 IK 以前一次求解结果为起点。
未来记录关节角仅用于求解后的误差检查。

## 执行

保持原场景 `assets/sence/sence1.usd` 和 Play 状态，先结束其他运动控制脚本。
在 Isaac Sim 的 Script Editor 中粘贴并执行：

```python
p = "/home/dt/openvla/shenle_projects/scripts/isaac/replay_action_episode.py"
g = {"__file__": p, "__name__": "isaac_action_replay"}
exec(compile(open(p).read(), p, "exec"), g)
g["start_replay"]()
```

脚本先完成全部 IK 规划和误差检查，然后重置机器人、方块并按原始仿真时间戳回放。
约需 26 秒仿真时间。结束应显示 `[replay] DONE task_success=True`。
报告保存到 `logs/replay_actions_*.json`，其中 `plan` 包含 IK 和参考轨迹误差。
暂停仿真会中止回放；重新运行前等当前任务停止。

## 检查与边界

单步 IK 位置误差不得超过 2 毫米、旋转误差不得超过 1 度、单关节跳变不得超过 20 度；
累积轨迹与记录相比不得超过 5 毫米或 2 度。检查失败时不会开始回放。
保留原始动作、速度和时间戳，不应用模型输出裁剪。

终端中仅做规划检查，不产生仿真运动：

```bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 scripts/isaac/replay_action_episode.py --check
/usr/bin/python3 -m unittest discover -s tests -p 'test_*replay.py' -v
```

这一步验证动作标签、顺序 IK 与场景物理执行。通过后仍需单独检查 ROS 实时反馈、
控制周期及模型预测；它不能证明旧模型已修复。当前训练流水线仍指向旧数据，暂勿重跑。
