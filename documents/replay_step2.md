# 第二步 A：在 Isaac 回放已记录关节轨迹

先验证场景、初始状态和物理执行能否重现成功示范。本步骤使用 `joint_positions.npy` 和原始仿真时间戳，不调用模型、不经过 IK，也不发送 ROS 命令。通过后才能进入动作标签加 IK 的回放验证；本步骤成功不等于模型或整条控制链路已验证。

## 操作

1. 在原终端用 Ctrl+C 停止 OpenVLA 服务以腾出显存，停止其他控制脚本。
2. 启动 Isaac：`cd ~/isaacsim5.1 && ./start_isaac_stream.sh`。
3. WebRTC 连接 `100.64.61.8`，打开 `/home/dt/openvla/shenle_projects/assets/sence/sence1.usd`，点击 Play。
4. 在 Isaac 的 Script Editor 中运行：

```python
p = "/home/dt/openvla/shenle_projects/scripts/isaac/replay_recorded_episode.py"
g = {"__file__": p}
exec(compile(open(p).read(), p, "exec"), g)
g["start_replay"]()
```

脚本自动将仿真机械臂和方块还原到 `datasets/raw_pose_v2/episode_000001` 的初始状态，重放约 25.8 秒仿真时间，再等待 1.5 秒观察方块。无需运行 `redo.py` 或采数脚本。实际耗时取决于仿真速度。

原始示范从桌面中央沿世界 −X 方向推走方块，最终约为 `[-0.928, 0.0002, -0.055]` 米。回放不要求末坐标完全相同；沿右侧推出/推下并满足与采数一致的成功门限即可。

## 看什么

- 机械臂是否重现接近、下降、接触、推走方块的顺序。
- 控制台是否显示 `[replay] DONE task_success=True`。
- 结果写到 `logs/replay_joints_<时间>.json`。`completed` 表示轨迹播放完成，`task_success` 才表示方块任务成功，两者不同。
- 若出现 `[replay] STOPPED` 或 `START FAILED`，记录完整信息。不要反复运行旧模型或马上重训。

出现持续关节跟踪误差会停止回放。手动点击 Isaac 的 Pause/Stop 也会终止回放；脚本不保存场景文件。不要在回放期间启动其他控制器。

## 验证范围

脚本的加载、插值、角度连续性与成功门限已经做离线回归检查；实际 Isaac 物理回放需要按以上步骤观察，尚未自动验收。
