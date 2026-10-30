# 第二步 C：记录动作通过 ROS 控制器执行

已通过关节角回放和修正动作经 IK 后的直接仿真回放。本步骤使用同一条
`raw_pose_v2/episode_000001` 的 129 个动作，经过实际 ROS 控制器，订阅
`/rgb`、`/joint_states` 并发布 `/joint_command`。动作来自记录，不调用模型服务。

## 时序修正

旧控制器会在上一段 0.75 秒插值尚未完成时接收新目标，且使用推理前缓存的关节角求解 IK。
现改为完成插值、实测关节到达目标后，等待新图像再获取下一个动作；推理返回后重新读取实测关节。
过期关节数据、非法动作、IK 失败或过大跳变会中止控制。

ROS 回放沿用上述控制器，仅替换动作提供方法；保留每步记录时长，不裁剪原始动作。
等待实际到位可能延长总时间，因此本步骤不严格复现原始接触速度。
未来记录关节角仅做误差检查，不参与 IK 初始化。偏离参考轨迹 10 毫米或 2 度时中止。

## 肘部姿态与稳定性修正

首次 ROS 回放发现：起始 HOME 接近肘关节奇异位形，实测关节角的微小变化可能让 IK
选择负肘分支，而演示使用正肘分支。末端位置接近不能证明整条机械臂姿态正确。
现将项目 IK 默认设为正肘分支，在接近伸直时用固定的正肘偏置初始化求解。
偏置只影响 IK 初始猜测，不直接发送到机器人，也不使用未来记录关节角。
`ik(..., elbow_branch=-1)` 可选择负肘分支，`0` 保留无分支约束的旧行为。

每步需要关节到位、位置窗口估计的变化速率低于 0.01 rad/s，并持续稳定至少 0.10 秒。
变化速率取至少 0.10 秒连续关节位置的最大范围除以时间跨度，能识别往返振荡。
仿真中曾出现位置连续 10 秒不变、速度字段却保持 0.0106 rad/s 的情况；
原始速度现在只用于诊断，不单独阻止已稳定的关节进入下一步。
位置误差阈值仍为 0.001 rad，未放宽。日志同时保存位置窗口速率和原始速度字段。
双 QoS 订阅收到的重复或过期时间戳会被忽略。
求解期间实测关节变化超阈值时，以新状态对同一个末端目标重算，最多三次；
不重复叠加动作，也不放宽原有位置、旋转、跳变限制。
回放额外检查各关节相对演示的角度误差，超过 15 度会中止，避免只检查末端而漏掉分支错误。
以上只修正执行链路，不表示模型训练已修复。

超时诊断会在控制日志记录 `motion_wait` 和 `motion_timeout`，列出具体未满足的条件、
原始目标关节角、实测关节角、误差、反馈新鲜度、速度和稳定时长。
报告中的 `motion_failure` 保存停车前的诊断，避免保持当前位置后覆盖原始失败目标。
新增诊断不改变到位阈值或超时时间；需要按实际未满足的条件继续定位。

## 1. 重置场景

结束其他运动控制脚本，保持原场景 Play。在 Isaac Script Editor 执行：

```python
import runpy
g = runpy.run_path("/home/dt/openvla/shenle_projects/scripts/isaac/replay_action_episode.py")
g["start_reset"]()
```

看到 `[replay] RESET READY` 后进入下一步。无需退出 Isaac Sim。

## 2. 在 dt_wl 终端执行

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 scripts/ros/replay_actions_ros.py
```

必须在能够连接当前 Isaac ROS bridge 的环境中运行，沿用原有 ROS_DOMAIN_ID 和 RMW 配置。
脚本先进行无运动的 IK 规划检查，再等待相机和关节状态，然后自动开始控制仿真。
成功执行全部动作后显示 `[ros-replay] DONE actions_completed=True` 并退出。
此标记只表示动作执行完毕；需要观察方块是否成功被推下右侧。
报告保存到 `logs/replay_ros_*.json`，其中 `task_success` 留空，避免将动作完成当作任务成功。
可以在终端按 Ctrl+C 提前停止；暂停 Isaac 也可停止场景运动。

若提示没有相机或关节消息，检查 Play 和场景原有 ROS bridge，不要重复启动控制器。
若报告轨迹偏离，保留日志进一步诊断；当前旧训练权重尚未修复。

## 无运动检查

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 -m unittest discover -s tests -p 'test_*.py' -v
/usr/bin/python3 -c 'import sys; sys.path.insert(0, "scripts/ros"); import replay_actions_ros; print("ROS imports OK")'
```
