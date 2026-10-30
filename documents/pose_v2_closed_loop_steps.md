# 第 800 步权重的有限步数闭环测试

服务默认固定加载 `trial_20261018-163831-005902` 的最佳
`adapter_step_000800`，同时加载原始 OpenVLA 和本轮 train-only 动作统计。
图像先按训练时的 RLDS 方法缩放到 224×224，再交给原始 processor。
不需要合并 LoRA，也不修改旧模型服务。

## 1. 终端 A：先启动模型服务

在 dt_wl 执行；此时先保持 Isaac Sim 关闭，等待其他 GPU 计算任务结束。

```bash
cd /home/dt/openvla/shenle_projects
/home/dt/miniconda3/envs/openvla/bin/python scripts/temp/serve_pose_v2.py
```

看到 `POSE_V2_MODEL_READY` 和
`Uvicorn running on http://127.0.0.1:8000` 后保留此终端运行。
启动时会用一张记录图片预热，不发送 ROS 指令。

可在另一终端检查服务：

```bash
curl -s http://127.0.0.1:8000/health
```

应包含 `"ready":true`、`"best_step":800`，以及
`"preprocessing":"training_rlds_resize_224_then_processor"`。
服务启动前检查端口和其他 GPU 计算进程；桌面显示服务不阻止启动。

## 2. Isaac Sim：打开场景并重置

启动原来使用的 Isaac Sim，打开
`/home/dt/openvla/shenle_projects/assets/sence/sence1.usd`，点击 Play。
先退出所有旧 ROS 控制器／动作回放终端，避免多路指令竞争。
在 Isaac 的 Script Editor 中完整执行以下三行：

```python
import runpy
g = runpy.run_path("/home/dt/openvla/shenle_projects/scripts/isaac/replay_action_episode.py")
g["start_reset"]()
```

看到 `[replay] RESET READY` 后继续。这里恢复记录的初始机器人关节和方块位置，
不播放演示，也不运行 `redo.py` 的采集动作。

## 3. 终端 B：确认 ROS 消息

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
ros2 topic list
```

确认 `/rgb`、`/joint_states` 存在，沿用之前成功 ROS 回放时的
ROS_DOMAIN_ID 和 RMW 配置。若缺少消息，先检查 Play 和 ROS2 Bridge；
不要通过重复启动运动脚本解决。

## 4. 终端 B：只执行前 20 步

```bash
/usr/bin/python3 scripts/ros/openvla_closed_loop.py --ros-args \
  -p server_url:=http://127.0.0.1:8000/act \
  -p unnorm_key:=ur10e_push_sim \
  -p delta_frame:=ee \
  -p max_steps:=20 \
  -p interp_duration:=0.20 \
  -p max_trans_m:=0.03 \
  -p max_rot_rad:=0.10
```

该命令会实际控制仿真。从当前实测关节出发，平移采用 EE 坐标系，
旋转采用左乘 RPY，保留已有 IK、反馈到位和超时检查。
不要在此时再次执行重置或启动其他运动脚本。
观察末端是否按预期接近方块、整臂姿态是否正常；20 步只是前段测试。

达到 20 步且实测到位后显示
`Max steps completed and measured target reached; holding last command.`
此时停止生成新动作，进程仍保持最后关节指令。在终端 B 按 Ctrl+C 退出控制。
若出现明显异常，可按 Ctrl+C，并在 Isaac 点击 Pause。
此阶段没有自动判断方块任务是否成功，也不会自动继续到第 21 步。

## 记录与重复测试

控制日志在 `logs/openvla_closed_loop_*.log`，最新副本为
`logs/openvla_closed_loop.latest.log`；服务请求记录在
`logs/pose_v2_server_*.jsonl`，健康接口会给出本次路径。
保留完整日志，记录开始／结束画面及异常发生的步数。
下一轮先 Ctrl+C 结束终端 B，再在 Isaac 重新 `start_reset()`，然后重新启动控制。
不要从偏离后的中间姿态直接追加完整测试。

服务准备检查已通过：新增 3 项协议测试通过；使用控制器实际序列化函数
对 4 条验证演示的图片发送 HTTP 请求，动作与离线训练缩放路径逐项完全相同。
双重编码兼容性及非法图片拒绝也通过。本次检查没有控制仿真，
临时 8001 端口服务已结束；Isaac 闭环效果仍需按上述步骤观察。
