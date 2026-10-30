# 一键启动 WebRTC、场景、Play 与复位

在 **dt_wl** 执行。默认复用 `/home/dt/isaacsim5.1/isaac-sim.streaming.sh`，
沿用原来的 `100.64.61.8:49100`，不修改 Isaac 5.1/6.0 安装目录或公共配置。

## 启动

如果随后要运行 OpenVLA 闭环，仍先在另一个终端启动模型服务，等待模型准备好：

```bash
cd /home/dt/openvla/shenle_projects
/home/dt/miniconda3/envs/openvla/bin/python scripts/temp/serve_pose_v2.py
```

关闭自己之前启动的 Isaac 会话、ROS 控制和回放进程，然后运行：

```bash
cd /home/dt/openvla/shenle_projects
bash scripts/isaac/start_pose_v2_webrtc.sh
```

入口会：

1. 检查场景、复位依赖、已有 Isaac/控制进程，以及信令和媒体端口。
2. 启动独立的 Isaac 5.1 WebRTC 实例，日志写入项目目录。
3. 通过 Kit `--exec` 异步加载 `assets/sence/sence1.usd`，等待 USD 加载完成。
4. 开始 Play，等待初始化帧，再执行原来的 `replay_action_episode.py:start_reset()`。
5. 检查本次新产生的复位报告，以及 WebRTC TCP 信令端口。

看到以下输出，表示服务端场景已准备好：

```text
READY: scene loaded, Play active, reset verified.
Connect your WebRTC client to 100.64.61.8:49100
```

启动器此时返回终端，Isaac 独立进程继续运行；关闭此终端不影响它。
在 Windows 等客户端打开 Isaac Sim WebRTC Streaming Client，连接原地址。
连接后场景已经打开、Play 并复位，无需再进 Script Editor 粘贴三行。
本入口不会启动模型服务、采集动作、轨迹回放或闭环控制器。

## 检查、状态和停止

只检查文件、冲突进程和端口，不启动 Isaac，也不进行场景复位：

```bash
bash scripts/isaac/start_pose_v2_webrtc.sh --check
```

查看本入口最后启动的实例，`running` 表示该进程是否仍在运行：

```bash
bash scripts/isaac/start_pose_v2_webrtc.sh --status
```

停止只针对本入口记录并核实身份的实例，不使用全局 `pkill`：

```bash
bash scripts/isaac/start_pose_v2_webrtc.sh --stop
```

启动等待过程中按 Ctrl+C 会关闭这次新启动的实例。准备失败或超过默认 600 秒
启动期限，也会请求关闭该实例并保留日志，不会把失败显示成 READY。
如果 Kit 关闭需要较长时间，以 `--status` 的 `running` 字段为准。
服务器 READY 只验证场景复位与 TCP 信令监听，视频画面与远端网络仍需客户端确认。

## 日志与范围

每次日志在 `logs/pose_v2_webrtc/<时间>/kit.log`，状态为同目录的 `status.json`。
`logs/pose_v2_webrtc/latest.json` 保存 PID、启动身份、路径和端口。
复位本身继续使用原 `logs/replay_reset_*.json`，成功条件为
`completed=true`、`scene_reset=true`，且没有 `error`。

场景变更只在本次内存会话中；入口不保存、覆盖 USD。
若已有 Isaac 或控制器运行，入口会拒绝再启动，不会结束其他用户的会话。
本入口用于新启动实例；已有实例内重复复位仍可调用原来的 `start_reset()`。
`--reset-step k` 复位到 `episode_000001` 第 k 帧的关节与方块（默认 0）；已有实例内对应 `start_reset(k)`。
`--mode collect` 在 Play 后跑 `collect_push_batch.py`，不复位；默认等待 5400 秒。采数前可以没有旧 episode。
两台电脑的迁移任务与本入口无关，保持停止。

## 后续 20 步闭环

客户端观察场景正常、复位 READY 后，在另一个 dt_wl 终端执行：

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 scripts/ros/openvla_closed_loop.py --ros-args \
  -p max_steps:=20 -p delta_frame:=ee -p interp_duration:=0.20 \
  -p max_trans_m:=0.03 -p max_rot_rad:=0.10 -p max_ik_backtracks:=2
```

20 步是控制链路的前段测试，不等于完整推块任务成功。

## 验证与依据

标准库测试（不加载真实 Kit、模型或 GPU）：

```bash
/usr/bin/python3 -B -m unittest discover -s tests -p test_pose_v2_scene_startup.py -v
```

覆盖加载/Play/复位顺序、复位异常和被原函数捕获的失败、暂停、旧成功报告隔离，
以及独立进程在启动器返回后继续运行和按身份停止。进程测试使用临时假服务器，
不是 Isaac GPU 实测；实际场景首次启动还需要确认 READY 和客户端画面。

- [NVIDIA Isaac Sim 5.1 WebRTC 启动说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html)
- [Kit 启动脚本参数 --exec](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.0.3/guide/python_scripting.html)
