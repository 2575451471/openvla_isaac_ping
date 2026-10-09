# UR12e 与 OpenVLA

场景在 Isaac Sim 5.1。OpenVLA 只提供本机 HTTP，ROS 2 负责把图和关节接上。

本文分两块：**命令指导**（开机复制）和 **进度记录**（做了什么、还差什么）。

---

# 命令指导

顺序：Clash → OpenVLA（等 8000）→ Isaac 串流 → Windows 客户端里播放 → ROS 脚本。

模型约占 15GB 显存。加载完之前不要开仿真。

## 1. Clash

本机 `127.0.0.1:7890` 要在听。可用 NoMachine 开 Clash，不要用它再开一份 Isaac。

## 2. SSH A：启动 OpenVLA

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

curl -I --max-time 15 https://huggingface.co

conda activate openvla
cd /home/dt/openvla
python vla-scripts/deploy.py
```

`curl` 要有 `200`。卡住就先查 Clash。这个终端不要关。

另开一个窗口确认服务：

```bash
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://127.0.0.1:8000/docs
```

应打印 `200`。

外网不通、但模型已经下过时：

```bash
conda activate openvla
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/dt/openvla
python vla-scripts/deploy.py
```

## 3. SSH C：Isaac 串流

等 8000 通了再开：

```bash
~/isaacsim5.1/start_isaac_stream.sh
```

这个终端不要关。脚本会无窗口启动 Isaac，端点是 `100.64.61.8:49100`。

## 4. Windows：WebRTC 客户端

打开 Isaac Sim WebRTC Streaming Client，连接：

- 地址：`100.64.61.8`
- 端口：`49100`

然后打开 `/home/dt/openvla/shenle_projects/assets/sence/sence1.usd`，确认 `/rgb`、`/joint_states`、`/joint_command` 已接好，再点播放。

### 附：Isaac Lab 走同一条 WebRTC

和上面同一个客户端（`100.64.61.8:49100`）。先关掉 `start_isaac_stream.sh`，不要两边同时占 `49100`。

```bash
cd /home/dt/isaaclab2.3
PUBLIC_IP=100.64.61.8 LIVESTREAM=1 ./isaaclab.sh -p scripts/demos/quadrupeds.py
```

或：

```bash
PUBLIC_IP=100.64.61.8 ./isaaclab.sh -p scripts/demos/quadrupeds.py --livestream 1
```

用 `LIVESTREAM=1`（会带上 `PUBLIC_IP` 和端口 49100）；`2` 不适合这套远程 IP。这是 Lab 自己的仿真，不是播 `sence1.usd`，也不接本目录的 ROS/OpenVLA。

## 5. SSH B：ROS 脚本

不要 `conda activate openvla`。Isaac 必须已经在播放。

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
```

按需选一条：

```bash
python3 scripts/save_rgb_30.py
python3 scripts/ur12e_random_motion.py
python3 scripts/ee_nudge.py
python3 scripts/openvla_one_shot.py
python3 scripts/openvla_closed_loop.py
```

| 脚本 | 作用 |
| --- | --- |
| `save_rgb_30.py` | 从 `/rgb` 每秒存 1 张，共 30 张 |
| `ur12e_random_motion.py` | 钉在截图初值上，只让 `shoulder_pan_joint` 低速逆时针转（推杯子物理测试） |
| `ee_nudge.py` | 末端在基座系平移约 5 cm（默认 +x），FK+IK 后发 `/joint_command` |
| `openvla_one_shot.py` | 一帧图发给模型，只打印 7 个数，不控臂 |
| `openvla_closed_loop.py` | 订 `/rgb` 和关节，循环 `/act` → 限幅 5 cm / 0.10 rad → IK → 关节插值 → `/joint_command`。默认指令 `push the red cylinder to the right`；`delta_frame:=ee`（末端系）或 `base`（基座系）。逐步动作写到 `logs/openvla_closed_loop.latest.log` |

改指令：

```bash
python3 scripts/openvla_closed_loop.py --ros-args -p instruction:="push the red cylinder to the right"
```

---

# 进度记录

日期：2026-10-06（隔离测试补记：2026-10-07）

## 已经完成

三条链路都单独验证过。桌上红圆柱能被 UR10e 末端推着滑，物理可以先用这套。

Isaac Sim 播放时，ROS 2 里能看到：

- `/rgb`：固定相机彩色图，`sensor_msgs/Image`，实测 `1280×720`、`rgb8`
- `/camera_info`
- `/joint_states`：当前关节角
- `/joint_command`：目标关节角

界面：

- 相机：Tools → Robotics → ROS 2 OmniGraphs → ROS 2 Camera。Prim 选固定相机里的彩色相机，话题 `/rgb`
- 关节：同一菜单 ROS 2 Joint States。Publisher `/joint_states`，Subscriber `/joint_command`，Move Robot 已勾。`/joint_command` 要用 RELIABLE 和 BEST_EFFORT 都能对上（Isaac 订阅端 QoS 不一定和文档一样）
- 固定相机外壳 `RSD455` 勾了 Kinematic Enabled，播放时不会被重力拉下去
- 机械臂底座在桌边。引用根 Translate 和 `root_joint` Local Position 在停止播放时改过
- 固定相机从桌边斜看桌面，不是垂直朝下
- 桌上有红色圆柱当杯子；`/rgb` 里红色已足够明显
- 推杯子测试：`ur12e_random_motion.py` 用关节面板截图作初值（度：pan -1.5，lift -117.4，elbow -137.2，wrist1 -69，wrist2/3 0），只低速转 `shoulder_pan_joint`。杯子会被推动，沿桌面滑动
- `ee_nudge.py`：运动学基座 +x 约 5 cm，实测位移约 5.0 cm
- `openvla_closed_loop.py`：图 → `/act` → 限幅 → IK → `/joint_command` 整条已跑通，臂会动。早期行为僵、会抬高、不靠近杯子。推理约 0.8 s/步
- 2026-10-07：过肩相机机位暂定为完成。闭环已加步间关节线性插值（默认发布 50 Hz、每步约 0.75 s）。逐步动作写到 `logs/openvla_closed_loop.latest.log`
- 2026-10-07：闭环 log 显示模型多为抬高（`dz>0`、`dx≈0`），IK 跟得上；疑点在 VLA 方向 / 基座系加法，不是控臂链路断了

### 隔离测试（定位「只在初值附近蹭」）

**1. 绕过 OpenVLA，只测 IK（2026-10-07 已做，通过）**

不跑闭环。臂到 home 附近后：

```bash
source /opt/ros/humble/setup.bash
python3 /home/dt/openvla/shenle_projects/scripts/ee_nudge.py
python3 /home/dt/openvla/shenle_projects/scripts/ee_nudge.py --ros-args -p dx:=0.0 -p dy:=0.05
python3 /home/dt/openvla/shenle_projects/scripts/ee_nudge.py --ros-args -p dx:=0.0 -p dy:=0.0 -p dz:=0.05
```

结果：基座 +x / +y / +z 约 5 cm 都能一次走完并停住。说明 FK+IK 与 `/joint_command` 正常，闭环「不朝杯子」不是控臂坏了。

**2. 只跑 OpenVLA，不控臂（2026-10-08 已做，通过）**

图微变时会跳 bin（前一轮有两簇）。Pause / 画面稳定后连跑三次，结果一致：

```
dx=-0.00513  dy=0.00961  dz=0.04003
droll≈-0.054  dpitch≈-0.049  dyaw≈-0.046
gripper=0.996
```

含义：服务稳定；**模型原值就是约抬 4 cm、水平约 1 cm**，不是闭环限幅才变成抬高。

**3. 对照闭环 log 三列（2026-10-08 新指令+5 cm 限幅后再看）**

读 `logs/openvla_closed_loop.latest.log`（`push the red cylinder to the right`，`clip_trans_m=0.05`）：

- `raw == clip` 全程 300/300（最大 raw 范数约 4.1 cm，5 cm 限幅几乎没裁）
- `|EE(b−a) − clip_xyz|` 平均约 0.08 mm，无 `ik_skip` → **IK 把 clip 忠实变成目标**
- 步间连续性一般：下一步 `EE` 起点相对上一步目标平均差约 0.7 cm（插值未完又推理 + 用 `q_meas` 种子）
- 净位移约 **40 cm**（`dx≈-34, dy≈+18, dz≈+12` cm），比旧跑「只抬高」动得多；clip 主轴多为 **−y / −z**，不是稳定「往右」

**4. 坐标系对比（2026-10-08 已跑，代码有效但未碰杯）**

参数 `delta_frame`：`ee`（默认）或 `base`。log 有 `world_xyz`。

短跑对比（`max_steps:=20`）：

- `ee`（`…170146.log`）：`world ≠ clip`（旋转生效）；`|EE−world|≈0.2 mm`；净位移约 22 cm，多为降 z / 挪 xy
- `base`（`…170223.log`）：`world == clip`；同样 IK 忠实；净位移约 20 cm，主轴偏 −y
- 两次起始关节不同（ee 跑前臂还在上轮残留姿态），不能当严格 A/B
- **仿真仍未碰到红圆柱** → 换系能改世界方向，但不足以单独解决「不朝杯子」

OpenVLA 在 `http://127.0.0.1:8000`。`/docs` 返回 200。指令 `move the red cup to the left`，`unnorm_key=bridge_orig`，约 0.8 秒返回 7 个数。

这 7 个数是末端一步位移，不是关节角：

- 前 3 个：平移，米
- 中间 3 个：转角，弧度
- 最后 1 个：夹爪。这台臂没有夹爪，不用

## 还没做完

- 隔离测试 4：末端系加法（可选）
- 杯子 Reset、可程序判断的成功条件都还没有
- 策略还不靠近杯子（zero-shot 域差为主：无夹爪、红圆柱、视角/臂与 Bridge 不同）

## 接下来

1. 隔离测试 1–4 已做完：控臂/IK/坐标系切换正常；碰不到杯子 → **zero-shot 域差**为主。
2. 不要再拧限幅/IK 指望完成推杯；若要任务成功：更像 Bridge 的场景/指令/物体，或微调。
3. 成功条件和 Reset 后补。
4. **并行新线**：见下方「计划：Isaac Sim 采集微调数据」（与现有 OpenVLA/ROS 闭环分轨）。

---

# 计划：Isaac Sim 采集微调数据

状态：进行中（另开一条线，不替代现有闭环验证）  
日期：2026-10-09；专家路径已通，**次日做落盘采数**

## 目标

用**脚本专家轨迹**在仿真里生成成功演示，供后续 OpenVLA 微调。桌上已换红方块/相关物体（无夹爪先推）；抓取等有夹爪再加。

## 栈边界（重要）

| 要 | 不要 |
| --- | --- |
| **只用 Isaac Sim**（现有 `sence1.usd` / UR） | **不接 OpenVLA**（不启 `deploy.py`、不调 `/act`） |
| 场景内脚本：waypoint → IK → 驱动关节 → 读相机写盘 | **不走现有 ROS 2 采数链路**（不必为采数开 `scripts/*.py`、不必依赖 `/rgb` `/joint_command`） |
| 落盘：图像 + 末端增量动作 + 指令文本 | 不引入 Isaac Lab / Mimic（本线不用） |

说明：采数阶段仿真里直接控臂、直接取 viewport/相机帧即可；现有 ROS OmniGraph 可留在场景里但**本线不依赖**。微调训练、以及之后再接 ROS 闭环部署，是采数完成之后的事。

## 方法（与常见「多 waypoint 录演示」一致）

1. 定稀疏末端 waypoint（相对物体或桌面）：如 home → 接近上方 → 下降接触 → 侧推 → 停。
2. waypoint 之间笛卡尔插值，得到稠密末端位姿序列。
3. 逐步 IK → 关节目标，在 Sim 里执行；IK 失败则丢弃该条。
4. 每控制/渲染步同时记录：RGB（优先过肩机位）、相对上一步的 EE 增量（与部署时 `delta_frame` 约定一致）、instruction、可选关节/绝对位姿。
5. 物体位姿等做随机化；**仅成功 episode** 写入数据集。
6. 成功条件与 Reset：本线要补（推到位/位移阈值等），与闭环线可共用定义。

## 步骤（待做）

- [x] 在 Isaac Sim 内选定驱动与读图 API（Articulation / Camera），确认不依赖 ROS 话题也能控臂出图  
      已验证：`/World/ur10e` + `apply_action` / `set_joint_positions`；相机可出图。  
      Script Editor 用 `scripts/redo.py`。注意：**没有** `set_joint_position_targets`。
- [x] waypoint + 笛卡尔插值 + IK（Sim 内，不经 ROS）  
      `redo.py`：World XYZ 点位 → base / FK 系（含 Isaac↔FK 的 Rz180）→ 预计算轨迹 → 连续流式执行。  
      姿态用目标 World 四元数 + waypoint `ori_alpha` 渐变；中间点 `HOLD_FRAMES=0` 减少停顿感。
- [x] 专家路径 P1…P7（World Translate）已定并在 Sim 里跑通运动测试  
      点位与四元数以当前 `scripts/redo.py` 顶部常量为准。
- [ ] **下一步（采数落盘）**：每步同步写 RGB + EE 增量（或 6 维无爪）+ instruction；对齐微调 `unnorm_key`/统计 ← **明天从这里接着**
- [ ] 物体随机化 + 成功过滤 + Reset，批量出 N 条
- [ ] （之后）用该数据微调 OpenVLA，再回到现有 ROS 闭环评测——那是下一阶段，不在本计划采数步骤内

## 与现有闭环线的关系

- **闭环线**（OpenVLA + ROS）：继续验证部署；零样本碰不到杯子已结论为域差。
- **本采数线**：只造监督数据；跑采数时不必开 Clash/OpenVLA/ROS 脚本。
- 两边暂时并行；采数脚本与 `openvla_closed_loop.py` 不要揉成一个进程。
