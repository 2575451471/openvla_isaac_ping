# shenle_projects 印象记忆

读这份文件，是为了在后续对话里快速恢复「这个目录是干什么的、做到哪一步、别和上层 OpenVLA 训练仓库混在一起」。

- 路径：`/home/dt/openvla/shenle_projects`
- 角色：挂在上游 OpenVLA 仓库里的**本机实验目录**，不是上游训练/评测代码
- 进度快照日期：2026-10-07（详见 `documents/README.md`）
- 本文整理日期：2026-10-07

更细的操作记录以 `documents/README.md` 为准；本文是整体地图和约定，不是逐步实验日志。

---

## 一句话

在 **Isaac Sim 5.1** 里放一台 UR 机械臂和固定相机，用 **ROS 2 Humble** 把图和关节接到本机 **OpenVLA HTTP 服务**。OpenVLA **不进仿真**，只在 `http://127.0.0.1:8000` 上根据图像和语言指令给出末端一步位移。

当前三条链路都通。点动 5 cm 已跟上。**闭环已跑通「会动」**。过肩相机机位暂定完成。闭环默认 `delta_frame=ee`、指令 push cylinder right、限幅 5 cm。隔离测试 1–4 已做完：IK/坐标系切换正常，仍碰不到红圆柱 → 域差为主。Reset/成功条件还没有。

---

## 和上层仓库的关系

本目录是 git 里 `openvla` 仓库的未跟踪子目录。上层是标准 OpenVLA（`prismatic/`、`vla-scripts/` 等）。

本目录会用到、但不要当成「本项目代码」的：

| 外部 | 用途 |
| --- | --- |
| `/home/dt/openvla/vla-scripts/deploy.py` | OpenVLA 推理服务（conda 环境 `openvla`） |
| Isaac Sim 5.1 | 场景播放、ROS 2 OmniGraph 出图/收关节 |
| `/opt/ros/humble` | 跑 `scripts/` 里的 Python 节点前必须 source |

改仿真场景、ROS 桥、控臂脚本：动这里。改模型加载、`/act` 协议：动上层 `deploy.py`。

---

## 目录地图

```
shenle_projects/
  documents/README.md          实验进度（中文，权威）
  scripts/
    ur12e_random_motion.py     截图初值 + 只转 shoulder_pan，已能推动红杯子
    save_rgb_30.py             从 /rgb 每秒存 1 张，共 30 张
    ee_nudge.py                末端点动 5 cm，FK+IK
    openvla_one_shot.py        取一帧 /rgb，POST /act，只打印 7 维，不发关节
    openvla_closed_loop.py     图+关节 → /act → 限幅 → IK → /joint_command
    rgb_capture/               存图目录（目前可能是空的）
  logs/
    openvla_closed_loop.latest.log  最近一次闭环逐步动作（读这个）
  assets/
    sence/sence1.usd           场景（目录名拼写是 sence，不是 scene）
    ur10e/                     UR 资产与 configuration 变体（夹爪/物理/传感器 usd）
  .cursor/PROJECT_IMPRESSION.md  本文件
```

没有 Python 包、没有 tests、没有自己的训练循环。脚本都是独立 ROS 2 节点。

---

## 架构印象

```
Isaac Sim（播放中）
  ROS 2 Camera graph  →  /rgb (1280×720 rgb8), /camera_info
  ROS 2 Joint States  →  /joint_states
                      ←  /joint_command   （Move Robot 已勾）

ROS 2 进程（scripts/*.py，source humble 后跑）
  订阅图 / 关节
  HTTP POST → OpenVLA /act
  （未来）IK 后发布 /joint_command

OpenVLA（conda openvla，先于 Isaac 启动）
  GET  /docs
  POST /act   {image, instruction, unnorm_key} → 7 维 numpy
```

启动顺序：**先等模型听端口，再开 Isaac Sim。** 模型大约占 15GB 显存，加载过程中不要开仿真。

```bash
conda activate openvla
cd /home/dt/openvla
python vla-scripts/deploy.py
```

跑本目录脚本：

```bash
source /opt/ros/humble/setup.bash
python3 /home/dt/openvla/shenle_projects/scripts/<脚本>.py
```

---

## 已验证事实

- `/rgb`：固定相机彩色图，`sensor_msgs/Image`，实测 `1280×720`、`rgb8`
- 关节：`/joint_states` 出当前角，`/joint_command` 进目标角；肩转已能推动红杯子。命令建议 RELIABLE 与 BEST_EFFORT 都发
- OpenVLA：`curl /docs` 200；用一帧 `/rgb`、指令 `move the red cup to the left`、`unnorm_key=bridge_orig`，约 0.8s 返回 7 个数
- 7 维含义：`dx, dy, dz`（米）+ `droll, dpitch, dyaw`（弧度）+ `gripper`。当前臂**没有夹爪**，最后一维丢掉
- 相机：Tools → Robotics → ROS 2 OmniGraphs → ROS 2 Camera；固定相机彩色相机；外壳 `RSD455` 勾了 Kinematic Enabled（否则播放会被重力拉下去）
- 视角：固定相机从桌边斜看桌面，不是垂直朝下
- 底座：引用根 Translate 和 `root_joint` Local Position 在**停止播放时**改过，播放后停在桌边

---

## 还没做完（后续工作默认从这里接着）

1. 隔离测试已结：碰不到杯子按 zero-shot 域差处理，不要再拧 IK
2. Reset / 成功条件后补
3. **并行新线**：Isaac Sim 采数。`scripts/redo.py` 专家路径 P1–P7 运动已通（不经 ROS/OpenVLA）。**下一步：落盘 RGB+动作增量+指令**，再随机化/成功过滤。详见 `documents/README.md`「计划：Isaac Sim 采集微调数据」
4. 采数后再谈微调；部署评测仍走现有 ROS 闭环

`openvla_one_shot.py` 只证明「图进模型、7 个数出来」。控臂看 `openvla_closed_loop.py`（已会动，任务尚未成功）。过肩相机机位暂定完成。IK 链路已用 `ee_nudge` 三轴验证通过。

---

## 命名和资产上的坑

- 脚本和文档写 **UR12e**，资产目录是 **`assets/ur10e`**。对话里两者可能指同一套仿真臂，改代码前先对一下关节名和 prim 路径，不要按型号名想当然。
- 场景路径是 `assets/sence/`（拼写如此），不要「纠正」成 `scene` 除非用户明确要改名。
- 臂关节名按 UR 官方六轴：`shoulder_pan_joint` … `wrist_3_joint`。Isaac 可能带 prim 前缀，匹配用后缀（见 `ur12e_random_motion.py`）。
- 图像 QoS：相机订阅用 **BEST_EFFORT**；关节命令用 **RELIABLE**。不要随便改成同一套 QoS 然后抱怨收不到图。
- `openvla_one_shot.py` 自己实现了 json_numpy 编解码，避免客户端必须装 `json_numpy`。
- 指令默认英文、数据集键 `bridge_orig`。换任务先改指令和桌上物体，不要先改模型。

---

## 改代码时的默认态度

- 这是实验胶水，优先可读、可单独跑通的脚本，不要做成大框架。
- 中文进度写在 `documents/`；`.cursor/` 只放给后续 Agent 用的地图和约定。
- 不要把 OpenVLA 权重加载、训练循环拉进本目录。
- 闭环已把位移加成末端再 IK；不要把 7 个数直接当关节角发 `/joint_command`。
- 用户未说「同意修改增删」时，按仓库规则：仍可谨慎改，但不要把那句口令当默认授权。
