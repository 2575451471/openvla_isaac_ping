# shenle_projects 印象（给 Agent）

权威操作流程：`documents/README.md`。这里只记地图和约定。

- 路径：`/home/dt/openvla/shenle_projects`，挂在上游 `openvla` 下的本机实验目录，不是训练仓库本体
- 栈：Isaac Sim 5.1 + ROS 2 Humble + 本机 OpenVLA HTTP（`127.0.0.1:8000`）。OpenVLA 不进仿真
- `/act` 返回末端 7 维增量，不是关节角。闭环：`delta_frame=ee`，RPY 左乘再 IK；旋转按采数软约束（3mm / 20°），默认 `ik()` 仍是 2mm / 1°
- 采数：`HOME_DEG=(0,-88.4,80.1,0,254.2,0)`，P1=HOME 末端；IK/运动在 `scripts/isaac/arm_motion.py`；批量 `collect_push_batch.py`（成功=从桌子右边/−X 推下）→ `datasets/raw_pose_v2` → `make_actions.py` → TFDS `ur10e_push_sim` 2.0.0 → 4bit LoRA。权威步骤 `documents/home_v3_recollect.md`
- 推理：`deploy.py --openvla_path <微调目录>`；闭环 `scripts/ros/openvla_closed_loop.py`，`unnorm_key:=ur10e_push_sim`
- scripts 分包：`isaac/`（含 `arm_motion`）/ `ros/` / `dataset_raw/` / `rlds_ur10e_push_sim/` / `legacy/`（见 `scripts/README.md`）
- 场景目录拼写是 `sence`；资产是 `assets/ur10e`（文档里可能写 UR12e，改代码先对 prim）
- 启动顺序：先 OpenVLA（约 15GB）再 Isaac。`CUDA_VISIBLE_DEVICES=` 空字符串会藏掉 GPU，用 `unset`
- 自己操作 Isaac 时用 `scripts/isaac/start_pose_v2_webrtc.sh` 的 streaming，不要在本机桌面开窗口
- 脚本独立，别做成大框架；训练/权重加载留在上层 `openvla`
