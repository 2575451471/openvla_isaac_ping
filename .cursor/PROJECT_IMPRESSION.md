# shenle_projects 印象（给 Agent）

权威操作流程：`documents/README.md`。这里只记地图和约定。

- 路径：`/home/dt/openvla/shenle_projects`，挂在上游 `openvla` 下的本机实验目录，不是训练仓库本体
- 栈：Isaac Sim 5.1 + ROS 2 Humble + 本机 OpenVLA HTTP（`127.0.0.1:8000`）。OpenVLA 不进仿真
- `/act` 返回末端 7 维增量，不是关节角。闭环：`delta_frame=ee`，RPY 左乘再 IK
- 采数：`scripts/isaac/redo.py` + `ENABLE_RAW_RECORD` → `datasets/raw` → `make_actions.py` → `rlds_env` 的 `tfds build` → `ur10e_push_sim` → LoRA
- 推理：`deploy.py --openvla_path <微调目录>`；闭环 `scripts/ros/openvla_closed_loop.py`，`unnorm_key:=ur10e_push_sim`
- scripts 分包：`isaac/` / `ros/` / `dataset_raw/` / `rlds_ur10e_push_sim/` / `legacy/`（见 `scripts/README.md`）
- 场景目录拼写是 `sence`；资产是 `assets/ur10e`（文档里可能写 UR12e，改代码先对 prim）
- 启动顺序：先 OpenVLA（约 15GB）再 Isaac。`CUDA_VISIBLE_DEVICES=` 空字符串会藏掉 GPU，用 `unset`
- 脚本独立，别做成大框架；训练/权重加载留在上层 `openvla`
