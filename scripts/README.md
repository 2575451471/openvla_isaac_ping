# write by shenle
scripts 目录

- `isaac/`：Isaac Script Editor 跑
  - `arm_motion.py`：UR10e FK/IK/样条/航点跟踪（共用库）
  - `redo.py`：单次手调 P1–P7 采数
  - `collect_push_batch.py`：批量 reset + 相对方块推程；成功=从桌子右边（−X）推下
- `ros/`：`source /opt/ros/humble` 后跑（闭环 / 点动 / 存图）
- `dataset_raw/`：离线打 `actions.npy`
- `rlds_ur10e_push_sim/`：`tfds build` 成 OpenVLA 数据集
- `legacy/`：早期试探，一般不用
- `rgb_capture/`：临时截图输出（gitignore）

主流程见 `documents/README.md`。
