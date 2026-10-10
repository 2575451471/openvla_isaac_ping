# write by shenle
raw 录制 → 离线 actions（完整流程见 `documents/README.md`）

约定和闭环一样：`action_frame=ee`，RPY 左乘（FK 系，含 Isaac↔FK 的 Rz180）。

## 录
`scripts/isaac/redo.py` 里 `ENABLE_RAW_RECORD = True`，Play 后跑。  
写到 `datasets/raw/episode_XXXXXX/`（图 + 绝对位姿 + 关节；还没有 `actions.npy`；不含 END_HOLD）。

## 打标签
```bash
cd /home/dt/openvla/shenle_projects
python3 scripts/dataset_raw/make_actions.py datasets/raw/episode_000001
# 或
python3 scripts/dataset_raw/make_actions.py datasets/raw --all
```

得到 `(N-1, 7)` 的 `actions.npy`。四元数 wxyz，见 `meta.json`。

下一步：`scripts/rlds_ur10e_push_sim/` 建成 TFDS。
