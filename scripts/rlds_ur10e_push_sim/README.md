# write by shenle
raw → TFDS `ur10e_push_sim`（完整流程见 `documents/README.md`）

用 conda `rlds_env`，别和 `openvla` / Isaac 混。

## 依赖（装过可跳过）
```bash
conda activate rlds_env
pip install "tensorflow==2.18.0" "tensorflow-datasets==4.9.7" "protobuf==6.31.1" pillow mlcroissant \
  -i https://pypi.org/simple
```

清华源 SSL 坏了就加代理 + 官方 PyPI。

## 构建
```bash
conda activate rlds_env
export UR10E_PUSH_RAW=/home/dt/openvla/shenle_projects/datasets/raw
export TFDS_DATA_DIR=$HOME/tensorflow_datasets
export CUDA_VISIBLE_DEVICES=
export NO_GCE_CHECK=true
cd /home/dt/openvla/shenle_projects/scripts/rlds_ur10e_push_sim/ur10e_push_sim
tfds build --overwrite
```

产物：`$TFDS_DATA_DIR/ur10e_push_sim/1.0.0/`。  
图尺寸写死 720×1280；改分辨率要改 builder 里 `IMAGE_SHAPE`。  
注册片段见同目录 `OPEN_VLA_REGISTER.md`（上游一般已加好）。
