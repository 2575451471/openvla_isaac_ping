# write by shenle
> 2026-10-27：当前采数/训练见 `documents/home_v3_recollect.md`（非奇异 HOME，覆盖 raw_pose_v2 与 TFDS 2.0.0，4bit 重训）。
> 2026-10-16：已确认旧采数的姿态约定错误。修复与当时执行步骤见 `documents/pose_repair_v2.md`。
> 下方旧数据、训练及权重路径为历史流程；不要直接重跑旧训练流水线。

关于微调 LoRA（权威流程）

## 收集数据

成功定义：**把红色方块从桌子右边（世界 −X）推下/推出**。掉下桌可以；从其它边掉下算失败。

运动/IK 共用库：`scripts/isaac/arm_motion.py`（`redo.py` / `collect_push_batch.py` 都调用它）。

### 单次手调（模板路径）
Isaac 打开 `assets/sence/sence1.usd`（目录名是 `sence`）→ Play → Script Editor 跑 `scripts/isaac/redo.py`。  
`ENABLE_RAW_RECORD = True` 时写到 `datasets/raw/episode_000xxx/`。

### 批量采数（推荐）
同一场景 Play 后跑 `scripts/isaac/collect_push_batch.py`：
- 每轮 reset 方块 + 臂回 home，按方块位姿生成接近/沿 −X 推程航点
- 成功门控：过桌子右缘且 |y| 仍在桌宽内（可掉下桌）
- 旋钮：`TARGET_SUCCESSES` / `MAX_ATTEMPTS` / `CUBE_XY_NOISE` / `PUSH_DISTANCE_RANGE` / `TABLE_X_RIGHT`

打动作标签：

```bash
cd /home/dt/openvla/shenle_projects
python3 scripts/dataset_raw/make_actions.py datasets/raw/episode_000xxx
# 多个 episode：
python3 scripts/dataset_raw/make_actions.py datasets/raw --all
```

得到 `actions.npy`。TFDS 会跳过 `meta.success=false` 的 episode。

## 转成 OpenVLA 的 RLDS
```bash
conda activate rlds_env
export UR10E_PUSH_RAW=/home/dt/openvla/shenle_projects/datasets/raw
export TFDS_DATA_DIR=$HOME/tensorflow_datasets
export CUDA_VISIBLE_DEVICES=
export NO_GCE_CHECK=true
cd /home/dt/openvla/shenle_projects/scripts/rlds_ur10e_push_sim/ur10e_push_sim
tfds build --overwrite
```

产物：`~/tensorflow_datasets/ur10e_push_sim/1.0.0/`  
数据集名固定 `ur10e_push_sim`（上游 `configs.py` / `transforms.py` 已注册）。

## LoRA 微调
先关 Isaac（腾显存）。**不要** `export CUDA_VISIBLE_DEVICES=`（空字符串会让 PyTorch 找不到 GPU），用 `unset`。

```bash
conda activate openvla
unset CUDA_VISIBLE_DEVICES
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TFDS_DATA_DIR=$HOME/tensorflow_datasets
export VLA_PATH=$HOME/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
cd /home/dt/openvla
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path "$VLA_PATH" --data_root_dir $HOME/tensorflow_datasets --dataset_name ur10e_push_sim --run_root_dir /home/dt/openvla/shenle_projects/runs/ft_one_ep --adapter_tmp_dir /home/dt/openvla/shenle_projects/runs/adapter-tmp --lora_rank 32 --batch_size 1 --grad_accumulation_steps 1 --learning_rate 5e-4 --max_steps 500 --save_steps 100 --image_aug False --shuffle_buffer_size 256
```

常改：`max_steps` / `save_steps`。本机 4090 用 `batch_size 1`，不要加 `--use_quantization`（容易炸）。  
权重写到 `runs/ft_one_ep/<一长串>/`。环境坑见 `documents/openvla_lora_4090.md`。

## 推理
顺序：OpenVLA → Isaac 串流 Play → 闭环脚本。

### 终端1（OpenVLA）
```bash
conda activate openvla
unset CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/dt/openvla
python vla-scripts/deploy.py \
  --openvla_path /home/dt/openvla/shenle_projects/runs/ft_one_ep/47a0ec7fc4ec123775a391911046cf33cf9ed83f+ur10e_push_sim+b1+lr-0.0005+lora-r32+dropout-0.0
```

`--openvla_path` = 微调产出目录（含 `dataset_statistics.json`）。  
`curl http://127.0.0.1:8000/docs` 应 200。模型约占 15GB，加载完再开仿真。

### 终端2（Isaac）
```bash
cd ~/isaacsim5.1/
./start_isaac_stream.sh
```

WebRTC 客户端连 `100.64.61.8`，打开 `sence1.usd`，Play。

### 终端3（闭环）
```bash
source /opt/ros/humble/setup.bash
cd ~/openvla/shenle_projects/scripts/ros/
python3 openvla_closed_loop.py --ros-args \
  -p unnorm_key:=ur10e_push_sim \
  -p instruction:="push the red cube to the right"
```

`unnorm_key` 必须是 `ur10e_push_sim`（不是 `bridge_orig`）。指令尽量和采数一致。  
脚本不强制回 home，从当前关节角开推。
