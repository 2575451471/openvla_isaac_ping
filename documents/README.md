# write by shenle
关于微调 LoRA（权威流程）

## 收集数据
先收集 episode（目前是手动设计的一条成功轨迹）。

Isaac 打开 `assets/sence/sence1.usd`（目录名是 `sence`）→ Script Editor 打开 `scripts/isaac/redo.py` → Play 后跑脚本。  
`ENABLE_RAW_RECORD = True` 时写到 `datasets/raw/episode_000xxx/`。

打动作标签：

```bash
cd /home/dt/openvla/shenle_projects
python3 scripts/dataset_raw/make_actions.py datasets/raw/episode_000xxx
# 多个 episode：
# python3 scripts/dataset_raw/make_actions.py datasets/raw --all
```

得到 `actions.npy`。

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
