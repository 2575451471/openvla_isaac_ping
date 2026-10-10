# write by shenle
4090 上 LoRA 的坑（主流程看 `documents/README.md`）

- 环境用 `openvla`，不是 `rlds_env`
- **不要** `export CUDA_VISIBLE_DEVICES=`，用 `unset CUDA_VISIBLE_DEVICES`
- 权重用本地路径 + `HF_HUB_OFFLINE=1`，否则会卡 HuggingFace
- 缺包时（装过可跳过）：

```bash
conda activate openvla
cd /home/dt/openvla
pip install -e . --no-deps
pip install "peft==0.11.1" wandb rich "Pillow==10.4.0" "sentencepiece==0.1.99" "draccus==0.8.0" \
  "tensorflow==2.15.0" "tensorflow-datasets==4.9.3" "protobuf==6.31.1" \
  --no-deps "git+https://github.com/moojink/dlimp_openvla" -i https://pypi.org/simple
pip install "accelerate==0.33.0" -i https://pypi.org/simple
python /home/dt/openvla/shenle_projects/scripts/rlds_ur10e_push_sim/smoke_openvla_train_env.py
```

看到 `SMOKE_OK` 即可。`MessageFactory` 警告可忽略。  
先 `max_steps 20` 冒烟，再 500。不要 `--use_quantization`。
