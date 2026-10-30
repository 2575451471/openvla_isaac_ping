# 修正数据的 TFDS 构建与训练／验证划分

输入为 `datasets/raw_pose_v2` 的 26 条成功演示。新数据保持名称
`ur10e_push_sim`，版本为 `2.0.0`，输出根目录为
`/home/dt/tensorflow_datasets_pose_v2`。

## 划分原则

按完整 episode 固定划分为 22 条 `train`、4 条 `val`。
ID 排序后使用固定字符串种子进行 SHA-256 排序；划分与源文件指纹写入
`datasets/pose_v2_preparation/split_manifest.json`，重复运行不会随机重分。
原有 OpenVLA 加载器识别 `val`；命名为 `validation` 会被它忽略。
同一 episode 不会跨集合；此验证集仍来自相同场景和采集方式，不代表新场景泛化能力。

## 运行

在 dt_wl 终端执行，无需 Isaac Sim 或模型服务：

```bash
cd /home/dt/openvla/shenle_projects
/home/dt/miniconda3/envs/rlds_env/bin/python scripts/dataset_raw/prepare_pose_v2_tfds.py --plan
```

这会检查修正标记、数组、图片、FK／动作一致性，并保存不可自动覆盖的划分清单。
源文件变化后会拒绝复用该清单，避免混入其他数据。

可选：首次用小样本测量资源，每个集合只构建一条演示：

```bash
/home/dt/miniconda3/envs/rlds_env/bin/python scripts/dataset_raw/prepare_pose_v2_tfds.py --pilot
```

试转换使用独立的 `datasets/pose_v2_preparation/pilot_tfds/`，不会作为正式训练数据。
正式构建：

```bash
/home/dt/miniconda3/envs/rlds_env/bin/python scripts/dataset_raw/prepare_pose_v2_tfds.py --build
```

构建后自动逐条核对集合成员、动作、状态、指令、RLDS 标记及每张图片的原始字节，
并抽查每条演示首／中／尾图片解码。成功显示 `TFDS_CHECK_PASSED`。
单独重新校验已有正式数据：

```bash
/home/dt/miniconda3/envs/rlds_env/bin/python scripts/dataset_raw/prepare_pose_v2_tfds.py --verify
```

## 资源与输出

所有模式都显式禁用 GPU，并将进程及其线程限制在最多 2 个可用逻辑核。
PNG 直接交给 TFDS 编码器，避免把整条演示的全部 RGB 像素展开驻留内存。
资源测量和校验报告写入 `datasets/pose_v2_preparation/`；既有 TFDS 输出不会被覆盖。

正式结果位于 `~/tensorflow_datasets_pose_v2/ur10e_push_sim/2.0.0/`。
原始数据及旧 `~/tensorflow_datasets/ur10e_push_sim/1.0.0/` 保留。
修正 raw 的图片仍链接旧 raw，当前不能删除旧 `datasets/raw/`。

此流程只准备数据，不启动训练。旧 `scripts/temp/run_ft_pipeline.sh` 仍使用旧数据路径和训练配置，
本阶段请使用上述专用命令。下一阶段还需让训练归一化统计只使用 `train`：
当前 OpenVLA 加载器默认用 `all` 计算统计，不能直接假定新验证集已从统计计算中排除。
