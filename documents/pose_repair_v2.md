# 姿态与动作标签修复：第一步

2026-10-16 的离线检查确认，旧采数将 USD 行向量旋转矩阵直接转换为列向量约定的四元数。平移记录准确，旋转记录需要取逆，因此已有 `actions.npy` 不能直接用于当前闭环控制器。

## 数据与代码

- `datasets/raw/` 保留旧数据，不覆盖；旧模型权重也保留。
- `datasets/raw_pose_v2/` 是修复后的成功示范副本。位姿、动作、元数据独立保存；`images/` 链接到旧数据中的原图，因此不要删除旧目录。
- 元数据使用 `pose_convention=world_from_local_column_v2`；生成动作时同时记录 `actions_pose_convention`，防止重复修正或混用格式。
- 新采数的 `redo.py`、`collect_push_batch.py` 默认写入新目录。

修复命令只允许创建不存在的目标目录；失败示范不复制到新训练数据集。所有动作必须先通过执行端 FK 校验，才会发布目标目录：

```bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 scripts/dataset_raw/repair_pose_dataset.py datasets/raw datasets/raw_pose_v2
```

重复执行会拒绝覆盖，不能把已修复的数据再次取逆。

## 验证

```bash
/usr/bin/python3 -m unittest discover -s tests -p 'test_pose_conventions.py' -v
/usr/bin/python3 scripts/dataset_raw/verify_pose_dataset.py datasets/raw_pose_v2
```

验证脚本只提取 `scripts/ros/ee_nudge.py` 的纯 FK 数学定义，不启动 ROS，不发送机械臂命令。它检查每个成功示范的关节/位姿/时间/动作/图像对应关系，以及用动作重建下一帧位姿的误差。位置与角度最大误差均需小于 0.01 毫米、0.01 度。

修复结果中的 `verification_report.json` 记录误差，`repair_manifest.json` 记录来源、原文件 SHA-256 和跳过的失败示范。

## 接下来的步骤

第一步通过不等于模型或仿真任务已修好。当前旧 checkpoint 对不同图像输出相同动作的问题仍需处理。

下一步应使用修复后的示范验证同一控制器的动作回放，并对齐执行节奏。之后在独立 TFDS/训练输出目录中重建数据和统计、训练和评估。不要直接重跑旧的 `run_ft_pipeline.sh`：它仍使用旧数据路径、旧完成标记和旧训练输出目录，后续需单独更新。

Isaac 已导入的 Python 模块不会因磁盘文件修改而自动刷新。下一次采数或回放前应重启 Isaac，确保加载新代码。
