# 模型动作的有限回退与 MoveIt2 评估

第二次第一步失败见 `logs/openvla_closed_loop_20261018-171715.log`。
这次模型输出与第一次不同，起点仍近伸直奇异位形。
原始目标不能通过当前求解器的 2 mm、1 度、20 度关节步长检查。
独立 SciPy 多初值求解在每关节变化不超过 20 度的范围内也未找到合格解；
允许更大关节变化时找到约 96 度腕部转动的精确解。这不是全局不可达性证明。
结果见 `logs/model_action2_scipy_probe_20261018.json`。

## 可选回退

控制器新增 `max_ik_backtracks`，默认 0，取值范围 0–3。
显式设为 2 时，依次尝试原动作、半步、四分之一步。
位移和 RPY 分量同比缩小，仍由同一起点构造目标，失败目标不累加、不执行。
每次均须通过原有 IK、旋转、关节步长和反馈检查；全部失败则停车。
仅 IK 规划拒绝触发回退，反馈过期等故障立即中止。
这会改变实际执行的模型动作，不能将回退后的成功解释为原动作可执行。

本次失败动作的半步离线结果：位置 1.335 mm、旋转 0.776 度、
最大关节变化 5.140 度。报告为
`logs/action_backtracking_model_probe_20261018.json`，没有控制仿真。
71 项测试通过，包括回退不累加、平移旋转同比缩放、失败次数有界、
过期反馈不回退，以及默认模式不缩放记录动作。

模型服务保持运行。在 Isaac 恢复 Play、执行原 `start_reset()` 后重试：

```bash
source /opt/ros/humble/setup.bash
cd /home/dt/openvla/shenle_projects
/usr/bin/python3 scripts/ros/openvla_closed_loop.py --ros-args \
  -p max_steps:=20 -p delta_frame:=ee -p interp_duration:=0.20 \
  -p max_trans_m:=0.03 -p max_rot_rad:=0.10 -p max_ik_backtracks:=2
```

`IK backtrack` 表示尝试更小目标；每步 `action_scale` 给出实际比例。
`Control stopped` 才表示控制中止。实测 20 步闭环仍待验证。
备份位于 `~/.local/state/openvla-action-backtracking/20261018-172455/`。

## MoveIt2

后续可评估 MoveIt Servo 接管连续末端命令及奇异位形、碰撞、关节限制处理。
它不会提高 OpenVLA 的动作预测精度，也不保证从奇异起点直接成功。
接入需要与当前 USD 一致的 URDF/SRDF、坐标系、关节反馈和输出控制接口；
推方块还需正确配置末端与方块的允许接触关系。
本次没有安装或切换 MoveIt2。

- [MoveIt Servo 官方文档，Humble](https://moveit.picknik.ai/humble/doc/examples/realtime_servo/realtime_servo_tutorial.html)
- [运动学与允许碰撞矩阵](https://moveit.picknik.ai/main/doc/concepts/kinematics.html)
