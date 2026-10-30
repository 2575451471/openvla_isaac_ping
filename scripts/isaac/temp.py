# Isaac Sim 5.1：UR10e + Robotiq 2F-85
# 在 Isaac Sim 内置 Script Editor 中运行。
# 运行前先单击 Stop。
#
# 功能：
# 1. 重新引用夹爪。
# 2. 检查夹爪内部关节和刚体。
# 3. 对齐腕部与夹爪 base_link。
# 4. 创建 Fixed Joint，合并到机械臂关节链。
# 5. 提供撤销和导出函数。
#
# 所有装配修改写入临时覆盖层，不覆盖原始资产。

import builtins
import os

import omni.usd
import omni.timeline

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


# ==================== 参数 ====================

ARM = "/World/ur10e"
WRIST = ARM + "/wrist_3_link"

ASSET = (
    "/home/dt/assets/isaac/5.1/Isaac/Robots/"
    "Robotiq/2F-85/payloads/Robotiq_2F_85_phyisics_mimic.usda"
)

# 新夹爪放在机械臂容器下，避免放进腕部刚体下面。
GRIPPER = ARM + "/Robotiq_script"
JOINT = ARM + "/script_gripper_fixed_joint"

BASE_LINK_NAME = "base_link"

# 夹爪 base_link 相对于腕部的位移，单位：米。
OFFSET_M = (0.0, 0.0, 0.0)

# 相对于腕部的 XYZ 旋转角，单位：度。
# 如果安装方向不对，可以修改，例如：
# ROTATE_XYZ_DEG = (0.0, 0.0, 90.0)
ROTATE_XYZ_DEG = (0.0, 0.0, 0.0)

# 根据你截图中的层级填写。
# 仅在本次装配层中暂时停用，撤销后恢复。
OLD_GRIPPERS = [
    ARM + "/wrist_3_link/Robotiq_2F_85_edit",
    "/World/Robotiq_2F_85_edit",
    "/World/Robotiq_2F_85",
]

# 禁止机械臂与夹爪互相碰撞。
# 不影响夹爪与外部物体的碰撞。
FILTER_ARM_GRIPPER_COLLISIONS = True

_KEY = "_ur10e_robotiq_script_state"


# ==================== 工具函数 ====================

def require_stop():
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise RuntimeError(
            "请先单击 Stop 停止仿真，再运行脚本。"
        )


def undo_gripper():
    """撤销本脚本操作，恢复到运行脚本之前。"""
    require_stop()

    record = getattr(builtins, _KEY, None)
    if record:
        session = record["stage"].GetSessionLayer()
        ident = record["layer"].identifier

        session.subLayerPaths = [
            path for path in session.subLayerPaths
            if path != ident
        ]

        setattr(builtins, _KEY, None)
        print("已撤销本脚本的装配，恢复到运行脚本前。")


def export_assembly(path):
    """将包含临时装配结果的当前场景导出为新 USD。"""
    require_stop()

    path = os.path.abspath(os.path.expanduser(path))

    if os.path.exists(path):
        raise RuntimeError(
            "目标文件已存在，请换一个新文件名：" + path
        )

    stage = omni.usd.get_context().get_stage()

    if not stage.Export(path):
        raise RuntimeError("导出失败：" + path)

    print("已导出包含当前装配的完整场景：", path)


def rigid_body(prim):
    """查找该 Prim 或其祖先上的有效刚体。"""
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)

            if api.GetRigidBodyEnabledAttr().Get() is False:
                raise RuntimeError(
                    "刚体被禁用：" + str(prim.GetPath())
                )

            return prim

        prim = prim.GetParent()

    return None


def world(prim):
    return UsdGeom.XformCache().GetLocalToWorldTransform(prim)


def within(path, roots):
    return any(
        path.HasPrefix(Sdf.Path(root))
        for root in roots
    )


# ==================== 装配 ====================

def assemble_gripper():
    require_stop()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开的场景。")

    previous = getattr(builtins, _KEY, None)

    if previous and previous["stage"] != stage:
        raise RuntimeError(
            "之前已在另一个场景运行本脚本，"
            "请先切回那个场景并执行 undo_gripper()。"
        )

    # 再次运行时，先撤销上一次由本脚本创建的装配。
    undo_gripper()

    arm = stage.GetPrimAtPath(ARM)
    mount = stage.GetPrimAtPath(WRIST)

    if not arm or not mount:
        raise RuntimeError(
            "找不到机械臂或腕部，请检查 ARM、WRIST。"
        )

    wrist_body = rigid_body(mount)
    if not wrist_body:
        raise RuntimeError(
            "腕部没有有效 RigidBodyAPI：" + WRIST
        )

    if rigid_body(arm):
        raise RuntimeError(
            "ARM 应指向机器人容器，不能是刚体 link。"
        )

    if not os.path.isfile(ASSET):
        raise RuntimeError("找不到夹爪 USD：" + ASSET)

    source = Usd.Stage.Open(ASSET)

    if not source or not source.GetDefaultPrim():
        raise RuntimeError(
            "夹爪入口缺少 Default Prim，请检查 ASSET。"
        )

    units = UsdGeom.GetStageMetersPerUnit(stage)
    asset_units = UsdGeom.GetStageMetersPerUnit(source)

    if abs(asset_units - units) > 1e-9:
        raise RuntimeError(
            "夹爪和场景的 metersPerUnit 不一致："
            f"夹爪={asset_units}，场景={units}。"
            "需要先统一单位。"
        )

    if (
        stage.GetPrimAtPath(GRIPPER)
        or stage.GetPrimAtPath(JOINT)
    ):
        raise RuntimeError(
            "GRIPPER 或 JOINT 路径已被占用，请修改参数。"
        )

    # 所有修改放入可整体移除的临时层。
    layer = Sdf.Layer.CreateAnonymous(
        "ur10e_robotiq_assembly.usda"
    )

    session = stage.GetSessionLayer()
    session.subLayerPaths.insert(0, layer.identifier)

    setattr(
        builtins,
        _KEY,
        {"stage": stage, "layer": layer},
    )

    try:
        with Usd.EditContext(stage, layer):

            # ---------- 暂时停用旧装配 ----------

            old = [
                path for path in OLD_GRIPPERS
                if stage.GetPrimAtPath(path)
            ]

            for path in old:
                if (
                    "robotiq" not in path.lower()
                    or path in (ARM, WRIST)
                ):
                    raise RuntimeError(
                        "旧夹爪路径不合理：" + path
                    )

            # 停用旧夹爪外部、但连接到旧夹爪的关节。
            for prim in list(stage.Traverse()):
                if (
                    prim.IsA(UsdPhysics.Joint)
                    and not within(prim.GetPath(), old)
                ):
                    joint = UsdPhysics.Joint(prim)

                    targets = (
                        joint.GetBody0Rel().GetTargets()
                        + joint.GetBody1Rel().GetTargets()
                    )

                    if any(within(t, old) for t in targets):
                        joint.CreateJointEnabledAttr(False)

            for path in old:
                stage.GetPrimAtPath(path).SetActive(False)
                print("临时停用旧夹爪：", path)

            # ---------- 检查机械臂 ----------

            arm_parts = list(Usd.PrimRange(arm))

            arm_bodies = [
                prim for prim in arm_parts
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]

            roots = [
                prim for prim in arm_parts
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]

            if len(roots) != 1:
                raise RuntimeError(
                    "机械臂应有一个 Articulation Root，目前为："
                    + str([str(p.GetPath()) for p in roots])
                )

            # ---------- 重新加载夹爪 ----------

            gripper = stage.DefinePrim(GRIPPER, "Xform")
            gripper.GetReferences().AddReference(ASSET)

            stage.Load(GRIPPER)

            # 在临时层中解开实例，方便编辑物理属性。
            while True:
                instances = [
                    prim for prim in Usd.PrimRange(gripper)
                    if prim.IsInstance()
                ]

                if not instances:
                    break

                for prim in instances:
                    prim.SetInstanceable(False)

            parts = list(Usd.PrimRange(gripper))

            bases = [
                prim for prim in parts
                if prim.GetName() == BASE_LINK_NAME
            ]

            if len(bases) != 1:
                raise RuntimeError(
                    "无法唯一找到夹爪 base_link，"
                    "请检查 BASE_LINK_NAME。候选："
                    + str([str(p.GetPath()) for p in bases])
                )

            base = bases[0]
            base_body = rigid_body(base)

            if (
                not base_body
                or not base_body.GetPath().HasPrefix(
                    gripper.GetPath()
                )
            ):
                raise RuntimeError(
                    "夹爪 base_link 没有自身的有效刚体。"
                )

            bodies = [
                prim for prim in parts
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]

            joints = [
                prim for prim in parts
                if prim.IsA(UsdPhysics.Joint)
            ]

            if not bodies or not joints:
                raise RuntimeError(
                    "夹爪未加载完整的刚体和关节。"
                )

            # 使用机械臂的 Articulation Root。
            for prim in parts:
                if prim.HasAPI(
                    UsdPhysics.ArticulationRootAPI
                ):
                    prim.RemoveAPI(
                        UsdPhysics.ArticulationRootAPI
                    )

            for prim in bodies:
                rigid_body(prim)

                # 附着后的夹爪由物理关节驱动。
                UsdPhysics.RigidBodyAPI(
                    prim
                ).CreateKinematicEnabledAttr(False)

                # 检查内部是否存在非法父子刚体嵌套。
                xf = UsdGeom.Xformable(prim)
                parent = prim.GetParent()
                reset = xf.GetResetXformStack()

                while (
                    parent
                    and not parent.IsPseudoRoot()
                    and not reset
                ):
                    if parent.HasAPI(
                        UsdPhysics.RigidBodyAPI
                    ):
                        raise RuntimeError(
                            "夹爪存在未隔离的父子刚体嵌套："
                            + str(prim.GetPath())
                        )

                    px = UsdGeom.Xformable(parent)
                    reset = bool(
                        px and px.GetResetXformStack()
                    )
                    parent = parent.GetParent()

            # ---------- 检查夹爪内部关节 ----------

            for prim in joints:
                joint = UsdPhysics.Joint(prim)

                if (
                    joint.GetJointEnabledAttr().Get()
                    is False
                ):
                    continue

                resolved = []

                for rel in (
                    joint.GetBody0Rel(),
                    joint.GetBody1Rel(),
                ):
                    targets = rel.GetTargets()

                    if len(targets) > 1:
                        raise RuntimeError(
                            "关节一端有多个目标："
                            + str(prim.GetPath())
                        )

                    if not targets:
                        resolved.append(None)
                        continue

                    target = stage.GetPrimAtPath(
                        targets[0]
                    )

                    if not target:
                        raise RuntimeError(
                            "关节目标不存在："
                            + str(prim.GetPath())
                            + " -> "
                            + str(targets[0])
                        )

                    resolved.append(rigid_body(target))

                body0, body1 = resolved
                live = [p for p in resolved if p]

                # 解除夹爪底座原来连接世界的固定关节。
                if (
                    len(live) == 1
                    and live[0] == base_body
                    and prim.IsA(UsdPhysics.FixedJoint)
                ):
                    joint.CreateJointEnabledAttr(False)
                    print(
                        "解除夹爪底座的世界固定关节：",
                        prim.GetPath(),
                    )
                    continue

                if (
                    not body0
                    or not body1
                    or body0 == body1
                ):
                    raise RuntimeError(
                        "夹爪内部关节的两端无效："
                        + str(prim.GetPath())
                    )

                if any(
                    not p.GetPath().HasPrefix(
                        gripper.GetPath()
                    )
                    for p in (body0, body1)
                ):
                    raise RuntimeError(
                        "夹爪内部关节连接到了资产外部："
                        + str(prim.GetPath())
                    )

            # ---------- 对齐夹爪与腕部 ----------

            # USD 矩阵采用行向量约定。
            offset = Gf.Matrix4d(1)

            axes = (
                Gf.Vec3d(1, 0, 0),
                Gf.Vec3d(0, 1, 0),
                Gf.Vec3d(0, 0, 1),
            )

            for axis, angle in zip(
                axes, ROTATE_XYZ_DEG
            ):
                offset = (
                    offset
                    * Gf.Matrix4d(1).SetRotate(
                        Gf.Rotation(axis, angle)
                    )
                )

            offset.SetTranslateOnly(
                Gf.Vec3d(
                    *(value / units for value in OFFSET_M)
                )
            )

            desired = offset * world(mount)
            delta = world(base).GetInverse() * desired

            # 同时处理内部设置了 Reset Xform Stack 的节点。
            islands = [gripper] + [
                prim for prim in parts
                if (
                    prim != gripper
                    and UsdGeom.Xformable(prim)
                    and UsdGeom.Xformable(
                        prim
                    ).GetResetXformStack()
                )
            ]

            poses = [
                (prim, world(prim) * delta)
                for prim in islands
            ]

            for prim, pose in poses:
                xf = UsdGeom.Xformable(prim)
                reset = xf.GetResetXformStack()

                local = (
                    pose
                    if reset
                    else (
                        pose
                        * world(
                            prim.GetParent()
                        ).GetInverse()
                    )
                )

                xf.MakeMatrixXform().Set(local)
                xf.SetResetXformStack(reset)

            # ---------- 创建装配固定关节 ----------

            joint = UsdPhysics.FixedJoint.Define(
                stage, JOINT
            )

            joint.CreateBody0Rel().SetTargets(
                [wrist_body.GetPath()]
            )

            joint.CreateBody1Rel().SetTargets(
                [base_body.GetPath()]
            )

            joint.CreateJointEnabledAttr(True)
            joint.CreateExcludeFromArticulationAttr(False)
            joint.CreateCollisionEnabledAttr(False)

            # 计算两端一致的关节局部坐标，避免启动时拉扯。
            frames = (
                (
                    wrist_body,
                    joint.CreateLocalPos0Attr(),
                    joint.CreateLocalRot0Attr(),
                ),
                (
                    base_body,
                    joint.CreateLocalPos1Attr(),
                    joint.CreateLocalRot1Attr(),
                ),
            )

            for body, pos_attr, rot_attr in frames:
                local = (
                    desired * world(body).GetInverse()
                )

                pos_attr.Set(
                    Gf.Vec3f(local.ExtractTranslation())
                )

                rot_attr.Set(
                    Gf.Quatf(
                        local.ExtractRotationQuat()
                        .GetNormalized()
                    )
                )

            # ---------- 碰撞过滤 ----------

            if FILTER_ARM_GRIPPER_COLLISIONS:
                for body in bodies:
                    rel = (
                        UsdPhysics.FilteredPairsAPI
                        .Apply(body)
                        .CreateFilteredPairsRel()
                    )

                    for other in arm_bodies:
                        rel.AddTarget(other.GetPath())

            # ---------- 更新已有机器人关系表 ----------

            additions_by_name = (
                (
                    "isaac:physics:robotLinks",
                    bodies,
                ),
                (
                    "isaac:physics:robotJoints",
                    joints + [joint.GetPrim()],
                ),
            )

            for name, additions in additions_by_name:
                rel = arm.GetRelationship(name)

                if not rel:
                    continue

                targets = [
                    target for target in rel.GetTargets()
                    if not within(target, old)
                ]

                for prim in additions:
                    if (
                        prim.IsA(UsdPhysics.Joint)
                        and UsdPhysics.Joint(prim)
                        .GetJointEnabledAttr().Get()
                        is False
                    ):
                        continue

                    if prim.GetPath() not in targets:
                        targets.append(prim.GetPath())

                rel.SetTargets(targets)

        print(
            "装配结构已创建：",
            WRIST,
            "->",
            base_body.GetPath(),
        )

        print("请点击 Play 检查物理仿真。")
        print("撤销：停止仿真后运行 undo_gripper()")
        print(
            "保存：停止仿真后运行 "
            'export_assembly('
            '"/home/dt/ur10e_robotiq_assembled.usd")'
        )

    except Exception:
        undo_gripper()
        raise


# 保存在 builtins 中，方便在其他 Script Editor 标签页调用。
builtins.undo_gripper = undo_gripper
builtins.export_assembly = export_assembly

assemble_gripper()
