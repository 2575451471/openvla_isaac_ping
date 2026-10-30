#!/bin/bash
# Direct execution and `bash thisfile` both land in Isaac's Python.
# The next line is a no-op string for Python and an exec for bash.
"exec" "/home/dt/isaacsim5.1/python.sh" "$0" "$@"
"""Open the push scene in Isaac, hold the new HOME pose, mark P1-P7, save views, exit.

    ./scripts/temp/preview_home_isaac.py

The process closes Isaac when the pictures are written. It does not save the USD.
"""

import json
import math
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "width": 1280, "height": 720, "sync_loads": True, "renderer": "RaytracedLighting"}
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/preview_home_isaac"
STAGE = ROOT / "assets/sence/sence1_gripper.usd"
HOME_DEG = (0.0, -88.4, 80.1, 0.0, 254.2, 0.0)
CUBE_XYZ = (0.0, 0.0, 0.040)
P3 = (0.45119, 0.16336, 0.56206)
P4 = (0.26073, 0.02425, 0.34563)
CONTACT_Z = 0.065
PUSH_DISTANCE = 0.70
SHOULDER_CAM = "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
WIDTH, HEIGHT = 1280, 720

COLORS = {
    "P1": (0.15, 0.85, 0.25),
    "P2": (0.95, 0.78, 0.10),
    "P3": (0.95, 0.50, 0.08),
    "P4": (0.92, 0.28, 0.16),
    "P5": (0.10, 0.80, 0.90),
    "P6": (0.20, 0.40, 0.95),
    "P7": (0.78, 0.18, 0.85),
}
LABELS = {
    "P1": "P1 起点（当前手的位置）",
    "P2": "P2 是 P1 与 P3 的中点",
    "P3": "P3 往下绕",
    "P4": "P4 接近桌面",
    "P5": "P5 方块外侧",
    "P6": "P6 贴上方块",
    "P7": "P7 沿 -X 推过桌沿",
}


def nominal_waypoints(p1):
    cx, cy, _ = CUBE_XYZ
    p1 = tuple(float(value) for value in p1)
    p2 = tuple((a + b) / 2.0 for a, b in zip(p1, P3))
    points = [("P1", *p1), ("P2", *p2), ("P3", *P3), ("P4", *P4)]
    points.append(("P5", cx + 0.26, cy, CONTACT_Z + 0.015))
    points.append(("P6", cx + 0.015, cy, CONTACT_Z))
    points.append(("P7", cx - PUSH_DISTANCE, cy, CONTACT_Z))
    return points


def look_quat(eye, target):
    from isaacsim.core.utils.rotations import gf_quat_to_np_array, lookat_to_quatf
    from pxr import Gf

    # Argument order matches camera_utils: target first, so USD -Z faces the target.
    quat = lookat_to_quatf(Gf.Vec3f(*target), Gf.Vec3f(*eye), Gf.Vec3f(0.0, 0.0, 1.0))
    return gf_quat_to_np_array(quat)


def add_marker(path, xyz, color, radius):
    from isaacsim.core.api.objects import VisualSphere
    import numpy as np

    VisualSphere(
        prim_path=path,
        name=path.rsplit("/", 1)[-1],
        position=np.asarray(xyz, dtype=float),
        radius=radius,
        color=np.asarray(color, dtype=float),
    )


def add_camera(path, eye, target, focal):
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(float(focal))
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 80.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    quat = look_quat(eye, target)
    matrix = Gf.Matrix4d().SetRotateOnly(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
    matrix.SetTranslateOnly(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
    xform.AddTransformOp().Set(matrix)
    return path


def settle_home(robot, arm_idx, q_home, frames):
    from isaac.arm_motion import read_arm_q, send_arm_target
    import numpy as np

    error = None
    for _ in range(frames):
        send_arm_target(robot, arm_idx, q_home)
        simulation_app.update()
        try:
            error = float(np.max(np.abs(read_arm_q(robot, arm_idx) - q_home)))
        except Exception:
            error = None
    return error


def capture(camera, points, hold):
    import numpy as np

    image = None
    for _ in range(12):
        hold()
        simulation_app.update()
        frame = camera.get_rgba()
        if frame is None:
            continue
        array = np.asarray(frame)
        if array.size == 0 or array.ndim != 3:
            continue
        image = array[:, :, :3].copy()
        if float(image.mean()) > 2.0:
            break
    coords = None
    if image is not None:
        try:
            coords = np.asarray(
                camera.get_image_coords_from_world_points(np.asarray(points, dtype=float)), dtype=float
            )
        except Exception:
            coords = None
    return image, coords


def load_font(size):
    from PIL import ImageFont

    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(path, size, index=0)


def annotate(image, names, coords):
    from PIL import Image, ImageDraw

    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    font = load_font(22)
    small = load_font(18)
    width, height = picture.size
    if coords is not None:
        for name, xy in zip(names, coords):
            if xy.shape[0] < 2 or not math.isfinite(float(xy[0])) or not math.isfinite(float(xy[1])):
                continue
            x, y = float(xy[0]), float(xy[1])
            if not (-40 <= x <= width + 40 and -40 <= y <= height + 40):
                continue
            color = tuple(int(channel * 255) for channel in COLORS[name])
            radius = 10
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
            draw.text((x + 12, y - 12), name, fill=color, font=font)
    legend = "绿P1起点  黄P2中点  橙P3下降  红P4近桌  青P5方块外  蓝P6贴块  紫P7沿-X推出"
    draw.rectangle((0, height - 40, width, height), fill=(0, 0, 0))
    draw.text((12, height - 32), legend, fill=(255, 255, 255), font=small)
    return picture


def montage(frames):
    from PIL import Image, ImageDraw

    cell_w, cell_h = 640, 390
    columns = 2
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = load_font(20)
    for index, (title, picture) in enumerate(frames):
        thumb = picture.resize((cell_w, cell_h - 30))
        column, row = index % columns, index // columns
        origin = (column * cell_w, row * cell_h)
        sheet.paste(thumb, origin)
        draw.text((origin[0] + 8, origin[1] + cell_h - 26), title, fill=(255, 255, 255), font=font)
    return sheet


def main():
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.stage import is_stage_loading
    from isaacsim.sensors.camera import Camera

    sys.path.insert(0, str(ROOT / "scripts"))
    from isaac.arm_motion import (  # noqa: E402
        EE_PRIM,
        ROBOT_PRIM,
        find_arm_indices,
        get_world_position,
        resolve_cube_prim,
        set_prim_world_translation,
    )
    from isaacsim.core.prims import SingleArticulation

    OUT.mkdir(parents=True, exist_ok=True)
    if not STAGE.is_file():
        raise FileNotFoundError(STAGE)

    context = omni.usd.get_context()
    context.open_stage(str(STAGE))
    simulation_app.update()
    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(40):
        simulation_app.update()

    cube = resolve_cube_prim(None)
    set_prim_world_translation(cube, np.asarray(CUBE_XYZ, dtype=float))
    for _ in range(20):
        simulation_app.update()

    robot = SingleArticulation(prim_path=ROBOT_PRIM)
    robot.initialize()
    arm_idx = find_arm_indices(list(robot.dof_names))
    from isaac.arm_motion import send_arm_target

    q_home = np.radians(np.asarray(HOME_DEG, dtype=float))

    def hold():
        send_arm_target(robot, arm_idx, q_home)

    joint_error = settle_home(robot, arm_idx, q_home, 180)
    for _ in range(10):
        hold()
        simulation_app.update()

    p1 = get_world_position(EE_PRIM)
    waypoints = nominal_waypoints(p1)
    for name, x, y, z in waypoints:
        add_marker(f"/World/preview/{name}", (x, y, z), COLORS[name], 0.045 if name == "P1" else 0.035)
    for _ in range(10):
        hold()
        simulation_app.update()

    look = (0.15, 0.30, 0.50)
    views = [
        ("quarter", "斜上方：整条臂和七个球", (1.85, 1.75, 1.55), look, 16.0),
        ("side", "从 +X 看：肘是否弯着", (2.35, 0.25, 1.05), (0.05, 0.40, 0.55), 16.0),
        ("top", "顶视：球在桌面上的平面位置", (0.05, 0.28, 3.15), (0.05, 0.28, 0.10), 18.0),
        ("front", "桌子对面看向机械臂", (0.10, -1.75, 1.15), (0.15, 0.50, 0.60), 16.0),
        ("elbow", "近看肘和手腕", (1.05, 1.15, 1.05), (0.40, 0.70, 0.75), 24.0),
    ]
    names = [item[0] for item in waypoints]
    xyz = [[item[1], item[2], item[3]] for item in waypoints]
    saved = []
    annotated = []
    for key, title, eye, target, focal in views:
        prim_path = add_camera(f"/World/preview/cam_{key}", eye, target, focal)
        camera = Camera(prim_path=prim_path, name=key, frequency=30, resolution=(WIDTH, HEIGHT))
        camera.initialize()
        image, coords = capture(camera, xyz, hold)
        if image is None:
            raise RuntimeError(f"{key} camera returned no image")
        picture = annotate(image, names, coords)
        path = OUT / f"{key}.png"
        picture.save(path)
        saved.append(str(path))
        annotated.append((title, picture))
        print("WROTE", path, flush=True)

    stage = context.get_stage()
    shoulder_prim = stage.GetPrimAtPath(SHOULDER_CAM)
    if shoulder_prim.IsValid() and shoulder_prim.GetTypeName() == "Camera":
        shoulder = Camera(prim_path=SHOULDER_CAM, name="shoulder", frequency=30, resolution=(WIDTH, HEIGHT))
        shoulder.initialize()
        image, coords = capture(shoulder, xyz, hold)
        if image is not None and float(image.mean()) > 2.0:
            picture = annotate(image, names, coords)
            path = OUT / "shoulder.png"
            picture.save(path)
            saved.append(str(path))
            annotated.append(("场景里的肩部相机", picture))
            print("WROTE", path, flush=True)

    sheet = montage(annotated)
    overview = OUT / "overview.png"
    sheet.save(overview)
    report = {
        "home_deg": list(HOME_DEG),
        "joint_error_deg": None if joint_error is None else round(math.degrees(joint_error), 3),
        "p1_world": [round(float(value), 5) for value in p1],
        "waypoints": [
            {"name": name, "xyz": [round(x, 5), round(y, 5), round(z, 5)], "meaning": LABELS[name]}
            for name, x, y, z in waypoints
        ],
        "cube_xyz": list(CUBE_XYZ),
        "images": saved,
        "overview": str(overview),
    }
    (OUT / "preview_home_isaac.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("OVERVIEW", overview, flush=True)
    print("JOINT_ERROR_DEG", report["joint_error_deg"], flush=True)
    print("P1", report["p1_world"], flush=True)
    print("PREVIEW_OK", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        text = traceback.format_exc()
        (OUT / "error.txt").write_text(text)
        print(text, flush=True)
        print("PREVIEW_FAILED", flush=True)
        code = 1
    finally:
        simulation_app.close()
    sys.exit(code)
