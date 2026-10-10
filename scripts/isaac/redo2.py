import omni.usd

EE_PRIM = "/World/ur10e/wrist_3_link"

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(EE_PRIM)

T = omni.usd.get_world_transform_matrix(prim)

q = T.ExtractRotationQuat()

w = float(q.GetReal())
v = q.GetImaginary()

x = float(v[0])
y = float(v[1])
z = float(v[2])

print("")
print("====================================")
print("DESIRED WRIST ORIENTATION")
print("====================================")

print(
    "TARGET_WORLD_QUAT_WXYZ = "
    "({:.12f}, {:.12f}, {:.12f}, {:.12f})".format(
        w, x, y, z
    )
)

print("====================================")