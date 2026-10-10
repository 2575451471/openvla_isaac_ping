# write by shenle
上游注册 `ur10e_push_sim`（本机一般已写好，缺了再贴）

## `prismatic/vla/datasets/rlds/oxe/configs.py`
```python
    "ur10e_push_sim": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
```

## `prismatic/vla/datasets/rlds/oxe/transforms.py`
```python
def ur10e_push_sim_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory
```

registry：
```python
    "ur10e_push_sim": ur10e_push_sim_dataset_transform,
```
