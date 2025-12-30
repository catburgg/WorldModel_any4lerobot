Franka_1RGB_Config = {
    "images": {
        "top_head": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "rgb"],
        },
    },
    "states": {
        "eef.left.wrist": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "r", "p", "y"],
        },
        "eef.left.hand": {
            "dtype": "float32",
            "shape": (1,),
        },
        "eef.right.wrist": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "r", "p", "y"],
        },
        "eef.right.hand": {
            "dtype": "float32",
            "shape": (1,),
        },
        "proprioception_raw": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "camera.intrinsic": {
            "dtype": "float32",
            "shape": (9,),
        },
        "camera.extrinsic": {
            "dtype": "float32",
            "shape": (16,),
        },
        "task_index": {
            "dtype": "int64",
            "shape": (1,),
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
        }
    },
}
