Franka_Fr3_Dual_Arm_Config = {
    "images": {
        "top_head": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "rgb"],
        }
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
            "shape": (16,),
            "names": [
                "left_joint_0",
                "left_joint_1",
                "left_joint_2",
                "left_joint_3",
                "left_joint_4",
                "left_joint_5",
                "left_joint_6",
                "left_gripper",
                "right_joint_0",
                "right_joint_1",
                "right_joint_2",
                "right_joint_3",
                "right_joint_4",
                "right_joint_5",
                "right_joint_6",
                "right_gripper",
            ],
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
    }
}