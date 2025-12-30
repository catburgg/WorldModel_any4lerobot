Tien_Kung_Prod1_Gello_1RGB_Config = {
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
                "left_arm_0",
                "left_arm_1",
                "left_arm_2",
                "left_arm_3",
                "left_arm_4",
                "left_arm_5",
                "left_arm_6",
                "left_hand_closure",
                "right_arm_0",
                "right_arm_1",
                "right_arm_2",
                "right_arm_3",
                "right_arm_4",
                "right_arm_5",
                "right_arm_6",
                "right_hand_closure",
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
