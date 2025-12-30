Sim_Tienkung_1RGB_Config = Tien_Kung_Gello_1RGB_Config = {
    "images": {
        "top_head": {
            "dtype": "video",
            "shape": (480, 640, 3),
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
            "shape": (38,),
            "names": [
                "left_arm_0",
                "left_arm_1",
                "left_arm_2",
                "left_arm_3",
                "left_arm_4",
                "left_arm_5",
                "left_arm_6",
                "right_arm_0",
                "right_arm_1",
                "right_arm_2",
                "right_arm_3",
                "right_arm_4",
                "right_arm_5",
                "right_arm_6",
                "left_thumb0_for_bending",
                "left_thumb1_for_rotation",
                "left_passive_thumb0",
                "left_passive_thumb1",
                "left_index_finger",
                "left_passive_index_finger",
                "left_middle_finger",
                "left_passive_middle_finger",
                "left_ring_finger",
                "left_passive_ring_finger",
                "left_little_finger",
                "left_passive_little_finger",
                "right_thumb0_for_bending",
                "right_thumb1_for_rotation",
                "right_passive_thumb0",
                "right_passive_thumb1",
                "right_index_finger",
                "right_passive_index_finger",
                "right_middle_finger",
                "right_passive_middle_finger",
                "right_ring_finger",
                "right_passive_ring_finger",
                "right_little_finger",
                "right_passive_little_finger",
            ]
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
