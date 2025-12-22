class Hot3DLeRobotFeatures:
    
    def __init__(self, img_width=512, img_height=512):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "eef.left.wrist":{
                "dtype": "float64",
                "shape": (6,),
            },
            "eef.right.wrist":{
                "dtype": "float64",
                "shape": (6,),
            },
            "eef.left.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "eef.right.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "camera.intrinsic":{
                "dtype": "float64",
                "shape": (3,3),
            },
            "camera.extrinsic":{
                "dtype": "float64",
                "shape": (4,4),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "names": ["channels", "height", "width"],
                "shape": self.img_shape,
            },
        }

class HoloassistLeRobotFeatures:
    
    def __init__(self, img_width=512, img_height=512):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "eef.left.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "eef.right.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "camera.intrinsic":{
                "dtype": "float64",
                "shape": (3,3),
            },
            "camera.extrinsic":{
                "dtype": "float64",
                "shape": (4,4),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "names": ["channels", "height", "width"],
                "shape": self.img_shape,
            },
        }

class HOI4DLeRobotFeatures:
    
    def __init__(self, img_width=960, img_height=540):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "eef.left.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "eef.right.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "camera.intrinsic":{
                "dtype": "float64",
                "shape": (3,3),
            },
            "camera.extrinsic":{
                "dtype": "float64",
                "shape": (4,4),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "names": ["channels", "height", "width"],
                "shape": self.img_shape,
            },
        }

class HOI4DLeRobotFeatures:
    
    def __init__(self, img_width=960, img_height=540):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "eef.left.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "eef.right.hand":{
                "dtype": "float64",
                "shape": (21,3),
            },
            "camera.intrinsic":{
                "dtype": "float64",
                "shape": (3,3),
            },
            "camera.extrinsic":{
                "dtype": "float64",
                "shape": (4,4),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "names": ["channels", "height", "width"],
                "shape": self.img_shape,
            },
        }