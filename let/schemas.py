class LetFeatures:
    
    def __init__(self, img_width, img_height):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "eef.left.wrist":{
                "dtype": "float32",
                "shape": (6,),
            },
            "eef.right.wrist":{
                "dtype": "float32",
                "shape": (6,),
            },
            "eef.left.hand":{
                "dtype": "float32",
                "shape": (1,),
            },
            "eef.right.hand":{
                "dtype": "float32",
                "shape": (1,),
            }
        }