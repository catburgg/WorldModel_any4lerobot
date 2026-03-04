class TasteRobFeatures:
    
    def __init__(self, img_width=960, img_height=540):
        self.img_shape = (3,img_width, img_height)

    def get_features(self):
        return {
            "observation.images.top_head": {
                "dtype": "video",
                "names": ["channels", "height", "width"],
                "shape": self.img_shape,
            },
        }