class Config:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


homography_config = {
  "load_width": 512,
  "load_height": 512,
  "image_width": 256,
  "image_height": 256
}