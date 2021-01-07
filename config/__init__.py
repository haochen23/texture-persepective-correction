class Config:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


homography_config = {
  "load_width": 1024,
  "load_height": 1024,
  "image_width": 512,
  "image_height": 512,
  "translation_range": 0.05
}