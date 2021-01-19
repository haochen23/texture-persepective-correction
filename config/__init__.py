class Config:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)


homography_config = {
  "load_width": 1024,
  "load_height": 1024,
  "image_width": 512,
  "image_height": 512,
  "translation_range": 0.05,
  "validation_split_ratio": 0.1
}


flownet_config = {
  "load_width": 512,
  "load_height": 512,
  "image_width": 256,
  "image_height": 256,
  "validation_split_ratio": 0.1
}

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]