from data import get_loader
from config import flownet_config as config

def test_flownet_dataset():
    data_loader = get_loader('dataset/processed/train/distorted/',
                             'dataset/processed/train/flow/',
                             batch_size=5,
                             data_type="flow"
                             )
    print(len(data_loader))
    images, flow_xs, flow_ys = next(iter(data_loader))
    width = config['image_width']
    height = config['image_height']
    assert images.shape == (5, 3, height, width)
    assert flow_xs.shape == (5, 1, height, width)
    assert flow_ys.shape == (5, 1, height, width)