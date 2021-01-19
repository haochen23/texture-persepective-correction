from data import get_homography_jit_loader
from config import homography_config
from utils.image_utils import get_train_val_paths


def test_dataset_homography_jit():
    data_dir = 'dataset/biglook/'
    out_t_lens = [5, 3]
    batch_size = 1

    train_paths, val_paths = get_train_val_paths(data_dir=data_dir,
                                                 split_ratio=homography_config['validation_split_ratio'])

    for out_t_len in out_t_lens:
        train_loader = get_homography_jit_loader(image_paths=train_paths,
                                                 batch_size=batch_size,
                                                 out_t_len=out_t_len)

        val_loader = get_homography_jit_loader(image_paths=val_paths,
                                               batch_size=batch_size,
                                               out_t_len=out_t_len)

        assert len(train_loader) == 9
        assert len(val_loader) == 1

        train_images, train_targets = next(iter(train_loader))
        val_images, val_targets = next(iter(val_loader))

        assert train_images.cpu().numpy().shape == (batch_size,
                                                    3,
                                                    homography_config['image_height'],
                                                    homography_config['image_width'])
        assert train_targets.cpu().numpy().shape == (batch_size, out_t_len)

        assert val_images.cpu().numpy().shape == (batch_size,
                                                    3,
                                                    homography_config['image_height'],
                                                    homography_config['image_width'])
        assert val_targets.cpu().numpy().shape == (batch_size, out_t_len)
