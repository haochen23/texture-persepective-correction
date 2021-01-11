from config import Config

model_config = Config(
    cuda=True,
    device="cuda",
    seed=2,
    lr=0.01,
    epochs=4,
    save_epoch=False,
    batch_size=16,
    log_interval=100,
    layers=[1, 1, 1, 1, 2],
    data_dir='dataset/homography/',
    save_path='output/'
)


def test_config():
        config = model_config
        assert config.cuda == True
        assert config.device == 'cuda'
        assert config.seed == 2
        assert config.lr == 0.01
        assert config.epochs == 4
        assert config.save_epoch == False
        assert config.batch_size == 16
        assert config.log_interval == 100
        assert config.layers == [1, 1, 1, 1, 2]
        assert config.data_dir == 'dataset/homography/'
        assert config.save_path == 'output/'
