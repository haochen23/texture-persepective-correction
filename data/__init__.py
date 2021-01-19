from .dataset_flownet import  FlowNetDataset
from .dataset_homography import HomographyDataset
from .dataset_homography_jit import HomographyDatasetJIT
from torchvision import transforms
from torch.utils import data


def get_loader(imageDir, targetDir, batch_size, data_type='homo'):
    """get dataloader according to the type of dataset
    Inputs:
        imageDir:   path to the training images
        targetDir:   path to the training target
        batch_size:  batch size
        dataset: either 'homo' for HomographyDataset or 'flow' for FlowNetDataset
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if data_type.lower() == 'flow':
        dataset = FlowNetDataset(imageDir, targetDir, transform)
    elif data_type.lower() == 'homo':
        dataset = HomographyDataset(imageDir, targetDir, transform)
    else:
        raise NotImplementedError("Unknown dataset type %s" % data_type)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True)

    return data_loader


def get_homography_jit_loader(image_paths, batch_size, out_t_len=3):
    """
    this function returns a data loader for HomographyDatasetJIT
    Args:
        image_paths:    paths to all image files, a list
        batch_size:     batch size

    Returns:
        a data loader

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = HomographyDatasetJIT(image_paths=image_paths,
                                   transform=transform,
                                   out_t_len=out_t_len)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)

    return data_loader

