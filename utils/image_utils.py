import cv2
from PIL import Image
import numpy as np
import os
from config import IMG_EXTENSIONS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 20)

def pil_from_cv2(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def cv2_from_pil(image):
    image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(data_dir, pattern='Albedo'):
    """
    get the paths all image files contains a certain pattern
    Args:
        data_dir:   data_dir contains images in it or in sub dirs of it
        pattern:    the pattern used to filter out images

    Returns:
        image_paths: a list contains paths of images

    """

    assert os.path.isdir(data_dir), '%s is not a valid derectory' % data_dir

    image_paths = []

    for root, dnames, fnames in sorted(os.walk(data_dir, followlinks=True)):
        for fname in fnames:
            if pattern in fname and is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)

    return image_paths


def get_train_val_paths(data_dir, split_ratio=0.2):
    all_paths = get_image_paths(data_dir, pattern='Albedo')
    data_size = len(all_paths)
    indices = np.random.permutation(data_size)
    val_split = int(data_size * split_ratio)

    val_paths = [all_paths[i] for i in indices[:val_split]]
    train_paths = [all_paths[i] for i in indices[val_split:]]

    return train_paths, val_paths


def tile_images(imgs, picturesPerRow=4):
    """
    Make image tiles
    Args:
        imgs:  input images
        picturesPerRow:  how many tiles per row

    Returns:
        tiled images

    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    """
    Converting a torch tensor to numpy arrays
    Args:
        image_tensor: torch tensor
        imtype:       image type
        normalize:    has the torch tensor been normalized
        tile:         make the images into tiles

    Returns:

    """
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, normalize=normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    plt.savefig('weights_gradient_flow.png')


if __name__ == '__main__':
    data_dir = "dataset/biglook/"
    train_paths, test_paths = get_train_val_paths(data_dir, split_ratio=0.1)
    print(train_paths)
    print(len(train_paths))
    print(test_paths)
    print(len(test_paths))
