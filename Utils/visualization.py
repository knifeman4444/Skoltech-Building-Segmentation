import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize(figsize=(20, 10), **images):
    """
    :param figsize: A tuple (width, height) specifying the size of the figure. Default is (20, 10).
    :param images: Keyword arguments where the key is the name of the image and the value is the corresponding image data.
    :return: None

    This method takes in multiple images and displays them in a grid layout using matplotlib. It supports numpy arrays and torch tensors as image data. The figsize parameter allows you to specify the size of the figure. Each image will be displayed with its corresponding name as the title.

    Examples:
        visualize(figsize=(10, 5), image1=image_data1, image2=image_data2)
    """
    n_images = len(images)
    plt.figure(figsize=figsize)
    for idx, (name, image) in enumerate(images.items()):

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.shape[0] == 1:
                image = image[0]

            image = image.transpose(1, 2, 0)

        if not isinstance(image, np.ndarray):
            raise ValueError(f'Image must be numpy array or torch tensor. Got {type(image)}')
        if image.shape[-1] == 1:
            image = image[:, :, 0]

        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def get_overlay(image, mask, alpha=0.5):
    """
    :param image: The input image. Can be a torch.Tensor or a numpy array.
    :param mask: The mask to overlay on the image. Can be a torch.Tensor or a numpy array.
    :param alpha: The opacity of the mask overlay. Default is 0.5.

    :return: The overlay image with the mask overlayed on top of the input image.

    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 1:
            image = image[0]
        image = image.transpose(1, 2, 0)

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        if mask.shape[0] == 1:
            mask = mask[0]
        mask = mask.transpose(1, 2, 0)

    if mask.shape[-1] == 1:
        mask = mask[:, :, 0]

    mask = np.expand_dims(mask, axis=-1)
    red_mask = np.concatenate([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1) * 255
    overlay = image * (1 - alpha) + red_mask * alpha

    return overlay.astype(np.uint8)