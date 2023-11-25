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

            if len(image.shape) == 3:
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


def get_overlay(image, red, green=None, alpha=0.5):
    """
    :param image: The input image. Can be a torch.Tensor or a numpy.ndarray.
    :param red: The red mask to overlay on the image. Can be a torch.Tensor or a numpy.ndarray.
    :param green: The green mask to overlay on the image (optional). Can be a torch.Tensor or a numpy.ndarray.
    :param alpha: The transparency of the overlay (default is 0.5).
    :return: The overlay image as a numpy.ndarray.

    This method takes an image and overlays a red mask on it. If a green mask is provided, it overlays both the red and green masks on the image.

    The input image and masks can be either torch.Tensor objects or numpy.ndarray objects. If they are torch.Tensor objects, they are detached from the computation graph, moved to the CPU, and converted to numpy.ndarray objects. If the image or mask has a single channel, the method expands it to 3 channels.

    The red and green masks are preprocessed by expanding their dimensions and converting them to 3 channels. The red mask is then multiplied by [255, 0, 0] to convert it to a red color. If a green mask is provided, it is multiplied by [0, 255, 0] to convert it to a green color. The red and green masks are then added together with the image using the alpha value (transparency) to create the overlay. The overlay image is returned as a numpy.ndarray.

    Example usage:
        image = torch.randn(3, 256, 256)  # Example input image
        red_mask = torch.zeros(1, 256, 256)  # Example red mask
        green_mask = torch.ones(1, 256, 256)  # Example green mask
        overlay = get_overlay(image, red_mask, green_mask, alpha=0.5)  # Get the overlay image
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 1:
            image = image[0]
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)

    def preprocess_mask(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            if mask.shape[0] == 1:
                mask = mask[0]
            if len(image.shape) == 3:
                mask = mask.transpose(1, 2, 0)

        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]

        mask = np.expand_dims(mask, axis=-1)

        return mask

    red = preprocess_mask(red)
    if green is not None:
        green = preprocess_mask(green)
    
    red_mask = np.concatenate([red, np.zeros_like(red), np.zeros_like(red)], axis=-1) * 255

    if green is not None:
        green_mask = np.concatenate([np.zeros_like(green), green, np.zeros_like(green)], axis=-1) * 255
        overlay = (image * (1 - alpha) + alpha * (red_mask + green_mask)).astype(np.uint8)
    else:
        overlay = (image * (1 - alpha) + alpha * red_mask).astype(np.uint8)

    return overlay.astype(np.uint8)