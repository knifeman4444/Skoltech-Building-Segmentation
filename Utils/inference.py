import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from dataclasses import dataclass
import os
from tqdm import tqdm
from Utils.visualization import get_overlay


def get_prediction(model, image: np.ndarray, preprocessing, tile_size=256, padding=64, threshold=0.5):
    """
    :param model: The PyTorch model used for prediction
    :param image: The input image as a NumPy array
    :param preprocessing: A function used to preprocess the image tiles before prediction
    :param tile_size: The size of each tile used for prediction (default: 256)
    :param padding: The amount of padding added to each tile (default: 64)
    :param threshold: The threshold used to binarize the prediction (default: 0.5)
    :return: The predicted segmentation mask as a NumPy array

    This method takes in a deep learning model, an input image, a preprocessing function, and various parameters for tile size, padding, and threshold. It performs tiled inference on the input image using the provided model, using the specified tile size and padding. The resulting prediction is binarized using the threshold value, and the final segmentation mask is returned as a NumPy array.
    """
    if len(image.shape) != 3:
        raise ValueError(f'Image must have 3 dimensions. Got {len(image.shape)}')
    
    window_size = tile_size + padding * 2
    w, h = image.shape[:2]
    image = np.pad(image, ((window_size, window_size), (window_size, window_size), (0, 0)))
    result = np.zeros((image.shape[0], image.shape[1]))
    
    for i in tqdm(range(window_size, w + window_size, tile_size)):
        for j in range(window_size, h + window_size, tile_size):
            tile = image[i-padding:i+tile_size+padding, j-padding:j+tile_size+padding]

            assert tile.shape == (window_size, window_size, 3)

            tile = preprocessing(tile)

            prediction = model(tile).detach().cpu().squeeze().numpy()

            prediction = prediction[padding:-padding, padding:-padding]
            prediction = np.where(prediction > threshold, 1, 0)
            result[i:i+tile_size, j:j+tile_size] = prediction
            
    return result[window_size:w+window_size, window_size:h+window_size]


class ModelConfig:
    """

    ModelConfig class represents the configuration for a model used in image segmentation tasks. It includes information such as the decoder name, encoder name, encoder weights, and preprocessing function.

    Attributes:
        - decoder_name (str): The name of the decoder architecture.
        - encoder_name (str): The name of the encoder architecture.
        - encoder_weights (str, optional): The weights for the encoder architecture. Defaults to "imagenet".
        - preprocessing_fn (callable, optional): The preprocessing function to be applied to input images. Defaults to None.
        - decoder_class (class): The class of the decoder architecture.
        - encoder_class (class): The class of the encoder architecture.

    Methods:
        - __init__(decoder_name, encoder_name, decoder_class, encoder_weights="imagenet"): Constructs a ModelConfig instance.
        - load_model(path_to_model, device="cpu"): Loads a pre-trained model based on the configuration and returns it.

    Example usage:

        # Create a ModelConfig instance
        config = ModelConfig(decoder_name="unet", encoder_name="resnet50", decoder_class=smp.Unet)

        # Load a pre-trained model
        model = config.load_model("path/to/model.pth", device="cuda")

    Note: This documentation only includes the description of the class and its attributes/methods. The example code and author/version tags are excluded as per the given instructions.

    """
    decoder_name: str
    encoder_name: str
    encoder_weights: str = "imagenet"
    preprocessing_fn: callable = None

    decoder_class = None
    encoder_class = None

    def __init__(self, decoder_name, encoder_name, decoder_class, encoder_weights="imagenet"):
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        self.decoder_class = decoder_class

        if encoder_weights:
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
        else:
            self.preprocessing_fn = lambda x: x

    def load_model(self, path_to_model, device="cpu"):
        model = self.decoder_class(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            classes=1,
            activation='sigmoid',
        )
        model.load_state_dict(torch.load(path_to_model, map_location=device))
        model.to(device)
        model.eval()
        return model


MODELS = [
    ModelConfig("linknet", "efficientnet-b5", smp.Linknet),
    ModelConfig("unet", "efficientnet-b5", smp.Unet),
    ModelConfig("unet++", "efficientnet-b5", smp.UnetPlusPlus),
    ModelConfig("PAN", "timm-efficientnet-b5", smp.PAN),
]


def get_models(path_to_models, device="cuda"):
    """
    Get models from a directory.

    :param path_to_models: Path to the directory containing model files.
    :param device: Device to load the models to. Defaults to "cuda".
    :return: A generator that yields tuples of model and model configuration.
    """
    for file in os.listdir(path_to_models):
        if not file.endswith(".pth"):
            continue

        for model_config in MODELS:
            if model_config.encoder_name not in file or model_config.decoder_name not in file:
                continue

            print(f"Loading model {file}")
            model = model_config.load_model(os.path.join(path_to_models, file), device=device)
            yield model, model_config


if __name__ == '__main__':

    sample_image = cv2.imread("../data/train/images/train_image_000.png")
    sample_mask = cv2.imread("../data/train/masks/train_mask_000.png")
    sample_mask = sample_mask[:, :, 0]

    for model, model_config in get_models("../models", "cpu"):

        def preprocess(image):
            """
            Preprocess the given image before feeding it into the model.

            :param image: NumPy array representing the input image.
            :return: Preprocessed image as a torch.Tensor.

            """
            return torch.tensor(model_config.preprocessing_fn(image)).permute(2, 0, 1).unsqueeze(0).float()

        prediction = get_prediction(model, sample_image, preprocess)
        overlay = get_overlay(sample_image, red=prediction, green=sample_mask)
        cv2.imwrite(f"../overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

