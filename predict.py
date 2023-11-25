import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from Utils.inference import get_models, get_prediction
from Utils.visualization import get_overlay


def iou(prediction, mask):
    intersection = np.logical_and(prediction, mask)
    union = np.logical_or(prediction, mask)

    if np.sum(union) == 0:
        return 1.0

    return np.sum(intersection) / np.sum(union)


def f1_score(prediction, mask):
    if np.sum(prediction) == 0 and np.sum(mask) == 0:
        return 1.0
    if np.sum(prediction) == 0 or np.sum(mask) == 0:
        return 0.0

    precision = np.sum(prediction * mask) / np.sum(prediction)
    recall = np.sum(prediction * mask) / np.sum(mask)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

"""
Example usage:
python predict.py --path_to_pics data/train/images/ --path_to_predictions  data/train/predictions --path_to_models models/ --device cpu --path_to_masks data/train/masks/ --path_to_overlays data/train/overlays
"""

def main():
    """
    Perform inference on images using trained models.

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_pics', type=str, required=True)
    parser.add_argument('--path_to_predictions', type=str, required=True)
    parser.add_argument('--path_to_models', type=str, required=True)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--path_to_masks', type=str, required=False)
    parser.add_argument('--path_to_overlays', type=str, required=False)

    args = parser.parse_args()

    path_to_pics = args.path_to_pics
    path_to_models = args.path_to_models
    path_to_masks = args.path_to_masks
    path_to_predictions = args.path_to_predictions
    path_to_overlays = args.path_to_overlays
    device = args.device

    if not os.path.exists(path_to_pics):
        raise ValueError(f'Path to pics does not exist: {path_to_pics}')
    if not os.path.exists(path_to_models):
        raise ValueError(f'Path to models does not exist: {path_to_models}')
    if path_to_masks is not None:
        if not os.path.exists(path_to_masks):
            raise ValueError(f'Path to masks does not exist: {path_to_masks}')
    if not os.path.exists(path_to_predictions):
        os.makedirs(path_to_predictions)
    if path_to_overlays is not None:
        if path_to_masks is None:
            raise ValueError(f'Path to masks is required to generate overlays')
        if not os.path.exists(path_to_overlays):
            os.makedirs(path_to_overlays)

    for model, model_config in get_models(path_to_models, device=device):
        print("-" * 50)
        print(f"Starting inference for model {model_config.decoder_name}-{model_config.encoder_name}")

        pics_list = list(sorted(os.listdir(path_to_pics)))
        masks_list = None
        if path_to_masks is not None:
            masks_list = list(sorted(os.listdir(path_to_masks)))

        assert masks_list is None or len(pics_list) == len(masks_list)

        model_name = f"{model_config.decoder_name}-{model_config.encoder_name}"
        os.makedirs(os.path.join(path_to_predictions, model_name), exist_ok=True)
        if path_to_overlays is not None:
            os.makedirs(os.path.join(path_to_overlays, model_name), exist_ok=True)

        pbar = tqdm(enumerate(pics_list), total=len(pics_list))
        for ind, pic_path in pbar:
            pbar.set_description(f"Processing {pic_path}")

            pic = cv2.imread(os.path.join(path_to_pics, pic_path))

            mask = None
            if masks_list is not None:
                mask = cv2.imread(os.path.join(path_to_masks, masks_list[ind]))
                mask = mask[:, :, 0]

            def preprocess_image(image):
                return torch.tensor(model_config.preprocessing_fn(image)).permute(2, 0, 1).unsqueeze(0).float()

            prediction = get_prediction(model, pic, preprocess_image)

            Image.fromarray(prediction.astype(np.uint8), mode='L').save(os.path.join(path_to_predictions, model_name, pic_path))

            if path_to_overlays is not None:
                overlay = get_overlay(pic, red=prediction, green=mask)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path_to_overlays, model_name, pic_path), overlay)

            if mask is not None:
                iou_value = iou(prediction, mask)
                f1_value = f1_score(prediction, mask)
                print(f"IoU: {iou_value:.4f}, F1: {f1_value:.4f}")


if __name__ == '__main__':
    main()
