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

DEBUG = False
MULT = 255 if DEBUG else 1


def do_voting(pics_list, model_names, path_to_predictions, path_to_masks):
    print('Starting voting')
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        dir_path = os.path.join(path_to_predictions, f'voting_{threshold}')
        os.makedirs(dir_path, exist_ok=True)
        pbar = tqdm(enumerate(pics_list), total=len(pics_list))
        for ind, pic_path in pbar:
            sum = None
            pbar.set_description(f"Processing {pic_path}")

            real_mask = cv2.imread(os.path.join(path_to_masks, pic_path.replace('image', 'mask')), cv2.IMREAD_GRAYSCALE)

            for model_name in model_names:
                pic = cv2.imread(os.path.join(path_to_predictions, model_name, pic_path), cv2.IMREAD_GRAYSCALE)
                if sum is None:
                    sum = pic.astype(np.float32)
                else:
                    sum += pic
            sum = sum / len(model_names) / MULT

            sum[sum < threshold] = 0
            sum[sum >= threshold] = 1

            print(f'F1 score for voting with threshold {threshold} is {f1_score(sum, real_mask)}')

            Image.fromarray(sum.astype(np.uint8) * MULT, mode='L').save(os.path.join(dir_path, pic_path))


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
        if not os.path.exists(path_to_overlays):
            os.makedirs(path_to_overlays)

    model_names = []

    for model, model_config in get_models(path_to_models, device=device):
        print("-" * 50)
        print(f"Starting inference for model {model_config.decoder_name}-{model_config.encoder_name}")

        pics_list = list(sorted(filter(lambda _: _.endswith('.png'), os.listdir(path_to_pics))))
        masks_list = None
        if path_to_masks is not None:
            masks_list = list(sorted(filter(lambda _: _.endswith('.png'), os.listdir(path_to_masks))))

        assert masks_list is None or len(pics_list) == len(masks_list)

        model_name = model_config.filename
        model_names.append(model_name)
        os.makedirs(os.path.join(path_to_predictions, model_name), exist_ok=True)
        if path_to_overlays is not None:
            os.makedirs(os.path.join(path_to_overlays, model_name), exist_ok=True)
        # if DEBUG:
        #     continue

        if model_config.best_threshold is not None:
            thresholds = [model_config.best_threshold]
        else:
            thresholds = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
        resulting_f1_scores = []
        for threshold in thresholds:
            print(f'Using threshold {threshold}')
            f1_scores = []
            iou_scores = []
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

                prediction = get_prediction(model, pic, preprocess_image, threshold=threshold)

                Image.fromarray(prediction.astype(np.uint8) * MULT, mode='L').save(os.path.join(path_to_predictions, model_name, pic_path))

                if path_to_overlays is not None:
                    if mask is not None:
                        overlay = get_overlay(pic, red=prediction, green=mask)
                    else:
                        overlay = get_overlay(pic, red=prediction)
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(path_to_overlays, model_name, pic_path), overlay)

                if mask is not None:
                    iou_value = iou(prediction, mask)
                    f1_value = f1_score(prediction, mask)
                    f1_scores.append(f1_value)
                    iou_scores.append(iou_value)
            if mask is not None:
                f1_scores = np.array(f1_scores)
                score = np.mean(f1_scores)
                print(f'F1 score for {model_name} with threshold {threshold} is {score:.4f}')
                print(f'mIoU score for {model_name} with threshold {threshold} is {np.mean(np.array(iou_scores)):.4f}')
                resulting_f1_scores.append(score)
        if model_config.best_threshold is None:
            best_index = np.argmax(resulting_f1_scores)
            print('=' * 150)
            print('=' * 150)
            print(f'Attention! Consider using threshold {thresholds[best_index]} for model {model_name}')
            print('=' * 150)
            print('=' * 150)

    do_voting(pics_list, model_names, path_to_predictions, path_to_masks)

if __name__ == '__main__':
    main()
