import torch
import cv2
import os

from sklearn.model_selection import train_test_split


def default_aug(image, mask):
    return image, mask


def default_preprocessing(image):
    return torch.from_numpy(image).permute(2, 0, 1).float()


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tiles,
                 masks,
                 augmentations,            # Аугментации
                 preprocessing   # Обработка данных для модели
                 ):
        self.has_mask = masks is not None
        self.tiles = tiles
        self.masks = masks
        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.tiles)

    def get_item(self, idx):
        image = self.tiles[idx]
        mask = None
        if self.has_mask:
            mask = self.masks[idx]
        image, mask = self.augmentations(image, mask)
        if self.has_mask:
            mask = torch.from_numpy(mask).float()
        before_preprocessing = image
        image = self.preprocessing(image)

        return image, mask, before_preprocessing

    def __getitem__(self, idx):
        image, mask, before_preprocessing = self.get_item(idx)
        if self.has_mask:
            return image, mask
        return image


def get_tiles(tile_size,
              path_to_pics,
              stride,
              path_to_masks=None,                   # None, если нет лейблов
              ):
    images = []
    masks = None

    for root, dirs, files in os.walk(path_to_pics):
        for file in sorted(files):
            img = cv2.imread(os.path.join(root, file))
            h, w, _ = img.shape

            for h_coord in range(0, (h - tile_size) // stride):
                for w_coord in range(0, (w - tile_size) // stride):
                    y, x = h_coord * stride, w_coord * stride
                    images.append(img[y: y + tile_size, x: x + tile_size])
    # Загрузка масок
    if path_to_masks is not None:
        masks = []
        for root, dirs, files in os.walk(path_to_masks):
            for file in sorted(files):
                mask = cv2.imread(os.path.join(root, file))
                mask = (mask[:, :, 0] > 0).astype('uint8')
                h, w = mask.shape

                for h_coord in range(0, (h - tile_size) // stride):
                    for w_coord in range(0, (w - tile_size) // stride):
                        y, x = h_coord * stride, w_coord * stride
                        masks.append(mask[y: y + tile_size, x: x + tile_size])

    return images, masks


def get_datasets(tile_size,
                 path_to_pics,
                 path_to_masks=None,
                 augmentations=default_aug,            # Аугментации
                 preprocessing=default_preprocessing   # Обработка данных для модели
                 ):
    tiles, masks = get_tiles(tile_size, path_to_pics, tile_size // 2, path_to_masks)
    train_tiles, val_tiles, train_masks, val_masks = train_test_split(tiles, masks, test_size=0.2, random_state=42)
    train_dataset = SegmentationDataset(train_tiles, train_masks, augmentations, preprocessing)
    val_dataset = SegmentationDataset(val_tiles, val_masks, default_aug, preprocessing)
    return train_dataset, val_dataset
