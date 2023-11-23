import torch
import cv2
import os


def default_aug(image, mask):
    return image, mask


def default_preprocessing(image):
    return torch.from_numpy(image).permute(2, 0, 1).float()


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tile_size,
                 path_to_pics,
                 path_to_masks=None,                   # None, если нет лейблов
                 augmentations=default_aug,            # Аугментации
                 preprocessing=default_preprocessing   # Обработка данных для модели
                 ):
        # Загрузка картинок
        self.images = []
        self.prefix_sizes = [0]
        self.tile_size = tile_size

        for root, dirs, files in os.walk(path_to_pics):
            for file in sorted(files):
                self.images.append(cv2.imread(os.path.join(root, file)))
                h, w, _ = self.images[-1].shape
                tile_count = (h // tile_size) * (w // tile_size)
                self.prefix_sizes.append(self.prefix_sizes[-1] + tile_count)

        # Загрузка масок
        self.has_mask = False
        if path_to_masks is not None:
            self.has_mask = True
            self.masks = []
            for root, dirs, files in os.walk(path_to_masks):
                for file in sorted(files):
                    mask = cv2.imread(os.path.join(root, file))
                    mask = (mask[:, :, 0] > 0).astype('uint8')
                    self.masks.append(mask)

        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __len__(self):
        return self.prefix_sizes[-1]

    def crop_tile(self, arr, img_index, tile_index):
        h, w, _ = self.images[img_index].shape
        w_tile_count = w // self.tile_size
        tile_idx_h, tile_idx_w = tile_index // w_tile_count, tile_index % w_tile_count
        h_start, w_start = self.tile_size * tile_idx_h, self.tile_size * tile_idx_w
        return arr[img_index][h_start: h_start + self.tile_size, w_start: w_start + self.tile_size]

    def get_item(self, idx):
        img_idx = 0
        while self.prefix_sizes[img_idx + 1] < idx:
            img_idx += 1
        tile_idx = idx - self.prefix_sizes[img_idx]
        image = self.crop_tile(self.images, img_idx, tile_idx)
        mask = None
        if self.has_mask:
            mask = self.crop_tile(self.masks, img_idx, tile_idx)
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
