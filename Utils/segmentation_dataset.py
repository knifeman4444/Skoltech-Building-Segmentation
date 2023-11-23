import torch
import cv2
import os


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_pics,
                 path_to_masks=None,   # None, если нет лейблов
                 augmentations=None,   # Аугментации
                 preprocessing=None,   # Обработка данных для модели
                 cnt_repeat_picture=1  # Количество повторений картинки (для train)
                 ):
        # Загрузка картинок
        self.images = []
        for root, dirs, files in os.walk(path_to_pics):
            for file in sorted(files):
                self.images.append(cv2.imread(os.path.join(root, file)))

        # Загрузка масок
        self.has_mask = False
        if path_to_masks is not None:
            self.has_mask = True
            self.masks = []
            for root, dirs, files in os.walk(path_to_masks):
                for file in sorted(files):
                    self.masks.append(cv2.imread(os.path.join(root, file)))

        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.cnt_repeat_picture = cnt_repeat_picture

    def __len__(self):
        return len(self.images) * self.cnt_repeat_picture

    def get_item(self, idx):
        im_idx = idx // self.cnt_repeat_picture
        image = self.images[im_idx]
        mask = None
        if self.has_mask:
            mask = self.masks[im_idx]
        if self.augmentations:
            image, mask = self.augmentations(image, mask)
        before_preprocessing = image
        if self.preprocessing:
            image = self.preprocessing(image)

        return image, mask, before_preprocessing

    def __getitem__(self, idx):
        image, mask, before_preprocessing = self.get_item(idx)
        if self.has_mask:
            return image, mask
        return image
