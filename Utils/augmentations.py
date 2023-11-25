import cv2
import random
from skimage.util import random_noise
import numpy as np


def augmentations(image, mask):
    rows, cols = mask.shape

    # Flip augmentations
    if random.randint(1, 2) == 1:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.randint(1, 2) == 1:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # Random noise augmentations
    image = random_noise(image, mode='s&p', amount=random.randint(5, 25) / 1000)
    image = np.array(255 * image, dtype=np.uint8)

    # Color augmentations
    col_aug = random.randint(1, 3)
    if col_aug == 1:
        ch_col = random.randint(80, 120) / 100
        image = image.astype('float64') * ch_col
    elif col_aug == 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        h += np.random.randint(0, 100, size=(rows, cols), dtype=np.uint8)
        s += np.random.randint(0, 20, size=(rows, cols), dtype=np.uint8)
        v += np.random.randint(0, 10, size=(rows, cols), dtype=np.uint8)
        image = cv2.merge([h, s, v])
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    else:
        h, s, v = cv2.split(image.astype(int))
        h += np.random.randint(-40, 40, size=(rows, cols))
        s += np.random.randint(-40, 40, size=(rows, cols))
        v += np.random.randint(-40, 40, size=(rows, cols))
        image = cv2.merge([h, s, v])
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype('uint8')

    # Stretching augmentations
    tx = random.randint(-cols // 4, cols // 4)
    ty = random.randint(-rows // 4, rows // 4)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (cols, rows))
    mask = cv2.warpAffine(mask, M, (cols, rows))
    x, y = max(tx, 0), max(ty, 0)
    w, h = cols - abs(tx), rows - abs(ty)
    image = image[y:y+h, x:x+w]
    image = cv2.resize(image, (cols, rows))
    mask = mask[y:y+h, x:x+w]
    mask = cv2.resize(mask, (cols, rows))
    image = image.astype('uint8')

    # Rotation augmentations
    Cx, Cy = rows, cols
    rand_angle = random.randint(-45,45)
    M = cv2.getRotationMatrix2D((Cy // 2, Cx // 2), rand_angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    mask = cv2.warpAffine(mask, M, (cols, rows))
    image = image.astype('uint8')

    # Blur
    if random.randint(1, 4) == 1:
        blur_val = random.randint(5, 10)
        image = cv2.blur(image, (blur_val, blur_val))
        mask = cv2.blur(mask, (blur_val, blur_val))
        mask = (mask > 0.5).astype('uint8')

    return image, mask