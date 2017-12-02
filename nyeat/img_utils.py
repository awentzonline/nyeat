import numpy as np
from PIL import Image


def arr_to_img(x, normalize=False):
    if normalize:
        axes = tuple(range(x.ndim - 1))
        x = 255. * (x - x.min(axis=axes)) / (x.max(axis=axes) - x.min(axis=axes))
    x = x.astype(np.uint8)
    mode = {
        2: 'L', 3: 'RGB', 4: 'RGBA'}[len(x.shape)]
    return Image.fromarray(x, mode=mode)
