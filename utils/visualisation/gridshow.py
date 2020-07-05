import cv2
import numpy as np


def gridshow(name, imgs, scales, cmaps, width, border=10):
    """
    Display images in a grid.
    :param name: cv2 Window Name to update
    :param imgs: List of Images (np.ndarrays)
    :param scales: The min/max scale of images to properly scale the colormaps
    :param cmaps: List of cv2 Colormaps to apply
    :param width: Number of images in a row
    :param border: Border (pixels) between images.
    """
    imgrows = []
    imgcols = []

    maxh = 0
    for i, (img, cmap, scale) in enumerate(zip(imgs, cmaps, scales)):

        # Scale images into range 0-1
        if scale is not None:
            img = (np.clip(img, scale[0], scale[1]) - scale[0]) / (scale[1] - scale[0])
        elif img.dtype == np.float:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # Apply colormap (if applicable) and convert to uint8
        if cmap is not None:
            try:
                imgc = cv2.applyColorMap((img * 255).astype(np.uint8), cmap)
            except:
                imgc = (img * 255.0).astype(np.uint8)
        else:
            imgc = img

        if imgc.shape[0] == 3:
            imgc = imgc.transpose((1, 2, 0))
        elif imgc.shape[0] == 4:
            imgc = imgc[1:, :, :].transpose((1, 2, 0))

        # Arrange row of images.
        maxh = max(maxh, imgc.shape[0])
        imgcols.append(imgc)
        if i > 0 and i % width == (width - 1):
            imgrows.append(np.hstack(
                [np.pad(c, ((0, maxh - c.shape[0]), (border // 2, border // 2), (0, 0)), mode='constant') for c in
                 imgcols]))
            imgcols = []
            maxh = 0

    # Unfinished row
    if imgcols:
        imgrows.append(np.hstack(
            [np.pad(c, ((0, maxh - c.shape[0]), (border // 2, border // 2), (0, 0)), mode='constant') for c in
             imgcols]))

    maxw = max([c.shape[1] for c in imgrows])

    cv2.imshow(name, np.vstack(
        [np.pad(r, ((border // 2, border // 2), (0, maxw - r.shape[1]), (0, 0)), mode='constant') for r in imgrows]))
