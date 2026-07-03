"""Standalone inference helpers for app.py (mirrors segment_image in predict.ipynb).

Kept as plain Python (not notebook %run) so the web app can import it directly
without pulling in Jupyter/matplotlib/tqdm overhead.
"""
import cv2
import numpy as np

H = 512
W = 512


def standardize_image(img, size=(W, H)):
    """Resize a BGR uint8 image to `size` with INTER_AREA. No-op if already that size."""
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def segment_image(model, image_bgr):
    """
    Run the model on a BGR image of any size. Returns a binary (0/1) mask
    at the image's original resolution.
    """
    h, w = image_bgr.shape[:2]
    img = standardize_image(image_bgr)
    x = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    y = model.predict(x, verbose=0)[0]
    y = cv2.resize(y, (w, h))
    return (y > 0.5).astype(np.uint8)
