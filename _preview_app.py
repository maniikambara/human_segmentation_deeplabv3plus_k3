import os
import sys
import types

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))

tf_mod = types.ModuleType("tensorflow")
keras_mod = types.ModuleType("tensorflow.keras")
models_mod = types.ModuleType("tensorflow.keras.models")
models_mod.load_model = lambda *a, **k: object()
keras_mod.models = models_mod
tf_mod.keras = keras_mod
sys.modules.setdefault("tensorflow", tf_mod)
sys.modules.setdefault("tensorflow.keras", keras_mod)
sys.modules.setdefault("tensorflow.keras.models", models_mod)

import inference


def _fake_segment_image(model, image_bgr):
    h, w = image_bgr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * 0.38, w * 0.24
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    return mask.astype(np.uint8)


inference.segment_image = _fake_segment_image

exec(compile(open(os.path.join(APP_DIR, "app.py")).read(), "app.py", "exec"))
