import io

import numpy as np
from fastapi import HTTPException
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from app.config import CLASSIFIER_THRESHOLD, IMG_SIZE, TOP_K
from app.schemas import Prediction
from app.state import state


def preprocess_image(
    data: bytes,
    img_size: tuple[int, int] | None = None,
    *,
    efficientnet: bool = True,
) -> np.ndarray:
    """bytes → (1, H, W, 3) array.

    - efficientnet=True: applies EfficientNet `preprocess_input` (current disease model expects this)
    - efficientnet=False: raw float32 RGB; useful when the model has its own Rescaling layer (leaf classifier)
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, detail=f"Cannot decode image: {exc}")

    target = img_size or IMG_SIZE
    img = img.resize(target, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    if efficientnet:
        arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def leaf_gate(data: bytes) -> tuple[bool, float] | None:
    """Return (is_leaf, score) if leaf classifier is loaded, otherwise None.

    The shipped classifier is a sigmoid binary model with output shape (1, 1).
    We interpret score >= CLASSIFIER_THRESHOLD as "leaf/plant" by default.
    """
    if state.leaf_classifier is None:
        return None

    size = state.leaf_classifier_input_size or (224, 224)
    x = preprocess_image(data, img_size=size, efficientnet=False)

    y = state.leaf_classifier.predict(x, verbose=0)
    score = float(y[0][0])

    is_leaf = score >= CLASSIFIER_THRESHOLD
    return is_leaf, score


def run_inference(x: np.ndarray) -> list[list[Prediction]]:
    """(N, 300, 300, 3) → list of top-K Prediction lists."""
    all_probs = state.model.predict(x, verbose=0)
    k = min(TOP_K, all_probs.shape[1])
    results = []
    for probs in all_probs:
        top_idx = np.argsort(probs)[::-1][:k]
        results.append([
            Prediction(
                label=state.idx_to_label.get(int(i), f"class_{i}"),
                confidence=round(float(probs[i]), 6),
            )
            for i in top_idx
        ])
    return results
