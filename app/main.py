import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from app.config import (
    CLASSIFIER_PATH,
    CLASSIFIER_REQUIRED,
    LABEL_MAP_PATH,
    MODEL_PATH,
)
from app.state import state
from app.routes import predict, upload, utility, user, activities, ai, weather


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not Path(LABEL_MAP_PATH).exists():
        raise RuntimeError(f"Label map not found: {LABEL_MAP_PATH}")

    # Leaf/plant gate model
    if not Path(CLASSIFIER_PATH).exists():
        msg = f"Leaf classifier not found: {CLASSIFIER_PATH}"
        if CLASSIFIER_REQUIRED:
            raise RuntimeError(msg)
        print(f"[WARN] {msg} (continuing without leaf gate)")

    @tf.keras.utils.register_keras_serializable()
    def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        loss   = alpha * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    # Load disease classifier
    state.model = load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss})
    state.model.trainable = False

    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)
    state.idx_to_label = {int(k): v for k, v in raw.items()}

    # Load leaf/plant gate classifier (if present)
    cls_t0 = time.time()
    if Path(CLASSIFIER_PATH).exists():
        state.leaf_classifier = load_model(CLASSIFIER_PATH)
        state.leaf_classifier.trainable = False

        # Derive expected input size (H, W)
        try:
            _h = int(state.leaf_classifier.input_shape[1])
            _w = int(state.leaf_classifier.input_shape[2])
            state.leaf_classifier_input_size = (_h, _w)
        except Exception:
            state.leaf_classifier_input_size = None

        state.leaf_classifier_load_time = time.time() - cls_t0
        sz = state.leaf_classifier_input_size
        print(f"[OK] Leaf classifier loaded in {state.leaf_classifier_load_time:.2f}s | input={sz}")

    state.load_time = time.time() - t0

    print(f"[OK] Disease model loaded in {state.load_time:.2f}s  |  {len(state.idx_to_label)} classes")
    yield

    # Cleanup
    if state.leaf_classifier is not None:
        del state.leaf_classifier
    del state.model


app = FastAPI(
    title="Crop Disease Classifier",
    description="EfficientNetB3 — 136 crop/disease classes. Upload to OCI, predict by object name.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
      allow_origins=["https://agrinex-sage.vercel.app","https://agrinexai.vercel.app","http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(user.router)
app.include_router(user.users_router)
app.include_router(utility.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(activities.router)
app.include_router(ai.router)
app.include_router(weather.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
