import io
import os
import sys
import types
import uuid

import pytest
from PIL import Image


os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _install_fake_tensorflow() -> None:
    if "tensorflow" in sys.modules:
        return

    tf = types.ModuleType("tensorflow")

    class _Config:
        @staticmethod
        def list_physical_devices(device_type=None):
            return []

    class _Math:
        @staticmethod
        def log(value):
            return value

        @staticmethod
        def reduce_sum(value, axis=None, keepdims=False):
            return value

    tf.config = _Config()
    tf.math = _Math()
    tf.clip_by_value = lambda value, clip_value_min, clip_value_max: value
    tf.reduce_mean = lambda value: value
    tf.pow = lambda x, y: x

    keras = types.ModuleType("tensorflow.keras")
    keras.Model = object

    utils = types.ModuleType("tensorflow.keras.utils")

    def register_keras_serializable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    utils.register_keras_serializable = register_keras_serializable

    models = types.ModuleType("tensorflow.keras.models")
    models.load_model = lambda *args, **kwargs: None

    applications = types.ModuleType("tensorflow.keras.applications")
    efficientnet = types.ModuleType("tensorflow.keras.applications.efficientnet")
    efficientnet.preprocess_input = lambda arr: arr
    applications.efficientnet = efficientnet

    keras.utils = utils
    keras.models = models
    keras.applications = applications
    tf.keras = keras

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.utils"] = utils
    sys.modules["tensorflow.keras.models"] = models
    sys.modules["tensorflow.keras.applications"] = applications
    sys.modules["tensorflow.keras.applications.efficientnet"] = efficientnet


def _install_fake_oci() -> None:
    if "oci" in sys.modules:
        return

    oci = types.ModuleType("oci")
    config = types.ModuleType("oci.config")
    object_storage = types.ModuleType("oci.object_storage")

    class _Response:
        def __init__(self, data):
            self.data = data

    class ObjectStorageClient:
        def __init__(self, config_value):
            self.config_value = config_value

        def get_namespace(self):
            return _Response(types.SimpleNamespace(data="namespace"))

        def get_object(self, namespace, bucket_name, object_name):
            raise NotImplementedError

        def put_object(self, namespace, bucket_name, object_name, data, content_type=None):
            raise NotImplementedError

        def list_objects(self, namespace_name, bucket_name, prefix=None):
            return _Response(types.SimpleNamespace(objects=[]))

    config.from_file = lambda: {}
    object_storage.ObjectStorageClient = ObjectStorageClient
    oci.config = config
    oci.object_storage = object_storage

    sys.modules["oci"] = oci
    sys.modules["oci.config"] = config
    sys.modules["oci.object_storage"] = object_storage


_install_fake_tensorflow()
_install_fake_oci()


from app.state import state


@pytest.fixture(autouse=True)
def reset_state():
    yield
    state.model = None
    state.idx_to_label = None
    state.load_time = None
    state.leaf_classifier = None
    state.leaf_classifier_input_size = None
    state.leaf_classifier_load_time = None
    state.oci_client = None
    state.oci_namespace = None


@pytest.fixture
def sample_image_bytes():
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def uuid_pair():
    return uuid.uuid4(), uuid.uuid4()
