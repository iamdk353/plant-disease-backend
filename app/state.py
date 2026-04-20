import tensorflow as tf


class AppState:
    # Disease classifier
    model: tf.keras.Model = None
    idx_to_label: dict = None
    load_time: float = None

    # Leaf/plant gate classifier
    leaf_classifier: tf.keras.Model = None
    leaf_classifier_input_size: tuple[int, int] | None = None
    leaf_classifier_load_time: float | None = None

    # OCI
    oci_client = None
    oci_namespace: str = None


state = AppState()
