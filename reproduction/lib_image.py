import tensorflow as tf
import numpy as np


def normalize_image(image, mean: np.ndarray, stddev: np.ndarray):
    if len(image.shape) == 4:
        per_channel = []
        for i, (m, s) in enumerate(zip(mean, stddev)):
            per_channel.append((image[:, :, :, i] - m) / s)
        return tf.stack(per_channel, axis=-1)

    assert len(image.shape) == 3
    per_channel = []
    for i, (m, s) in enumerate(zip(mean, stddev)):
        per_channel.append((image[:, :, i] - m) / s)
    return tf.stack(per_channel, axis=-1)
