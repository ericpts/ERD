import os
from typing import Dict
import tensorflow as tf
import tensorflow_datasets as tfds
import gin
from .lib_image import normalize_image


IMAGE_SIZE = (224, 224)


def filter_for_id(example: Dict) -> bool:
    male = example["attributes"]["Male"]
    blond = example["attributes"]["Blond_Hair"]
    female = tf.logical_not(male)
    brunette = tf.logical_not(blond)

    return tf.logical_or(tf.logical_and(blond, female), tf.logical_and(brunette, male))


def filter_for_ood(example: Dict) -> bool:
    male = example["attributes"]["Male"]
    blond = example["attributes"]["Blond_Hair"]
    female = tf.logical_not(male)
    brunette = tf.logical_not(blond)

    return tf.logical_or(tf.logical_and(blond, male), tf.logical_and(brunette, female))


@gin.configurable
def extract_image_and_label(example, label: str = "gender"):
    assert label in ["hair", "gender"]

    X = example["image"]

    hair = tf.cast(example["attributes"]["Blond_Hair"], tf.int32)
    gender = tf.cast(example["attributes"]["Male"], tf.int32)

    if label == "hair":
        y = hair
    else:
        y = gender

    X = tf.image.convert_image_dtype(X, dtype=tf.float32, saturate=False)
    return (X, y)


def transform_image(X, y):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    X = tf.keras.layers.experimental.preprocessing.CenterCrop(
        orig_min_dim, orig_min_dim
    )(X)
    X = tf.keras.layers.experimental.preprocessing.Resizing(*IMAGE_SIZE)(X)
    X = normalize_image(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return X, y


def dataset_extract_images(D: tf.data.Dataset) -> tf.data.Dataset:
    D = D.map(extract_image_and_label, num_parallel_calls=4)
    D = D.batch(1).map(transform_image, num_parallel_calls=4)
    D = D.unbatch()
    return D


def impl_load(split: str):
    return tfds.load(
        "celeb_a",
        split=split,
        data_dir=os.path.join(os.getenv("SCRATCH", "~/"), ".datasets"),
    )


def get_celeb_a_id(split: str) -> tf.data.Dataset:
    return dataset_extract_images(impl_load(split).filter(filter_for_id))


def get_celeb_a_ood(split: str) -> tf.data.Dataset:
    return dataset_extract_images(impl_load(split).filter(filter_for_ood))


def load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    assert is_celeb_a(dataset)
    if dataset == "celeb_a_id":
        return get_celeb_a_id(split)
    else:
        return get_celeb_a_ood(split)


def is_celeb_a(dataset_name: str):
    return dataset_name in ["celeb_a_id", "celeb_a_ood"]


def get_image_size(dataset_name: str):
    return (*IMAGE_SIZE, 3)
