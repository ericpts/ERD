import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from .lib_image import normalize_image

WATERBIRDS_TARBALL_LINK = (
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
)

IMAGE_SIZE = (224, 224)


def transform_image(image: np.ndarray) -> tf.Tensor:
    image = tf.cast(image / 255, "float32")
    initial_resolution = (256, 256)
    image = tf.image.resize(image, initial_resolution)
    image = tf.image.central_crop(image, IMAGE_SIZE[0] / initial_resolution[0])
    image = normalize_image(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert image.shape == (224, 224, 3)
    return image


def read_image(filepath: Path):
    return np.asarray(Image.open(str(filepath)).convert("RGB"))


def download_waterbirds(cache_dir: Path):
    print("Downloading tarball")
    tf.keras.utils.get_file(
        "waterbirds", WATERBIRDS_TARBALL_LINK, extract=True, cache_dir=str(cache_dir)
    )


# y: | 0 -> land bird
#    | 1 -> water bird
#
# y_biased: | 0 -> land
#           | 1 -> water
def generate_waterbirds(split: str):
    dataset_dir = (
        Path(os.environ["SCRATCH"]) / "datasets" / "waterbird_complete95_forest2water2"
    )
    if not dataset_dir.exists():
        download_waterbirds(Path(os.environ["SCRATCH"]))

    dfm = pd.read_csv(dataset_dir / "metadata.csv")
    split_dict = {"train": 0, "val": 1, "test": 2}
    dfm = dfm[dfm["split"] == split_dict[split]].reset_index(drop=True)
    X = [
        transform_image(read_image(dataset_dir / f)) for f in dfm["img_filename"].values
    ]
    X = tf.stack(X, 0)
    y = dfm["y"].values
    y_biased = dfm["place"].values

    return X, y, y_biased


def impl_load_waterbirds(split: str, include_biased: bool = False):
    cache_filename = (
        Path(os.environ["SCRATCH"]) / ".datasets" / f"waterbirds_processed_{split}.npz"
    )

    if not cache_filename.exists():
        print(f"Generating waterbirds and saving to {cache_filename}")
        X, y, y_biased = generate_waterbirds(split)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == y_biased.shape[0]
        np.savez(cache_filename, X=X, y=y, y_biased=y_biased)

    npz = np.load(cache_filename)

    X, y, y_biased = npz["X"], npz["y"], npz["y_biased"]

    if include_biased:
        return tf.data.Dataset.from_tensor_slices((X, y, y_biased))
    else:
        return tf.data.Dataset.from_tensor_slices((X, y))


def filter_for_id(X, y, y_biased):
    return y == y_biased


def filter_for_ood(X, y, y_biased):
    return y != y_biased


def discard_biased(X, y, y_biased):
    return X, tf.cast(y, tf.int32)


def load_waterbirds_id(split: str):
    D = impl_load_waterbirds(split, True)
    return D.filter(filter_for_id).map(discard_biased)


def load_waterbirds_ood(split: str):
    D = impl_load_waterbirds(split, True)
    return D.filter(filter_for_ood).map(discard_biased)


def load_dataset(dataset: str, split: str) -> tf.data.Dataset:
    assert is_waterbirds(dataset)
    if dataset == "waterbirds_id":
        return load_waterbirds_id(split)
    else:
        assert dataset == "waterbirds_ood"
        return load_waterbirds_ood(split)


def is_waterbirds(dataset_name: str):
    return dataset_name in ["waterbirds_id", "waterbirds_ood"]


def get_image_size(dataset_name: str):
    return (*IMAGE_SIZE, 3)
