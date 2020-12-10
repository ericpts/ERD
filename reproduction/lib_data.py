import numpy as np
from pathlib import Path
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from typing import Tuple, List, Optional
import sklearn.metrics as metrics
from .lib_image import normalize_image
from . import lib_celeb_a
from . import lib_waterbirds

import sys

sys.path.append(str(Path(__file__).parent.parent))


def disable_tf_memory_hog():
    tf.config.set_visible_devices([], "GPU")


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def fix_label_to(D: tf.data.Dataset, label: int) -> tf.data.Dataset:
    def f_example(X, y):
        return X, label

    return D.map(f_example)


def load_objectnet(dataset_name: str) -> tf.data.Dataset:
    assert "/" in dataset_name
    prefix, size = dataset_name.split("/")
    assert prefix == "objectnet"

    assert "x" in size
    width, height = map(int, size.split("x"))
    assert width == height

    base_dir = (
        Path(os.environ["SCRATCH"])
        / ".datasets"
        / "objectnet"
        / f"objectnet_{width}x{height}"
        / "box"
    )

    assert base_dir.exists()

    D = tf.keras.preprocessing.image_dataset_from_directory(
        str(base_dir),
        image_size=(width, height),
    ).unbatch()
    return D


def parse_dataset_name(raw_dataset_name: str) -> Tuple[str, np.ndarray]:
    dataset_name, str_filter_labels = raw_dataset_name.split(":")
    if "-" not in str_filter_labels:
        filter_labels = [int(l) for l in str_filter_labels.split(",")]
    else:
        first, last = [int(l) for l in str_filter_labels.split("-")]
        filter_labels = list(range(first, last))
    return dataset_name, np.sort(filter_labels)


# A|B means binary classification where all of A is one class, and all
# of B is the other class.
def load_datasets_as_classes(dataset_name: str, split: str) -> tf.data.Dataset:
    assert "+" in dataset_name
    datasets = []
    for cls, ds in enumerate(dataset_name.split("+")):
        D = load_dataset(ds, split).map(lambda X, y: (X, cls))
        datasets.append(D)

    D_acc = datasets[0]
    for d in datasets[1:]:
        D_acc = D_acc.concatenate(d)
    return D_acc


def load_dataset(
    dataset_name: str,
    split: str,
    reindex_labels: bool = False,
) -> tf.data.Dataset:
    if "+" in dataset_name:
        return load_datasets_as_classes(dataset_name, split)

    filter_labels = None  # type: Optional[List[int]]

    if ":" in dataset_name:
        dataset_name, filter_labels = parse_dataset_name(dataset_name)

    if "imagenet" in dataset_name:
        assert split != "val", "TODO: Split validation from the training data."
        if split == "test":
            split = "validation"

    if "objectnet" in dataset_name:
        assert split == "test"
        D = load_objectnet(dataset_name)
    elif lib_medical_ood.is_medical(dataset_name):
        return lib_medical_ood.load_dataset(dataset_name, split)
    elif lib_celeb_a.is_celeb_a(dataset_name):
        return lib_celeb_a.load_dataset(dataset_name, split)
    elif lib_waterbirds.is_waterbirds(dataset_name):
        return lib_waterbirds.load_dataset(dataset_name, split)
    else:
        D = tfds.load(
            dataset_name,
            split=split,
            data_dir=os.path.join(os.getenv("SCRATCH", "~/"), ".datasets"),
            as_supervised=True,
            try_gcs=False,
        )

    if filter_labels is not None:

        def f_filter(X, y) -> bool:
            return tf.math.count_nonzero(tf.equal(y, filter_labels)) > 0

        def f_remap_labels(X, y):
            y = tf.argmax(tf.equal(y, filter_labels))
            return X, y

        D = D.filter(f_filter)
        if reindex_labels:
            D = D.map(f_remap_labels)

    def f_normalize_image(X, y):
        X = tf.cast(X, "float32") / 255.0
        y = tf.cast(y, "int32")
        return X, y

    D = D.map(f_normalize_image)

    return D


def to_rgb_32_x_32(D: tf.data.Dataset) -> tf.data.Dataset:
    def f_map(X, y):
        X = tf.image.resize(X, (32, 32))
        X = tf.image.grayscale_to_rgb(X)
        return X, y

    return D.map(f_map)


def to_torch(D: tf.data.Dataset) -> CustomTensorDataset:
    Xs = []
    ys = []
    for X, y in D:
        Xs.append(np.transpose(X.numpy(), [2, 0, 1]))
        ys.append(int(y.numpy()))

    Xs = np.stack(Xs)
    ys = np.stack(ys)

    Xs = torch.from_numpy(Xs)
    ys = torch.from_numpy(ys)

    return CustomTensorDataset((Xs, ys))


def get_num_classes(dataset_name: str, reindex_labels: bool = False) -> int:
    if lib_medical_ood.is_medical(dataset_name):
        return lib_medical_ood.get_num_classes(dataset_name)

    if lib_celeb_a.is_celeb_a(dataset_name):
        return 2

    if lib_waterbirds.is_waterbirds(dataset_name):
        return 2

    if "+" in dataset_name:
        return len(dataset_name.split("+"))

    if ":" in dataset_name:
        dataset_name, filter_labels = parse_dataset_name(dataset_name)
        if reindex_labels:
            return len(filter_labels)

    builder = tfds.builder(dataset_name)
    info = builder.info
    return info.features["label"].num_classes


def get_image_size(dataset_name: str) -> Tuple[int, int, int]:
    if "+" in dataset_name:
        datasets = dataset_name.split("+")
        image_size = get_image_size(datasets[0])
        for d in datasets[1:]:
            assert get_image_size(d) == image_size
        return image_size

    if lib_medical_ood.is_medical(dataset_name):
        return lib_medical_ood.get_image_size(dataset_name)

    if lib_celeb_a.is_celeb_a(dataset_name):
        return lib_celeb_a.get_image_size(dataset_name)

    if lib_waterbirds.is_waterbirds(dataset_name):
        return lib_waterbirds.get_image_size(dataset_name)

    if ":" in dataset_name:
        dataset_name, _filter_labels = dataset_name.split(":")
    builder = tfds.builder(dataset_name)
    info = builder.info
    return info.features["image"].shape


def in_tfds(dataset_name: str):
    all_datasets = tfds.list_builders()
    name_without_params = dataset_name.split("/")[0]
    return name_without_params in all_datasets


def get_compute_acc_on_id_and_ood(ood_dataset: str):
    for d in ["corrupted", "objectnet", "cifar10_1"]:
        if d in ood_dataset:
            return True
    return False


def get_eval_metrics(
    id_y_true,
    id_y_pred,
    id_test_statistic,
    ood_y_true,
    ood_y_pred,
    ood_test_statistic,
    tpr_level: float = 95,
    tnr_level: float = 95,
    compute_acc_on_id_and_ood=False,
):
    id_ood_labels = np.concatenate(
        (np.zeros(id_test_statistic.shape[0]), np.ones(ood_test_statistic.shape[0]))
    )
    all_test_statistics = np.concatenate((id_test_statistic, ood_test_statistic))
    auroc = metrics.roc_auc_score(id_ood_labels, all_test_statistics)
    ap = metrics.average_precision_score(id_ood_labels, all_test_statistics)

    # Computing AUPR.
    precision, recall, _ = metrics.precision_recall_curve(
        id_ood_labels, all_test_statistics
    )
    aupr = metrics.auc(recall, precision)

    # Computing FPR@TPR level.
    sorted_ood_statistics = np.sort(ood_test_statistic)
    idx = int((100 - tpr_level) / 100 * sorted_ood_statistics.shape[0])
    while (
        idx + 1 < sorted_ood_statistics.shape[0]
        and sorted_ood_statistics[idx + 1] == sorted_ood_statistics[idx]
    ):
        idx += 1
    tpr_based_threshold = sorted_ood_statistics[idx]
    approx_tpr = 1 - idx / sorted_ood_statistics.shape[0]
    fpr = (id_test_statistic > tpr_based_threshold).mean()

    # Computing TPR@TNR level.
    sorted_id_statistics = np.sort(id_test_statistic)
    idx = int(tnr_level / 100 * sorted_id_statistics.shape[0])
    threshold = sorted_id_statistics[idx]
    tpr = (ood_test_statistic > threshold).mean()

    # Computing accuracy @ TNR level.
    true_labels = np.concatenate((id_y_true, ood_y_true), axis=0)
    acc_at_tnr = get_accuracy_at_tnr_level(
        y_true=true_labels,
        y_pred=np.concatenate((id_y_pred, ood_y_pred), axis=0),
        test_statistic=all_test_statistics,
        tnr_level=tnr_level,
        id_ood_mask=id_ood_labels,
        compute_on_id_and_ood=compute_acc_on_id_and_ood,
    )

    return {
        "auroc": auroc,
        "ap": ap,
        "aupr": aupr,
        "fpr_at_tpr": fpr,
        "tpr_at_tnr": tpr,
        "approx_tpr": approx_tpr,
        "acc_at_tnr": acc_at_tnr,
    }


def get_accuracy_at_tnr_level(
    y_true,
    y_pred,
    test_statistic,
    tnr_level,
    id_ood_mask,
    compute_on_id_and_ood=False,
):
    id_test_statistic = test_statistic[id_ood_mask == 0]
    sorted_id_statistics = np.sort(id_test_statistic)
    idx = int(tnr_level / 100 * sorted_id_statistics.shape[0])
    tnr_based_threshold = sorted_id_statistics[idx]
    if not compute_on_id_and_ood:
        # Count any OOD sample on which we predict a label as incorrect.
        y_true[id_ood_mask == 1] = -1
    idx_filter = np.where(test_statistic < tnr_based_threshold)
    return (y_pred[idx_filter] == y_true[idx_filter]).mean()
