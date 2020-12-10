import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from lib_biased_mnist import make_biased_mnist_data
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import time
import reproduction.lib_data as lib_data
from pathlib import Path


# By default, the total size of target ID + target OOD is 20k for all settings.
# If we want to use a holdout target set, then we fetch a target set that has 20k elements and skip the first
# `target_size` elements, that were used for training.
DEFAULT_TOTAL_TARGET_SIZE = 20_000

SHUFFLE_SIZE = 80_000


class DataWrapper:
    def __init__(
        self,
        source_dataset,
        ood_dataset,
        source_train_and_valid_size,
        target_size,
        valid_ratio,
        target_ood_ratio,
        use_data_augmentation,
        labeling_scheme,
        labeling_index,
        batch_size,
        is_eval=False,
        use_holdout_target=False,
    ):
        self.batch_size = batch_size
        self.use_data_augmentation = use_data_augmentation
        self.labeling_scheme = labeling_scheme

        self.num_classes = lib_data.get_num_classes(source_dataset, reindex_labels=True)
        self.target_ood_ratio = target_ood_ratio

        if use_holdout_target and target_size == DEFAULT_TOTAL_TARGET_SIZE:
            raise RuntimeError(
                "Cannot use holdout target, because all the target set has been already used for training."
            )

        # Load the source training and validation sets. These will only be used for evaluation.
        self.source_train_data, self.valid_data = preprocess_dataset(
            source_dataset,
            "train",
            source_train_and_valid_size,
            valid_ratio,
        )
        self.image_shape = list(
            tf.compat.v1.data.get_output_shapes(self.source_train_data)[0]
        )

        if labeling_scheme == "binary_classifier":
            self.source_train_data = self.source_train_data.map(
                lambda x, y: (x, tf.cast(0, tf.int32))
            )
            self.valid_data = self.valid_data.map(
                lambda x, y: (x, tf.cast(0, tf.int32))
            )

        # If labeling_scheme is None then we have to prepare data for a vanilla ensemble so we don't care about target.
        if labeling_scheme is None:
            if is_eval:
                self.train_data = None
            else:
                if self.use_data_augmentation:
                    self.train_data = (
                        self.source_train_data.cache()
                        .map(
                            self.distort_training_samples_fn,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        )
                        .shuffle(SHUFFLE_SIZE)
                    )
                else:
                    self.train_data = self.source_train_data.cache().shuffle(
                        SHUFFLE_SIZE
                    )
                self.train_data = self.train_data.batch(batch_size).prefetch(
                    tf.data.experimental.AUTOTUNE
                )

            self.source_train_data = (
                self.source_train_data.cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            self.valid_data = (
                self.valid_data.cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            self.target_id_data, self.target_ood_data = None, None

            self.train_size = get_dataset_size(self.train_data)
            self.source_train_size = get_dataset_size(self.source_train_data)
            self.valid_size = get_dataset_size(self.valid_data)
            self.target_id_size, self.target_ood_size = 0, 0

            print(f"train_size: {self.train_size}.")
            print(f"source_train_size: {self.source_train_size}.")
            print(f"valid_size: {self.valid_size}.")

            return

        # orig_target_id_data will have the same samples as target_id_data, but it will have the original labels.
        # Only used for evaluation.
        self.orig_target_id_data, _ = preprocess_dataset(
            source_dataset,
            "test",
            int(target_size * (1 - self.target_ood_ratio)),
            use_complement=use_holdout_target,
        )
        self.target_id_data = self.orig_target_id_data

        use_complement_for_ood = use_holdout_target

        if source_dataset == ood_dataset:
            assert source_dataset == "cifar10"
            assert target_size == 10_000
            use_complement_for_ood = True

        self.target_ood_data, _ = preprocess_dataset(
            ood_dataset,
            "test",
            int(target_size * self.target_ood_ratio),
            use_complement=use_complement_for_ood,
        )
        # Original labels no longer apply, but all ID data has label 0.
        if labeling_scheme == "binary_classifier":
            self.orig_target_id_data = self.orig_target_id_data.map(
                lambda x, y: (x, tf.cast(0, tf.int32))
            )

        # Assign random labels to target_id and target_ood, and then concatenate them over train_data.
        self.target_id_data = self.random_label_function(
            self.target_id_data, labeling_scheme, labeling_index
        )
        self.target_ood_data = self.random_label_function(
            self.target_ood_data, labeling_scheme, labeling_index
        )

        self.target_id_size = get_dataset_size(self.target_id_data)
        self.target_ood_size = get_dataset_size(self.target_ood_data)

        if target_ood_ratio == 0.5 and self.target_id_size != self.target_ood_size:
            final_size = min(self.target_id_size, self.target_ood_size)
            self.target_id_data = self.target_id_data.take(final_size)
            self.target_ood_data = self.target_ood_data.take(final_size)

            self.target_id_size = final_size
            self.target_ood_size = final_size

        if is_eval:
            self.train_data = None
        else:
            print(f"Total data size: {source_train_and_valid_size + target_size:,}.")

            print(f"self.source_train_data: { self.source_train_data.element_spec }")
            print(f"self.target_id_data: { self.target_id_data.element_spec }")
            print(f"self.target_ood_data: { self.target_ood_data.element_spec }")

            self.train_data = self.source_train_data.concatenate(
                self.target_id_data.concatenate(self.target_ood_data)
            ).cache()
            if self.use_data_augmentation:
                self.train_data = self.train_data.map(
                    self.distort_training_samples_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            self.train_data = (
                self.train_data.shuffle(SHUFFLE_SIZE)
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

        # Batch and set prefetch for all datasets.
        self.target_id_data = (
            self.target_id_data.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.target_ood_data = (
            self.target_ood_data.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.orig_target_id_data = (
            self.orig_target_id_data.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.source_train_data = (
            self.source_train_data.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.valid_data = (
            self.valid_data.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.train_size = get_dataset_size(self.train_data)
        self.source_train_size = get_dataset_size(self.source_train_data)
        self.valid_size = get_dataset_size(self.valid_data)

    def random_label_function(self, dataset, labeling_scheme, labeling_index):
        if labeling_scheme == "one_label":
            # Assigns a single (random) label to the whole set.
            label = labeling_index
            return dataset.map(
                lambda x, y: (x, tf.cast(label, tf.int32)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        elif labeling_scheme == "keep_original":
            # Keeps the original labels.
            return dataset
        elif labeling_scheme == "shift_left":
            # Shifts the original labels by shift_offset to the left (e.g. label 1 becomes 2 if shift_offset == 1).
            shift_offset = labeling_index + 1
            return dataset.map(
                lambda x, y: (
                    x,
                    (y + self.num_classes - shift_offset) % self.num_classes,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        elif labeling_scheme == "uniform_random":
            tf.random.set_seed(hash(dataset))
            # Assigns a random label sampled uniformly from the set of possible labels.
            return dataset.map(
                lambda x, y: (
                    x,
                    tf.random.uniform(
                        shape=y.shape,
                        maxval=self.num_classes,
                        dtype=tf.dtypes.int32,
                    ),
                ),
                num_parallel_calls=1,
            )
        elif labeling_scheme == "cluster_shift_left":
            # Clusters the target set. Then shifts the cluster labels by shift_offset to the left.
            shift_offset = labeling_index + 1
            return dataset.map(
                lambda x, y: (
                    x,
                    (y + self.num_classes - shift_offset) % self.num_classes,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        elif labeling_scheme == "binary_classifier":
            return dataset.map(lambda x, y: (x, tf.cast(1, tf.int32)))
        else:
            raise RuntimeError("Unknown labeling scheme {}".format(labeling_scheme))

    def distort_training_samples_fn(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
        image = tf.image.random_crop(image, self.image_shape)
        return image, label

    @staticmethod
    def pretrained_model(input_shape):
        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = tf.keras.applications.VGG16(
            include_top=False, input_shape=input_shape, pooling="avg"
        )(x)
        return tf.keras.Model(inputs=[inputs], outputs=[x])

    # Fix a deterministic labeling for the clusters that result from k-means.
    @staticmethod
    def convert_cluster_labels_to_canonical(labels, num_clusters):
        canonical_labels_map = {}
        i, j = 0, 0
        while j < num_clusters:
            i += 1
            if labels[i - 1] in canonical_labels_map:
                continue
            canonical_labels_map[labels[i - 1]] = j
            j += 1
        return np.array([canonical_labels_map[l] for l in labels])


# Preprocesses and loads dataset as tf.data.Dataset.
# If use_complement is set, then instead of taking the first dataset_size items, it will skip the first
# dataset_size items.
def preprocess_dataset(
    dataset_name,
    split,
    dataset_size,
    split_ratio=0.0,
    use_complement=False,
):
    if dataset_name == "biased_mnist_corr":
        data = generate_correlated_data()
        data = data.map(
            lambda x, y: (tf.cast(tf.clip_by_value(x, 0.0, 1.0), tf.float32), y)
        )
    elif dataset_name == "biased_mnist_uncorr":
        data = generate_uncorrelated_data()
        data = data.map(
            lambda x, y: (tf.cast(tf.clip_by_value(x, 0.0, 1.0), tf.float32), y)
        )
    else:
        data = lib_data.load_dataset(dataset_name, split, reindex_labels=True)

    data = data.shuffle(SHUFFLE_SIZE, seed=0)

    if use_complement:
        data = data.skip(dataset_size)
    else:
        data = data.take(dataset_size)

    actual_dataset_size = get_dataset_size(data)
    if actual_dataset_size < dataset_size:
        dataset_size = actual_dataset_size

    split2_size = int(dataset_size * split_ratio)

    if split2_size > 0:
        split2_data = data.take(split2_size)
    else:
        split2_data = None

    split1_data = data.skip(split2_size)
    return split1_data, split2_data


def get_dataset_size(dataset):
    if dataset is None:
        return 0

    size = 0
    for batch_X, batch_y in dataset:
        y_shape = batch_y.shape
        size += y_shape[0] if len(y_shape) > 0 else 1
    return size


def pretty_dataset_name(dataset_name):
    return (
        dataset_name.replace(":", "")
        .replace(",", "")
        .replace("/", "_")
        .replace("+", "_and_")
    )


def get_labeling_scheme(ensemble_type):
    labeling_schemes = {
        "vanilla": None,
        "assign_one_label": "one_label",
        "assign_per_class": "shift_left",
        "uniform_random": "uniform_random",
        "assign_per_cluster": "cluster_shift_left",
        "binary_classifier": "binary_classifier",
    }
    return labeling_schemes[ensemble_type]


def load_pretrained_model_for_dataset(source_dataset, model_arch, nn_depth):
    root = os.environ.get("RANDOM_LABELS_PRETRAINED_MODELS_ROOT")
    model_path = os.path.join(
        root,
        "{}_{}_depth{}".format(
            pretty_dataset_name(source_dataset), model_arch, nn_depth
        ),
    )
    model = tf.keras.models.load_model(model_path)

    if model_arch == "resnet":
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["acc"],
    )
    return model


# Generators for BiasedMNIST.
def generate_correlated_data():
    return (
        tf.data.Dataset.from_generator(
            make_biased_mnist_data("~/.datasets/mnist/", 1.0, train=True),
            output_types=(tf.float32, tf.int32),
        )
        .cache()
        .shuffle(60_000)
    )


def generate_uncorrelated_data():
    return (
        tf.data.Dataset.from_generator(
            make_biased_mnist_data("~/.datasets/mnist/", 0.0, train=False),
            output_types=(tf.float32, tf.int32),
        )
        .cache()
        .shuffle(10_000)
    )
