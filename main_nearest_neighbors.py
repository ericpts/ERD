import pickle
import argparse
import sklearn
import tensorflow as tf
from pathlib import Path
import utils
import os
import reproduction.lib_data as lib_data
import mlflow
import numpy as np
from contextlib import contextmanager
import time
from pathlib import Path


@contextmanager
def print_elapsed(message):
    t0 = time.time()
    try:
        yield None
    finally:
        print(f"{message}; took {time.time() - t0:.2f}")


def build_index(X: tf.Tensor):
    model = sklearn.neighbors.NearestNeighbors(algorithm="ball_tree", n_jobs=-1)
    model.fit(X)
    return model


def extract_X(D: tf.data.Dataset) -> tf.Tensor:
    X = []
    for x, _ in D:
        X.append(x)
    num_examples = len(X)
    X = np.stack(X)
    X = np.reshape(X, (num_examples, -1))
    return X


def extract_y(D: tf.data.Dataset) -> tf.Tensor:
    ys = []
    for _, y in D:
        ys.append(y)
    num_examples = len(ys)
    y = np.stack(ys)
    assert y.shape == (num_examples,)
    return y


def build_cache(model, n_neighbors, base_cache_dir: Path, ood: str):
    ood_cache = base_cache_dir / f"{ood}.pickle"
    ood_cache.parent.mkdir(parents=True, exist_ok=True)
    D_ood_test = lib_data.load_dataset(ood, "test")

    if not ood_cache.exists():
        X_ood_test = extract_X(D_ood_test)
        with print_elapsed(f"Queried OOD({ood}) test"):
            ood_test_dist, ood_test_indices = model.kneighbors(
                X_ood_test, n_neighbors=n_neighbors
            )
        ood_test_dist = np.mean(ood_test_dist, axis=-1)

        with ood_cache.open("wb") as f:
            pickle.dump((ood_test_dist, ood_test_indices), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
    )
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, action="append", required=True)
    parser.add_argument("--n_neighbors", type=int, default=8)
    args = parser.parse_args()

    lib_data.setup_mlflow()

    base_cache_dir = (
        Path(os.environ["SCRATCH"])
        / args.experiment_name
        / "cache"
        / args.id_dataset
        / str(args.n_neighbors)
    )
    base_cache_dir.mkdir(exist_ok=True, parents=True)

    D_train = lib_data.load_dataset(args.id_dataset, "train")
    D_id_test = lib_data.load_dataset(args.id_dataset, "test")

    print(f"Processing {args.id_dataset} vs {args.ood_dataset}")

    with print_elapsed("Built dataset"):
        X_train = extract_X(D_train)
        y_train = extract_y(D_train)
        X_id_test = extract_X(D_id_test)

    index_cache = base_cache_dir / "index.pickle"

    if not index_cache.exists():
        with print_elapsed("Built index"):
            model = build_index(X_train)

        with index_cache.open("wb") as f:
            pickle.dump(model, f)

    with index_cache.open("rb") as f:
        model = pickle.load(f)

    mlflow.set_experiment(args.experiment_name)

    id_cache = base_cache_dir / "id.pickle"
    if not id_cache.exists():
        id_test_dist, id_test_indices = model.kneighbors(
            X_id_test, n_neighbors=args.n_neighbors
        )
        id_test_dist = np.mean(id_test_dist, axis=-1)
        with id_cache.open("wb") as f:
            pickle.dump((id_test_dist, id_test_indices), f)

    with id_cache.open("rb") as f:
        (id_test_dist, id_test_indices) = pickle.load(f)

    for ood in args.ood_dataset:
        build_cache(model, args.n_neighbors, base_cache_dir, ood)

    for ood in args.ood_dataset:
        print(f"Processing {ood}")
        D_ood_test = lib_data.load_dataset(ood, "test")
        with mlflow.start_run(run_name="nearest_neighbors") as curr_run:
            mlflow.log_params(
                {
                    "id_dataset": args.id_dataset,
                    "ood_dataset": ood,
                    "n_neighbors": args.n_neighbors,
                    "run_id": curr_run.info.run_id,
                    "experiment_id": curr_run.info.experiment_id,
                    "experiment_name": args.experiment_name,
                }
            )

            ood_cache = base_cache_dir / f"{ood}.pickle"
            assert ood_cache.exists()

            with ood_cache.open("rb") as f:
                (ood_test_dist, ood_test_indices) = pickle.load(f)

            def predict_from_indices(y, indices):
                num_classes = lib_data.get_num_classes(args.id_dataset)
                big = np.zeros((y.shape[0], num_classes))
                big[np.arange(y.size), y] = 1
                per_class_probabilities = np.mean(big[indices], axis=1)
                assert per_class_probabilities.shape[0] == indices.shape[0]
                assert per_class_probabilities.shape[1] == num_classes
                return np.argmax(per_class_probabilities, axis=-1)

            id_y_true = extract_y(D_id_test)
            id_y_pred = predict_from_indices(y_train, id_test_indices)

            ood_y_true = extract_y(D_ood_test)
            ood_y_pred = predict_from_indices(y_train, ood_test_indices)

            metrics = lib_data.get_eval_metrics(
                id_y_true,
                id_y_pred,
                id_test_dist,
                ood_y_true,
                ood_y_pred,
                ood_test_dist,
                compute_acc_on_id_and_ood=lib_data.get_compute_acc_on_id_and_ood(ood),
            )

            if metrics["auroc"] < 0.5:
                metrics = lib_data.get_eval_metrics(
                    id_y_true,
                    id_y_pred,
                    -id_test_dist,
                    ood_y_true,
                    ood_y_pred,
                    -ood_test_dist,
                    compute_acc_on_id_and_ood=lib_data.get_compute_acc_on_id_and_ood(
                        ood
                    ),
                )

            mlflow.log_metrics(metrics)
            print(metrics)


if __name__ == "__main__":
    main()
