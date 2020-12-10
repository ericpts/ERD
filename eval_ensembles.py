import argparse
import mlflow
import numpy as np
import os
import pickle
import sklearn.metrics as metrics
from pathlib import Path
import sys
import tensorflow as tf
import time
import traceback
import utils
import reproduction.lib_data as lib_data
from typing import Dict, Tuple


def retry(f):
    while True:
        try:
            return f()
        except:
            print("Retrying MLFlow...")
            time.sleep(5)
            continue


def aggregate_ensemble(
    exp_dir,
    ckpt_epoch_or_best,
    ensemble_size,
    data,
    only_probs_for_storing=False,
):
    # Load model checkpoints for all models in the ensemble.
    models = []
    for i in range(ensemble_size):
        ckpt_file = (
            "model.{:02d}".format(int(ckpt_epoch_or_best))
            if ckpt_epoch_or_best != "best"
            else "model.best"
        )
        curr_model_path = os.path.join(exp_dir, "model_{}".format(i), ckpt_file)
        if not os.path.exists(curr_model_path):
            print("Couldn't load model from {}.".format(curr_model_path))
            break
        curr_model = tf.keras.models.load_model(curr_model_path)
        models.append(curr_model)
    if len(models) != ensemble_size:
        return None, None, None, None, None

    val_probs, id_probs, ood_probs = [], [], []

    for model in models:
        val_probs.append(
            model.predict(data.valid_data).reshape(data.valid_size, data.num_classes)
        )
        id_probs.append(
            model.predict(data.target_id_data).reshape(
                data.target_id_size, data.num_classes
            )
        )
        ood_probs.append(
            model.predict(data.target_ood_data).reshape(
                data.target_ood_size, data.num_classes
            )
        )
    val_probs, id_probs, ood_probs = (
        np.array(val_probs),
        np.array(id_probs),
        np.array(ood_probs),
    )
    if only_probs_for_storing == False:
        id_diffs, ood_diffs = get_diffs(id_probs), get_diffs(ood_probs)
        return (
            val_probs.mean(axis=0),
            id_probs.mean(axis=0),
            ood_probs.mean(axis=0),
            id_diffs,
            ood_diffs,
        )
    else:
        return val_probs, id_probs, ood_probs, None, None


def get_diffs(all_probs):
    ensemble_size = all_probs.shape[0]
    all_probs_t = np.transpose(all_probs, axes=(1, 0, 2))
    diffs = []
    for ens_outputs in all_probs_t:
        curr_sample_diffs = []
        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                curr_sample_diffs.append(
                    np.linalg.norm(ens_outputs[i] - ens_outputs[j], ord=1)
                )
        diffs.append(curr_sample_diffs)
    return np.array(diffs)


def compute_entropies(test_probabilities):
    return -np.sum(np.log(test_probabilities + 1e-10) * test_probabilities, axis=-1)


def get_test_statistics(
    id_test_probs,
    ood_test_probs,
    id_test_diffs,
    ood_test_diffs,
    decision_type,
) -> Tuple[float, float]:
    if decision_type == "max_p":
        id_statistics = -np.max(id_test_probs, axis=-1)
        ood_statistics = -np.max(ood_test_probs, axis=-1)
    elif decision_type == "entropy":
        id_statistics = compute_entropies(id_test_probs)
        ood_statistics = compute_entropies(ood_test_probs)
    elif decision_type == "avg_diff":
        id_statistics = id_test_diffs.mean(axis=1)
        ood_statistics = ood_test_diffs.mean(axis=1)
    elif decision_type == "max_diff":
        id_statistics = id_test_diffs.max(axis=1)
        ood_statistics = ood_test_diffs.max(axis=1)
    elif decision_type == "binary_classifier":
        id_statistics = id_test_probs[:, 1]
        ood_statistics = ood_test_probs[:, 1]
    else:
        raise RuntimeError("Unrecognized decision type.")

    return (id_statistics, ood_statistics)


def get_labels(dataset):
    labels = []
    for _, batch_y in dataset:
        labels += [batch_y.numpy()]
    return np.concatenate(labels)


def get_ensemble_accuracy(dataset, ensemble_probs):
    true_labels = get_labels(dataset)
    pred_labels = np.argmax(ensemble_probs, axis=-1)
    ensemble_accuracy = (true_labels == pred_labels).mean()
    return ensemble_accuracy


def load_data(params, use_holdout_target, arg_target_size, arg_target_ood_ratio):
    if "total_source_train_and_valid_size" not in params:
        params["total_source_train_and_valid_size"] = 50000
    if "total_target_size" not in params:
        params["total_target_size"] = 20000

    return utils.DataWrapper(
        source_dataset=params["source_dataset"],
        ood_dataset=params["ood_dataset"],
        source_train_and_valid_size=params["total_source_train_and_valid_size"],
        target_size=(
            arg_target_size
            if arg_target_size is not None
            else params["total_target_size"]
        ),
        valid_ratio=params["valid_ratio"],
        target_ood_ratio=(
            arg_target_ood_ratio
            if arg_target_ood_ratio is not None
            else (params["target_ood_ratio"] if "target_ood_ratio" in params else 0.5)
        ),
        batch_size=params["batch_size"],
        use_data_augmentation=False,
        labeling_scheme="keep_original",
        labeling_index=0,
        is_eval=True,
        use_holdout_target=use_holdout_target,
    )


def is_default_target_config(params, args):
    if (
        (
            args.target_size is not None
            and params["total_target_size"] != args.target_size
        )
        or (
            args.target_ood_ratio is not None
            and params["target_ood_ratio"] != args.target_ood_ratio
        )
        or (args.ensemble_size != params["ensemble_size"])
    ):
        return False
    return True


def main(args):
    assert os.path.exists(args.exp_dir)

    with open(os.path.join(args.exp_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    client = mlflow.tracking.MlflowClient()
    for im in range(args.ensemble_size):
        model_subdir = Path(args.exp_dir) / f"model_{im}"
        if (model_subdir / "model.best").exists():
            continue

        val_loss_per_epoch = {
            int(m.step): m.value
            for m in client.get_metric_history(params["run_id"], f"{im}_val_loss")
        }

        saved_epochs = []
        for sd in model_subdir.iterdir():
            epoch = int(sd.name.split(".")[-1])
            saved_epochs.append(epoch)

        best_epoch = None
        for epoch in saved_epochs:
            if (
                best_epoch is None
                or val_loss_per_epoch[epoch] < val_loss_per_epoch[best_epoch]
            ):
                best_epoch = epoch
        assert best_epoch is not None

        (model_subdir / "model.best").symlink_to(
            model_subdir / f"model.{best_epoch:02}", target_is_directory=True
        )

    ckpt_epochs_or_best = args.epochs
    if -1 in ckpt_epochs_or_best:
        ckpt_epochs_or_best = []
        epochs = []
        for ckpt_dir in os.listdir(os.path.join(args.exp_dir, "model_0")):
            epoch = ckpt_dir.rsplit(".", 1)[1]
            if epoch == "best":
                continue
            epochs.append(int(epoch))

        if args.eval_on_all_epochs:
            ckpt_epochs_or_best = epochs + ["best"]
        else:
            ckpt_epochs_or_best = ["best"]
            if len(epochs) > 0:
                ckpt_epochs_or_best.append(max(epochs))

    data = load_data(
        params, args.use_holdout_target, args.target_size, args.target_ood_ratio
    )

    if not params["use_pretrained_model"]:
        args.eval_accuracy_on_pretrained_model = False

    if args.eval_accuracy_on_pretrained_model:
        pretrained_model_for_eval = utils.load_pretrained_model_for_dataset(
            utils.pretty_dataset_name(params["source_dataset"]),
            params["model_arch"],
            params["nn_depth"],
        )
        pretrained_model_val_probs = pretrained_model_for_eval.predict(
            data.valid_data
        ).reshape(data.valid_size, data.num_classes)
        pretrained_model_id_probs = pretrained_model_for_eval.predict(
            data.target_id_data
        ).reshape(data.target_id_size, data.num_classes)
        pretrained_model_ood_probs = pretrained_model_for_eval.predict(
            data.target_ood_data
        ).reshape(data.target_ood_size, data.num_classes)

    assert args.ensemble_size <= params["ensemble_size"]

    all_id_probs, all_ood_probs = {}, {}
    statistics = ["max_p", "entropy"]
    if args.ensemble_size >= 2:
        statistics.extend(["avg_diff", "max_diff"])

    if params["ensemble_type"] == "binary_classifier":
        statistics = ["binary_classifier"]

    mlflow.set_experiment(params["experiment_name"])

    if is_default_target_config(params, args):
        run_id = params["run_id"]
        new_run_name = None
    else:
        # If evaluating a model with a different target configuration (different target size or different OOD
        # ratio) from the one that was stored in MLFlow when the model was trained, then it creates a new run,
        # only with eval data.
        mlflow_client = mlflow.tracking.MlflowClient()
        orig_run = mlflow_client.get_run(params["run_id"])
        new_run_name = "eval_{}".format(orig_run.data.tags.get("mlflow.runName", "run"))
        new_run_params = orig_run.data.params
        new_run_params["total_target_size"] = args.target_size
        new_run_params["target_ood_ratio"] = args.target_ood_ratio
        new_run_params["is_eval_only"] = True
        new_run_params["training_run_id"] = params["run_id"]
        new_run_params["ensemble_size"] = args.ensemble_size
        run_id = None

    with mlflow.start_run(run_id, run_name=new_run_name):
        if run_id is None:
            mlflow.log_params(new_run_params)
            params = new_run_params
        for epoch_or_best in ckpt_epochs_or_best:
            val_probs, id_probs, ood_probs, id_diffs, ood_diffs = aggregate_ensemble(
                args.exp_dir,
                epoch_or_best,
                args.ensemble_size,
                data,
                only_probs_for_storing=True
                if args.store_predictions_path is not None
                else False,
            )

            if id_probs is None:
                continue

            if args.store_predictions_path is not None:
                all_id_probs[epoch_or_best] = id_probs
                all_ood_probs[epoch_or_best] = ood_probs
                continue

            metrics = {}
            if epoch_or_best != "best":
                metrics["ensemble_acc_val"] = get_ensemble_accuracy(
                    data.valid_data, val_probs
                )
                metrics["ensemble_acc_target_id"] = get_ensemble_accuracy(
                    data.target_id_data, id_probs
                )
                metrics["ensemble_acc_target_ood"] = get_ensemble_accuracy(
                    data.target_ood_data, ood_probs
                )
            else:
                metrics["heur_ensemble_acc_val"] = get_ensemble_accuracy(
                    data.valid_data, val_probs
                )
                metrics["heur_ensemble_acc_target_id"] = get_ensemble_accuracy(
                    data.target_id_data, id_probs
                )
                metrics["heur_ensemble_acc_target_ood"] = get_ensemble_accuracy(
                    data.target_ood_data, ood_probs
                )

            if args.eval_accuracy_on_pretrained_model:
                metrics["pretrained_model_acc_val"] = get_ensemble_accuracy(
                    data.valid_data, pretrained_model_val_probs
                )

            for statistic in statistics:
                id_statistics, ood_statistics = get_test_statistics(
                    id_probs,
                    ood_probs,
                    id_diffs,
                    ood_diffs,
                    decision_type=statistic,
                )

                if args.eval_accuracy_on_pretrained_model:
                    id_y_pred = np.argmax(pretrained_model_id_probs, axis=1)
                    ood_y_pred = np.argmax(pretrained_model_ood_probs, axis=1)
                else:
                    id_y_pred = np.argmax(id_probs, axis=1)
                    ood_y_pred = np.argmax(ood_probs, axis=1)

                metrics_dict = lib_data.get_eval_metrics(
                    id_y_true=get_labels(data.target_id_data),
                    id_y_pred=id_y_pred,
                    id_test_statistic=id_statistics,
                    ood_y_true=get_labels(data.target_ood_data),
                    ood_y_pred=ood_y_pred,
                    ood_test_statistic=ood_statistics,
                    tpr_level=95,
                    tnr_level=95,
                    compute_acc_on_id_and_ood=lib_data.get_compute_acc_on_id_and_ood(
                        params["ood_dataset"]
                    ),
                )

                print("{} AUROC: {}".format(statistic, metrics_dict["auroc"]))
                print("{} AUPR: {}".format(statistic, metrics_dict["aupr"]))
                if epoch_or_best != "best":
                    metrics["auroc_{}".format(statistic)] = metrics_dict["auroc"]
                    metrics["aupr_{}".format(statistic)] = metrics_dict["aupr"]
                    metrics["ap_{}".format(statistic)] = metrics_dict["ap"]
                    metrics["fpr95_{}".format(statistic)] = metrics_dict["fpr_at_tpr"]
                    metrics["tpr95_{}".format(statistic)] = metrics_dict["tpr_at_tnr"]
                    metrics["acc_at_tnr_{}".format(statistic)] = metrics_dict[
                        "acc_at_tnr"
                    ]
                    metrics["_95_{}".format(statistic)] = metrics_dict["approx_tpr"]
                else:
                    metrics["heur_auroc_{}".format(statistic)] = metrics_dict["auroc"]
                    metrics["heur_aupr_{}".format(statistic)] = metrics_dict["aupr"]
                    metrics["heur_ap_{}".format(statistic)] = metrics_dict["ap"]
                    metrics["heur_fpr95_{}".format(statistic)] = metrics_dict[
                        "fpr_at_tpr"
                    ]
                    metrics["heur_tpr95_{}".format(statistic)] = metrics_dict[
                        "tpr_at_tnr"
                    ]
                    metrics["heur_acc_at_tnr_{}".format(statistic)] = metrics_dict[
                        "acc_at_tnr"
                    ]
                    metrics["heur_95_{}".format(statistic)] = metrics_dict["approx_tpr"]

            if args.use_holdout_target:
                new_metrics = {}
                for k, v in metrics.items():
                    new_metrics["holdout_" + k] = v
                metrics = new_metrics

            def log():
                if epoch_or_best != "best" and args.eval_on_all_epochs:
                    mlflow.log_metrics(metrics, step=int(epoch_or_best))
                else:
                    mlflow.log_metrics(metrics)

            retry(log)

        if args.store_predictions_path is not None:
            with open(args.store_predictions_path, "wb") as f:
                pickle.dump([all_id_probs, all_ood_probs], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="..", help="Path to root.")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        help="Epochs at which to evaluate OOD detection. If -1, it evaluates at all epochs.",
        default=[-1],
    )
    parser.add_argument(
        "--use_holdout_target",
        type=bool,
        default=False,
        help="If true, it uses a holdout target set for evaluation.",
    )
    parser.add_argument("--ensemble_size", type=int, default=5, help="Ensemble size")
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target size. By default, it uses the target config from the MLFlow run.",
    )
    parser.add_argument(
        "--target_ood_ratio",
        type=float,
        default=None,
        help="Target OOD ratio. By defaul, it uses the target config from the MLFlow run.",
    )
    parser.add_argument(
        "--eval_accuracy_on_pretrained_model",
        action="store_true",
        default=False,
        help="When computing accuracy @ tnr, get the predicted labels "
        "from the original pretrained model, not from the fine tuned ones.",
    )
    parser.add_argument(
        "--store_predictions_path",
        type=str,
        default=None,
        help="If set, it doesn't compute AUROCs, but only stores the predictions to the location indicated.",
    )
    parser.add_argument(
        "--eval_on_all_epochs",
        action="store_true",
        help="If given, eval and report metrics on all epochs. "
        "By default, it will only report the best results.",
    )
    args = parser.parse_args()
    main(args)
