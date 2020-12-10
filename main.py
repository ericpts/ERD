from __future__ import print_function

import argparse
import gin
import shutil
import requests

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import mlflow
import os
import pickle
import tensorflow as tf
import time
import gc
from pathlib import Path

import utils
from models.simple_cnn import simple_cnn
from models.mlp import mlp
from models.vgg import vgg
from models.resnet import resnet_v1
from models.wideresnet import wrn28_10
from models.densenet import densenet121

from reproduction.lib_data import get_image_size

from typing import Optional

def retry(f):
    while True:
        try:
            return f()
        except requests.exceptions.ProxyError:
            print("Retrying MLFlow...")
            time.sleep(3)
            continue


@gin.configurable(
    "data",
    blacklist=[
        "labeling_scheme",
        "labeling_index",
        "params",
    ],
)
def load_data(
    source_dataset,
    ood_dataset,
    source_train_and_valid_size,
    target_size,
    valid_ratio,
    target_ood_ratio,
    batch_size,
    use_data_augmentation,
    labeling_scheme,
    labeling_index,
    params,
):
    params.update(
        {
            "source_dataset": source_dataset,
            "ood_dataset": ood_dataset,
            "total_source_train_and_valid_size": source_train_and_valid_size,
            "total_target_size": target_size,
            "valid_ratio": valid_ratio,
            "target_ood_ratio": target_ood_ratio,
            "batch_size": batch_size,
            "labeling_scheme": labeling_scheme,
            "use_data_augmentation": use_data_augmentation,
        }
    )

    return utils.DataWrapper(
        source_dataset=source_dataset,
        ood_dataset=ood_dataset,
        source_train_and_valid_size=source_train_and_valid_size,
        target_size=target_size,
        valid_ratio=valid_ratio,
        target_ood_ratio=target_ood_ratio,
        batch_size=batch_size,
        use_data_augmentation=use_data_augmentation,
        labeling_scheme=labeling_scheme,
        labeling_index=labeling_index,
    )


def lr_scheduler(epoch, init_lr):
    constant_lr_until_epoch = 10
    lr = init_lr
    if epoch < constant_lr_until_epoch:
        return lr
    else:
        return lr * tf.math.exp(0.05 * (constant_lr_until_epoch - epoch))


def resnet_lr_scheduler(epoch, init_lr):
    lr = init_lr
    if epoch < 50:
        return lr
    elif epoch < 70:
        return lr * 0.2 ** 1
    elif epoch < 90:
        return lr * 0.2 ** 2
    elif epoch < 180:
        return lr * 0.2 ** 3
    else:
        return lr * 0.0005


@gin.configurable("logging")
class LogMetrics(tf.keras.callbacks.Callback):
    def __init__(
        self,
        all_results,
        data,
        model_index,
        training_epochs,
        use_pretrained_model,
        log_every_n_epochs=1,
    ):
        super(LogMetrics, self).__init__()
        self.all_results = all_results
        self.log_every_n_epochs = log_every_n_epochs
        self.data = data
        self.model_index = model_index
        self.training_epochs = training_epochs
        self.use_pretrained_model = use_pretrained_model

    def _eval_model(self, model):
        if hasattr(model, "_eval_data_handler"):
            old_handler = None
            model._eval_data_handler, old_handler = (
                old_handler,
                model._eval_data_handler,
            )
        val_loss, val_acc = model.evaluate(self.data.valid_data, verbose=0)
        _, source_acc = model.evaluate(self.data.source_train_data, verbose=0)
        if self.data.labeling_scheme is not None:
            _, target_id_acc = model.evaluate(self.data.target_id_data, verbose=0)
            _, target_id_true_acc = model.evaluate(
                self.data.orig_target_id_data, verbose=0
            )
            _, target_ood_acc = model.evaluate(self.data.target_ood_data, verbose=0)
        else:
            target_id_acc, target_id_true_acc, target_ood_acc = 0, 0, 0
        if hasattr(model, "_eval_data_handler"):
            model._eval_data_handler, old_handler = (
                old_handler,
                model._eval_data_handler,
            )

        return (
            val_loss,
            val_acc,
            source_acc,
            target_id_acc,
            target_id_true_acc,
            target_ood_acc,
        )

    def on_train_begin(self, logs=None):
        if not self.use_pretrained_model:
            return

        initial_results = {}
        for k, v in logs.items():
            initial_results["{}_{}".format(self.model_index, k)] = v
        (
            val_loss,
            val_acc,
            source_acc,
            target_id_acc,
            target_id_true_acc,
            target_ood_acc,
        ) = self._eval_model(self.model)

        initial_results["{}_val_loss".format(self.model_index)] = val_loss
        initial_results["{}_val_acc".format(self.model_index)] = val_acc
        initial_results["{}_source_train_acc".format(self.model_index)] = source_acc
        initial_results["{}_target_id_train_acc".format(self.model_index)] = (
            target_id_acc or 0
        )
        initial_results["{}_target_id_true_label_acc".format(self.model_index)] = (
            target_id_true_acc or 0
        )
        initial_results[
            "{}_target_ood_train_acc".format(self.model_index)
        ] = target_ood_acc

        retry(lambda: mlflow.log_metrics(initial_results))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every_n_epochs == 0 or epoch == self.training_epochs - 1:
            self.all_results[epoch] = {}
            for k, v in logs.items():
                self.all_results[epoch]["{}_{}".format(self.model_index, k)] = v
            (
                val_loss,
                val_acc,
                source_acc,
                target_id_acc,
                target_id_true_acc,
                target_ood_acc,
            ) = self._eval_model(self.model)

            for v in [
                val_loss,
                val_acc,
                source_acc,
                target_id_acc,
                target_id_true_acc,
                target_ood_acc,
            ]:
                assert v is not None

            self.all_results[epoch]["{}_val_loss".format(self.model_index)] = val_loss
            self.all_results[epoch]["{}_val_acc".format(self.model_index)] = val_acc
            self.all_results[epoch][
                "{}_source_train_acc".format(self.model_index)
            ] = source_acc
            self.all_results[epoch][
                "{}_target_id_train_acc".format(self.model_index)
            ] = (target_id_acc or 0)
            self.all_results[epoch][
                "{}_target_id_true_label_acc".format(self.model_index)
            ] = (target_id_true_acc or 0)
            self.all_results[epoch][
                "{}_target_ood_train_acc".format(self.model_index)
            ] = target_ood_acc

            retry(lambda: mlflow.log_metrics(self.all_results[epoch], step=epoch))


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_filename, training_epochs, ckpt_epoch_freq=10):
        super(CustomModelCheckpoint, self).__init__()
        self.ckpt_filename = ckpt_filename
        self.training_epochs = training_epochs
        self.ckpt_epoch_freq = ckpt_epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.ckpt_epoch_freq == 0:
            self.model.save(self.ckpt_filename.format(epoch=epoch))


class SaveCheckpointEndOfTraining(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_filename, training_epochs):
        super(SaveCheckpointEndOfTraining, self).__init__()
        self.ckpt_filename = ckpt_filename
        self.training_epochs = training_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.training_epochs - 1:
            self.model.save(self.ckpt_filename.format(epoch=epoch))


@gin.configurable(
    "train",
    blacklist=[
        "data",
        "ckpt_dir",
        "params",
        "resumed_model",
        "image_shape",
        "initial_epoch",
        "source_dataset_for_pretrained",
    ],
)
def train(
    data,
    model_arch,
    image_shape,
    epochs,
    lr,
    model_index,
    l2_reg,
    nn_depth,
    ckpt_epoch_freq,
    save_best_ckpt_only,
    ckpt_dir=None,
    params=None,
    use_pretrained_model=False,
    resumed_model=None,
    initial_epoch: int = 0,
    source_dataset_for_pretrained: str = None,
):
    if model_index == 0:
        params.update(
            {
                "model_arch": model_arch,
                "image_shape": image_shape,
                "epochs": epochs,
                "lr": lr,
                "l2_reg": l2_reg,
                "nn_depth": nn_depth,
                "ckpt_epoch_freq": ckpt_epoch_freq,
                "save_best_ckpt_only": save_best_ckpt_only,
                "use_pretrained_model": use_pretrained_model,
            }
        )
        retry(lambda: mlflow.log_params(params))

    if resumed_model is not None:
        model: tf.keras.Model = resumed_model
    elif not use_pretrained_model:
        if model_arch == "mlp":
            model = mlp(
                input_shape=image_shape,
                l2_reg=l2_reg,
                num_layers=nn_depth,
                num_classes=data.num_classes,
            )
        elif model_arch == "simple_cnn":
            model = simple_cnn(input_shape=image_shape, num_classes=data.num_classes)
        elif model_arch == "vgg":
            model = vgg(
                input_shape=image_shape,
                l2_reg=l2_reg,
                nn_depth=nn_depth,
                num_classes=data.num_classes,
            )
        elif model_arch == "resnet":
            model = resnet_v1(
                input_shape=image_shape,
                depth=nn_depth,
                num_classes=data.num_classes,
                l2_reg=l2_reg,
            )
        elif model_arch == "wideresnet":
            model = wrn28_10(
                input_shape=image_shape,
                depth=nn_depth,
                num_classes=data.num_classes,
                l2_reg=l2_reg,
            )
        elif model_arch == "densenet":
            model = densenet121(
                input_shape=image_shape,
                depth=nn_depth,
                num_classes=data.num_classes,
            )
        else:
            raise RuntimeError("Unknown model type.")
    else:
        model = utils.load_pretrained_model_for_dataset(
            source_dataset=source_dataset_for_pretrained,
            model_arch=model_arch,
            nn_depth=nn_depth,
        )

    if save_best_ckpt_only:
        ckpt_callback = ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "model.best"),
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch",
        )
    else:
        ckpt_callback = CustomModelCheckpoint(
            ckpt_filename=os.path.join(ckpt_dir, "model.{epoch:02d}"),
            training_epochs=epochs,
            ckpt_epoch_freq=ckpt_epoch_freq,
        )
    end_of_training_ckpt_callback = SaveCheckpointEndOfTraining(
        ckpt_filename=os.path.join(ckpt_dir, "model.{epoch:02d}"),
        training_epochs=epochs,
    )

    lr_scheduler_fn = None

    if model_arch in ["resnet", "wideresnet", "densenet"]:
        lr_scheduler_fn = resnet_lr_scheduler
    elif model_arch in ["mlp", "simple_cnn", "vgg"]:
        lr_scheduler_fn = lr_scheduler
    else:
        assert False, f"Unknown model arch: {model_arch}"

    all_results = {}
    model.fit(
        x=data.train_data,
        epochs=epochs,
        validation_data=data.valid_data,
        callbacks=[
            ckpt_callback,
            end_of_training_ckpt_callback,
            LearningRateScheduler(
                lambda epoch: lr_scheduler_fn(epoch=epoch, init_lr=lr)
            ),
            LogMetrics(all_results, data, model_index, epochs, use_pretrained_model),
        ],
        verbose=2,
        initial_epoch=initial_epoch,
    )
    return model, all_results


def main(
    root_dir,
    log_file,
    ensemble_type: str,
    ensemble_size: int,
    epochs: int,
    lr: float,
    source_dataset,
    ood_dataset,
    target_size,
    target_ood_ratio,
    l2_reg,
    nn_depth,
    use_pretrained_model,
    goal_tag,
    run_id: Optional[str],
    model_index: Optional[int],
):
    if run_id or model_index:
        assert run_id is not None
        assert model_index is not None
        assert 0 <= model_index and model_index < ensemble_size

    labeling_scheme = utils.get_labeling_scheme(ensemble_type)

    curr_run = retry(lambda: mlflow.start_run(run_id=run_id, run_name=ensemble_type))

    curr_exp_dir = Path(root_dir) / ensemble_type / curr_run.info.run_id
    curr_exp_dir.mkdir(exist_ok=True, parents=True)

    default_experiment_name = "{}_vs_{}".format(
        utils.pretty_dataset_name(source_dataset),
        utils.pretty_dataset_name(ood_dataset),
    )
    if epochs < 10 and not use_pretrained_model:
        default_experiment_name = "test_" + default_experiment_name

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default_experiment_name)

    retry(lambda: mlflow.set_experiment(experiment_name))

    params = {
        "exp_dir": curr_exp_dir,
        "ensemble_size": ensemble_size,
        "ensemble_type": ensemble_type,
        "hostname": os.uname()[1],
    }
    if log_file is not None:
        params["log_file"] = log_file

    all_results = []

    print(f"run_id: { curr_run.info.run_id }")
    params.update(
        {
            "run_id": curr_run.info.run_id,
            "experiment_id": curr_run.info.experiment_id,
            "experiment_name": experiment_name,
        }
    )

    if goal_tag is not None:
        retry(lambda: mlflow.set_tag("goal", goal_tag))

    if model_index is not None:
        models_to_train = [model_index]
    else:
        models_to_train = list(range(ensemble_size))

    for i in models_to_train:
        gc.collect()
        ckpt_dir = curr_exp_dir / f"model_{i}"

        initial_epoch = 0

        if ckpt_dir.exists():
            print(f"Cleaning checkpoint dir {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        # TODO: Loading a saved model results in NaN loss.
        # subdirs = list(ckpt_dir.iterdir())
        # saved_epochs = []
        # for sd in subdirs:
        #     [_model, epoch] = str(sd.name).split(".")
        #     if _model != "model":
        #         print(f"Unrecognized subdir: {_model}")
        #         continue
        #     if epoch == "best":
        #         continue
        #     saved_epochs.append(int(epoch))
        # if len(saved_epochs) == 0:
        #     print(
        #         f"Could not find any saved model in {ckpt_dir} (of the form model.$epoch)"
        #     )
        #     resumed_model = None
        # else:
        #     initial_epoch = max(saved_epochs)
        #     subdir = f"model.{initial_epoch:02}"
        #     model_dir = ckpt_dir / subdir
        #     assert (model_dir).exists(), f"Expected to find {model_dir}"
        #     print(f"Trying to reuse saved model from {model_dir}.")
        #     resumed_model = tf.keras.models.load_model(model_dir)

        os.mkdir(ckpt_dir)
        resumed_model = None

        start = time.time()
        data = load_data(
            source_dataset=source_dataset,
            ood_dataset=ood_dataset,
            target_size=target_size,
            target_ood_ratio=target_ood_ratio,
            labeling_scheme=labeling_scheme,
            labeling_index=i,
            params=params,
        )
        if i == 0:
            params.update(
                {
                    "train_size": data.train_size,
                    "source_train_size": data.source_train_size,
                    "valid_size": data.valid_size,
                    "target_id_size": data.target_id_size,
                    "target_ood_size": data.target_ood_size,
                }
            )

        model, results = train(
            data=data,
            image_shape=get_image_size(source_dataset),
            epochs=epochs,
            model_index=i,
            l2_reg=l2_reg,
            lr=lr,
            nn_depth=nn_depth,
            ckpt_dir=ckpt_dir,
            params=params,
            use_pretrained_model=use_pretrained_model,
            resumed_model=resumed_model,
            initial_epoch=initial_epoch,
            source_dataset_for_pretrained=source_dataset,
        )

        all_results.append(results)
        print(
            "[{}] Training model {} took {:.2f}s to train. ExpID: {}. RunID: {}.".format(
                ensemble_type,
                i,
                time.time() - start,
                curr_run.info.experiment_id,
                curr_run.info.run_id,
            )
        )

        if i == 0:
            with open(os.path.join(curr_exp_dir, "params.pkl"), "wb") as f:
                pickle.dump(params, f)

    retry(lambda: mlflow.end_run())

    with open(
        os.path.join(root_dir, "all_results_{}.pkl".format(ensemble_type)), "wb"
    ) as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
    )
    parser.add_argument(
        "--gin_file", action="append", help="Relative path to gin config file."
    )
    parser.add_argument("--gin_param", action="append", help="Extra gin parameters.")
    parser.add_argument(
        "--ensemble_type",
        type=str,
        default="vanilla",
        choices=[
            "vanilla",
            "assign_one_label",
            "assign_per_class",
            "uniform_random",
            "assign_per_cluster",
            "binary_classifier",
        ],
        help="Labeling scheme.",
    )
    parser.add_argument("--ensemble_size", type=int, default=5, help="Ensemble size.")
    parser.add_argument(
        "--goal_tag",
        type=str,
        default=None,
        help="If not None, the tag to be set for the run",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to file where stdout and stderr are redirected.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="If training models in parallel, specify which run id to continue running from.",
    )
    parser.add_argument(
        "--model_index",
        type=int,
        default=None,
        help="If training models in parallel, train only this model.",
    )
    args = parser.parse_args()

    print(args.gin_file, args.gin_param)
    # Parses all the gin_files and all the parameters provided when running the script.
    gin.parse_config_files_and_bindings(args.gin_file, args.gin_param)

    if (
        os.uname()[0] != "Darwin"
        and os.uname()[1] != "lo-login-01"
        and len(tf.config.experimental.list_physical_devices("GPU")) == 0
    ):
        raise RuntimeError("No GPU found!!")

    root_dir = os.path.join(
        args.root,
        "{}_vs_{}_{}_{}ep_l2reg{}_depth{}_lr{}{}{}{}".format(
            utils.pretty_dataset_name(gin.query_parameter("data.source_dataset")),
            utils.pretty_dataset_name(gin.query_parameter("data.ood_dataset")),
            gin.query_parameter("train.model_arch"),
            gin.query_parameter("train.epochs"),
            gin.query_parameter("train.l2_reg"),
            gin.query_parameter("train.nn_depth"),
            gin.query_parameter("train.lr"),
            "_target_size{}".format(gin.query_parameter("data.target_size"))
            if gin.query_parameter("data.target_size") != 20000
            else "",
            "_pretraining" if gin.query_parameter("train.use_pretrained_model") else "",
            "_ood_ratio{}".format(gin.query_parameter("data.target_ood_ratio"))
            if gin.query_parameter("data.target_ood_ratio") != 0.5
            else "",
            "_moreckpt" if not gin.query_parameter("train.save_best_ckpt_only") else "",
        ),
    )
    os.makedirs(root_dir, exist_ok=True)

    main(
        root_dir=root_dir,
        log_file=args.log_file,
        ensemble_type=args.ensemble_type,
        ensemble_size=args.ensemble_size,
        epochs=gin.query_parameter("train.epochs"),
        lr=gin.query_parameter("train.lr"),
        source_dataset=gin.query_parameter("data.source_dataset"),
        ood_dataset=gin.query_parameter("data.ood_dataset"),
        target_size=gin.query_parameter("data.target_size"),
        target_ood_ratio=gin.query_parameter("data.target_ood_ratio"),
        l2_reg=gin.query_parameter("train.l2_reg"),
        nn_depth=gin.query_parameter("train.nn_depth"),
        use_pretrained_model=gin.query_parameter("train.use_pretrained_model"),
        goal_tag=args.goal_tag,
        run_id=args.run_id,
        model_index=args.model_index,
    )
