#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import numpy as np
from utils import pretty_dataset_name
import itertools
import reproduction.lib_data as lib_data
import mlflow
import lib_jobs

default_epochs = 100
ensemble_size = 5

bw_model_arch_list = ["mlp"]
small_image_arch_list = ["resnet"]
big_image_arch_list = ["densenet"]

# ensemble_type_list = ["vanilla", "assign_one_label"]
ensemble_type_list = ["assign_one_label"]
# ensemble_type_list = ["vanilla"]
use_pretrained_model_list = [False]
# use_pretrained_model_list = [True, False]
target_size_list = [20000]
# target_size_list = [40, 100, 200, 600, 1000, 2000, 10000, 20000]
# target_size_list = [10000]
target_ood_ratio_list = [0.5]
# target_ood_ratio_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
save_best_ckpt_only = True


def main():
    parser = argparse.ArgumentParser("Launches a run that trains an ensemble.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "-wp",
        "--with_param",
        type=str,
        action="append",
        default=[],
        help="Optional repeated argument of the form k=[v], "
        "will be included in the cartesian product of parameters, "
        "using k as the gin parameter name. "
        "Example usage: --with_param data.batch_size=[32,64,128]",
    )
    parser.add_argument(
        "--goal_tag",
        type=str,
        default="",
        required=False,
        help="Optional goal tag to set.",
    )

    args = parser.parse_args()

    lib_data.setup_mlflow()

    if args.experiment_name is None:
        args.experiment_name = "{}_vs_{}".format(
            pretty_dataset_name(args.id_dataset), pretty_dataset_name(args.ood_dataset)
        )

    print(args.experiment_name)
    mlflow.set_experiment(args.experiment_name)

    log_dir = Path(os.environ["RANDOM_LABELS_LOGS_DIR"])

    if os.environ.get("TFDS_DATA_DIR") is None:
        scratch = Path(os.environ["SCRATCH"])
        os.environ["TFDS_DATA_DIR"] = str(scratch / ".datasets")

    global ensemble_size

    if os.environ.get("MLFLOW_EXPERIMENT_NAME") is None:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    memory_per_cpu = 4096
    if "mnist" in args.id_dataset or "mnist" in args.ood_dataset:
        curr_model_arch_list = bw_model_arch_list
    elif (
        lib_data.lib_medical_ood.is_medical(args.id_dataset)
        or lib_data.lib_celeb_a.is_celeb_a(args.id_dataset)
        or lib_data.lib_waterbirds.is_waterbirds(args.id_dataset)
    ):
        memory_per_cpu = 30_000
        curr_model_arch_list = big_image_arch_list
        ensemble_size = 2
    else:
        curr_model_arch_list = small_image_arch_list

    if "+" in args.id_dataset:
        ensemble_size = lib_data.get_num_classes(args.id_dataset)

    global target_size_list
    if "imagenet" in args.id_dataset:
        target_size_list = [100_000]

    params = {}
    for param_string in args.with_param:
        [k, v] = param_string.split("=")
        v = eval(v)  # Risky but what you gonna do about it.
        assert type(v) == type([])
        params[k] = v

    for config in itertools.product(
        curr_model_arch_list,
        ensemble_type_list,
        use_pretrained_model_list,
        target_size_list,
        target_ood_ratio_list,
        *params.values(),
    ):
        (
            model_arch,
            ensemble_type,
            use_pretrained_model,
            target_size,
            target_ood_ratio,
            *gin_args_tuple,
        ) = config

        gin_args_from_cli = list(zip(params.keys(), gin_args_tuple))

        epochs = 10 if use_pretrained_model else default_epochs

        if int(target_size * target_ood_ratio) == 0:
            print("Skipping", target_size, target_ood_ratio)
            continue

        if "imagenet" in args.id_dataset:
            nhours = 120
            memory_per_cpu = 20_000
        elif epochs > 30 and model_arch in ["resnet", "wideresnet", "densenet"]:
            nhours = 24
        else:
            nhours = 4

        base_cli_args = [
            ("root", os.environ["RANDOM_LABELS_PROJECT_ROOT"]),
            ("ensemble_type", ensemble_type),
            ("ensemble_size", ensemble_size),
            ("gin_file", "configs/{}.gin".format(model_arch)),
        ]

        if args.goal_tag != "":
            base_cli_args.append(("goal_tag", args.goal_tag))

        gin_args = [
            ("data.source_dataset", args.id_dataset),
            ("data.ood_dataset", args.ood_dataset),
            ("data.target_size", target_size),
            ("data.target_ood_ratio", target_ood_ratio),
            ("train.model_arch", model_arch),
            ("train.epochs", epochs),
            ("train.use_pretrained_model", use_pretrained_model),
            ("train.save_best_ckpt_only", save_best_ckpt_only),
            *gin_args_from_cli,
        ]

        log_file = os.path.join(
            log_dir,
            "log_{}_vs_{}_{}_{}ep_{}_target_size{}{}_{}".format(
                pretty_dataset_name(args.id_dataset),
                pretty_dataset_name(args.ood_dataset),
                model_arch,
                epochs,
                ensemble_type,
                target_size,
                "_pretrained" if use_pretrained_model else "",
                np.random.randint(1e9),
            ),
        )
        print("Logging to", log_file)

        if use_pretrained_model:
            gin_args.append(("logging.log_every_n_epochs", 1))
            if model_arch in ["resnet", "wideresnet"]:
                gin_args.append(("train.lr", 0.001))
            elif model_arch == "vgg":
                gin_args.append(("train.lr", 0.0001))
            elif model_arch == "densenet":
                gin_args.append(("train.lr", 0.001 * 0.01))
            else:
                assert False

        if model_arch in big_image_arch_list:
            gin_args.append(("data.batch_size", 32))

        if args.parallel:
            print("Training models in parallel.")
            run = mlflow.start_run()
            run_id = run.info.run_id
            print(f"Using run_id {run_id}")
            for model_index in range(ensemble_size):
                cur_log_file = "{}_model_index_{}".format(log_file, model_index)
                cli_args = base_cli_args + [
                    ("log_file", cur_log_file),
                    ("model_index", model_index),
                    ("run_id", run_id),
                ]
                lib_jobs.launch_bsub(
                    nhours,
                    "main.py",
                    cli_args,
                    gin_args,
                    log_file=cur_log_file,
                    memory_per_cpu=memory_per_cpu,
                )
            mlflow.end_run()

        else:
            cli_args = base_cli_args + [("log_file", log_file)]
            if os.uname()[0] != "Darwin":
                lib_jobs.launch_bsub(
                    nhours,
                    "main.py",
                    cli_args,
                    gin_args,
                    log_file=log_file,
                    memory_per_cpu=memory_per_cpu,
                )
            else:
                lib_jobs.launch_local("main.py", cli_args, gin_args)


def list2cmd(l):
    return " ".join(l)


def add_arg(args, k, v):
    args.append(f"--{k}")
    args.append(str(v))


def add_gin_arg(args, k, v):
    if isinstance(v, str):
        v = '\\"{}\\"'.format(v)
    args.append("--gin_param")
    args.append(f'"{k}={v}"')


if __name__ == "__main__":
    main()
