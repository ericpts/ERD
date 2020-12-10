#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser("Run ID vs OOD experiment.")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    args = parser.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    os.environ["TFDS_DATA_DIR"] = str(scratch / ".datasets")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    subprocess_args = ["python3", "main.py"]

    def add_arg(k, v):
        subprocess_args.append(f"--{k}")
        subprocess_args.append(str(v))

    def add_gin_arg(k, v):
        if isinstance(v, str):
            v = '"{}"'.format(v)
        subprocess_args.append("--gin_param")
        subprocess_args.append(f"{k}={v}")

    add_arg("root", scratch / args.experiment_name)
    add_arg("ensemble_size", 1)
    add_arg("ensemble_type", "binary_classifier")
    add_arg("gin_file", "configs/vgg.gin")
    add_arg("gin_file", "configs/id_vs_ood.gin")
    add_gin_arg("data.source_dataset", args.id_dataset)
    add_gin_arg("data.ood_dataset", args.ood_dataset)

    if "nih" in args.id_dataset:
        add_gin_arg("data.batch_size", 32)

    subprocess.check_call(subprocess_args, env=os.environ)


if __name__ == "__main__":
    main()
