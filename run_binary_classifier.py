#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import lib_jobs


def main():
    parser = argparse.ArgumentParser("Run binary classification experiment.")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    args = parser.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    os.environ["TFDS_DATA_DIR"] = str(scratch / ".datasets")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    cli_args = [
        ("root", scratch / args.experiment_name),
        ("ensemble_size", 1),
        ("ensemble_type", "binary_classifier"),
        ("gin_file", "configs/resnet.gin"),
        ("gin_file", "configs/binary_classifier.gin"),
    ]
    gin_args = [
        ("data.source_dataset", args.id_dataset),
        ("data.ood_dataset", args.ood_dataset),
    ]

    nhours = 4
    memory_per_cpu = 4096

    tesla = False

    if "imagenet" in args.id_dataset:
        nhours = 24
        memory_per_cpu = 30_000

    if "nih" in args.id_dataset:
        nhours = 24
        memory_per_cpu = 30_000
        gin_args.append(("data.batch_size", 32))
        tesla = True

    lib_jobs.launch_bsub(
        nhours,
        "main.py",
        cli_args,
        gin_args,
        memory_per_cpu=memory_per_cpu,
        tesla=tesla,
    )


if __name__ == "__main__":
    main()
