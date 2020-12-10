import argparse
from pathlib import Path
import itertools
import subprocess
import mlflow
import os
import glob
from typing import Tuple, List
import reproduction.lib_data as lib_data


eval_accuracy_on_pretrained_model = True


def do_eval(
    curr_run,
    log_dir,
    ensemble_size,
    use_holdout_target,
    store_predictions_path,
    target_size,
    target_ood_ratio,
):
    exp_dir = curr_run.data.params["exp_dir"]
    ckpt_idx_list = [
        f.rsplit(".", 1)[1] for f in glob.glob(os.path.join(exp_dir, "model_0/model.*"))
    ]
    if "best" in ckpt_idx_list:
        epoch_chunk_list = ["-1"]
    else:
        epoch_chunk_list = list(divide_chunks([str(int(x)) for x in ckpt_idx_list], 3))
        epoch_chunk_list = [" ".join(chunk) for chunk in epoch_chunk_list]

    memory = 8096

    if curr_run.data.params["source_dataset"] in ["celeb_a_id", "pcam"]:
        memory = 20_000

    for epoch_chunk in ["-1"]:
        params = curr_run.data.params
        if "log_file" not in params:
            print(f"Could not find key log_file in {params}.")
            continue

        log_filename = "log_eval_{}".format(
            Path(params["log_file"]).stem.split("_", 1)[1]
        )
        log_file = os.path.join(log_dir, log_filename)
        bsub_args = [
            "bsub",
            "-W",
            "4:00",
            "-n",
            str(4),
            "-R",
            f'"rusage[mem={memory},ngpus_excl_p=1]"',
            "-R",
            '"select[gpu_mtotal0>=10240]"',
            "-o",
            log_file,
        ]

        python_args = [
            "python3",
            "eval_ensembles.py",
            "--exp_dir",
            exp_dir,
            "--epochs",
            epoch_chunk,
            "--ensemble_size",
            str(ensemble_size),
        ]
        if target_size is not None:
            python_args += ["--target_size", str(target_size)]
        if target_ood_ratio is not None:
            python_args += ["--target_ood_ratio", str(target_ood_ratio)]
        if use_holdout_target:
            python_args.append("--use_holdout_target=True")
        if store_predictions_path is not None:
            exp_dir_root = os.path.normpath(exp_dir).rsplit("/", 2)[1]
            full_store_predictions_path = os.path.join(
                store_predictions_path,
                "{}_{}_{}.pkl".format(
                    exp_dir_root,
                    curr_run.data.params.get("ensemble_type"),
                    epoch_chunk.replace(" ", "_"),
                ),
            )
            python_args += ["--store_predictions_path", full_store_predictions_path]

        if eval_accuracy_on_pretrained_model:
            python_args.append("--eval_accuracy_on_pretrained_model")

        python_command = " ".join(python_args)
        bsub_command = " ".join(bsub_args + [python_command])
        if os.uname()[0] != "Darwin":
            stdout = subprocess.check_output(bsub_command, env=os.environ, shell=True)
            print(str(stdout))
        else:
            subprocess.check_output(python_args, env=os.environ)
        print("Logging to", log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_holdout_target",
        dest="use_holdout_target",
        action="store_true",
        help="If true, it uses a holdout target set for evaluation.",
    )
    parser.add_argument("--ensemble_size", type=int, default=5, help="Ensemble size")
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target size to be used for evaluation. Only for vanilla ensembles. For RETO we use the value in the "
        "config.",
    )
    parser.add_argument(
        "--target_ood_ratio",
        type=float,
        default=None,
        help="Target OOD ratio to be used for evaluation. Only for vanilla ensembles. For RETO we use the value in "
        "the config.",
    )
    parser.add_argument(
        "--store_predictions_path",
        type=str,
        default=None,
        help="If set, it doesn't compute AUROCs, but only stores the predictions to the location indicated.",
    )
    parser.add_argument(
        "-qp",
        "--query_parameter",
        type=str,
        action="append",
        help="Must be of the form k=v. Can be specified multiple times.",
    )
    parser.set_defaults(use_holdout_target=False)
    args = parser.parse_args()

    log_dir = Path(os.environ["RANDOM_LABELS_LOGS_DIR"])

    all_active_exps = filter(
        lambda exp: exp.lifecycle_stage == "active", mlflow_client.list_experiments()
    )
    experiments = {exp.experiment_id: exp for exp in all_active_exps}

    def format_as_query(k: str, v: str):
        return f'{k} = "{v}"'

    def eval_matching(query_parameters: List[Tuple[str, str]]):
        query = " and ".join([format_as_query(*t) for t in query_parameters])

        query_result = mlflow.search_runs(
            experiments.keys(),
            filter_string=query,
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        )
        for run_id in query_result["run_id"]:
            curr_run = mlflow_client.get_run(run_id)

            id_dataset = curr_run.data.params.get("source_dataset")
            ood_dataset = curr_run.data.params.get("ood_dataset")
            if id_dataset is None or ood_dataset is None:
                continue

            if (
                args.target_size is not None or args.target_ood_ratio is not None
            ) and curr_run.data.params.get("ensemble_type") != "vanilla":
                continue

            do_eval(
                curr_run,
                log_dir,
                args.ensemble_size,
                args.use_holdout_target,
                args.store_predictions_path,
                target_size=args.target_size,
                target_ood_ratio=args.target_ood_ratio,
            )

    query_parameters = [("attribute.status", "FINISHED")]

    for qp in args.query_parameter:
        assert "=" in qp
        [k, v] = qp.split("=")
        query_parameters.append((k, v))
    eval_matching(query_parameters)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    main()
