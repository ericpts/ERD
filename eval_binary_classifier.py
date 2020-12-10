import reproduction.lib_data as lib_data
import subprocess
import os
import mlflow
import mlflow.tracking
import argparse
import tensorflow as tf
from pathlib import Path
import pickle
import eval_ensembles
import lib_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    lib_data.setup_mlflow()

    mlflow.set_experiment(args.experiment_name)

    exp = mlflow.get_experiment_by_name(args.experiment_name)

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(exp.experiment_id)
    print(f"got {len(runs)} runs")
    for r in runs:
        run_id = r.info.run_id
        exp_dir = Path(r.data.params["exp_dir"])
        assert exp_dir.exists(), f"Could not find exp dir {exp_dir}"

        params_path = exp_dir / "params.pkl"

        if not params_path.exists():
            print(f"Could not find params {params_path}")
            continue

        with params_path.open("rb") as f:
            params = pickle.load(f)

        if params["source_dataset"] != "imagenet_resized/32x32":
            continue

        print(params)
        print(f"Processing {exp_dir}")

        if params["run_id"] != run_id:
            print(f"Fixing run_id of {params_path}")
            params["run_id"] = run_id

            params_path.replace(exp_dir / "old_params.pkl")

            with params_path.open("wb") as f:
                pickle.dump(params, f)

        cli_args = [
            ("exp_dir", exp_dir),
            ("epochs", -1),
            ("ensemble_size", 1),
        ]

        print(f"Launching run {run_id}")
        lib_jobs.launch_bsub(
            4, "eval_ensembles.py", cli_args, [], job_name=f"eval_run_{run_id}"
        )


if __name__ == "__main__":
    main()
