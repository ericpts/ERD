import lib_data
import mlflow
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    args = parser.parse_args()

    lib_data.setup_mlflow()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        os.environ["MLFLOW_RUN_ID"] = run_id
        mlflow.log_param("run_id", run_id)

        mlflow.log_params(
            {
                "id_dataset": args.id_dataset,
                "ood_dataset": args.ood_dataset,
            }
        )


if __name__ == "__main__":
    main()
