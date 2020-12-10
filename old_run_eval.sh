#!/bin/bash

log_path="output_logs/log_eval_"
# epochs="0_10_20 30_40_50 60_70_80 90_99"
# ensemble_types="vanilla assign_one_label assign_per_class"
# ensemble_types="vanilla assign_one_label assign_per_cluster"
ensemble_types="vanilla assign_one_label"
# ensemble_types="uniform_random"

function run_eval_ensemble {
    root=${1%/}

#    bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#      -R "select[gpu_model0==TeslaV100_SXM2_32GB]" -oo ${log_path}${root} \
#      "python3 eval_ensembles.py --root ${RANDOM_LABELS_PROJECT_ROOT}/${root} --epochs ${epochs}"

#     for epoch_subset_ in ${epochs}; do
#       epoch_subset=`echo ${epoch_subset_} | tr "_" " "`
#       for ensemble_type in ${ensemble_types}; do
#           exp_dir=${RANDOM_LABELS_PROJECT_ROOT}/${root}/${ensemble_type}
#           log_file=${log_path}${root}_${ensemble_type}_${epoch_subset_}
# 
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} \
#             --epochs ${epoch_subset}"
#           echo "Started eval with logs at "${log_file}
#       done
#     done 

    for ensemble_type in ${ensemble_types}; do
        exp_dir=${RANDOM_LABELS_PROJECT_ROOT}/${root}/${ensemble_type}
        log_file=${log_path}${root}_${ensemble_type}

#         bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#           -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#           "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs -1"

#         bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#           -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#           "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs -1 --use_holdout_target=True"

        store_predictions=${exp_dir}/../../../model_predictions/${root}_${ensemble_type}.pkl
        bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
          -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
          "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 10 50 99 --store_predictions ${store_predictions}"

#         ckpt_file=${exp_dir}/model_0/model.
#         if [ -d ${ckpt_file}01 ]; then
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 0 1 2 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 3 4 5 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 6 7 8 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 9"
#         elif [ -d ${ckpt_file}02 ]; then
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 0 2 4 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 6 8 10 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 12 14 16 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 18 20 22 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 24 26 28"
#         elif [ -d ${ckpt_file}05 ]; then
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 0 5 10 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 15 20 25 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 29"
#         else
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 0 10 20 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 30 40 50 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 60 70 80 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 90"
# #             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 90 99"
#           bsub -W 4:00 -n 1 -R "rusage[mem=8192,ngpus_excl_p=1]" \
#             -R "select[gpu_mtotal0>=10240]" -oo ${log_file} \
#             "python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 100 110 120 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 130 140 150 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 160 170 180 && \
#             python3 eval_ensembles.py --exp_dir ${exp_dir} --epochs 190 199"
#         fi
        echo "Started eval with logs at "${log_file}
    done
}

for root in $1; do
    run_eval_ensemble ${root}
done
