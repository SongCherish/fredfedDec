#!/bin/bash
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8


source ~/.bash_profile
conda activate fl-bench
conda env list
#methods=("fredfedrep" "fedrep" "cfl" "fedavg" "fedper" "local" "moon" "fedprox"   "pfedsim" "perfedavg" )
#methods=("fredfedrep" "fedrep" "cfl" "fedavg" "fedper" "local" "moon" "fedprox" "fedbn"  "pfedsim" "perfedavg" "fedgen" "pfedhn" "knnper" "pfedla")
#methods=( "pfedfda" "fedas" "flute" "pefll" "floco" "fedala")

## all method
#methods=(  "flute" "floco" "fedala" "fedavg" "fedprox" "moon" "fedgen" "fedbn" "cfl" "knnper" )
methods=( "fredfedrep" "fedrep")
seeds=(21 43 44 45 46 75 85 95 16 111 222 342 546 8569 334 255 8743 998)

config="fredFedRep-fmnist"
dataset="cifar10"
model="lenet5"
# shellcheck disable=SC2068
echo ${methods[@]}
echo $config
# shellcheck disable=SC2068
# shellcheck disable=SC2154
for seed in "${seeds[@]}"
do
  for method in "${methods[@]}"

  do
    now=$(date +"%Y-%m-%d-%H-%M-%S")
    sh_log_dir="./sh_result/seed-${seed}/${dataset}/${model}/${method}"
    sh_log_path="${sh_log_dir}/${now}.log"

    if [ ! -d "$sh_log_dir" ];then
      mkdir -p $sh_log_dir
      echo "创建文件夹 $sh_log_dir"
    fi
    echo "python main.py --config-name $config method=$method dataset.name=$dataset model.name=$model  common.seed=$seed |& tee $sh_log_path 2>&1"
    python -u main.py --config-name $config method=$method dataset.name=$dataset model.name=$model common.seed=$seed |& tee -a $sh_log_path 2>&1
  done
done

