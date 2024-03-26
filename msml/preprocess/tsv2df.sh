#!/bin/bash
START_TIME=$SECONDS

mz_bin="$1"
rt_bin="$2"
mz_bin_post="$3"
rt_bin_post="$4"
spd="$5"
ms_level="$6"
experiment="$7"
split_data="$8"
feature_selection="$9"
run_name="${10}"
test="${11}"
resources_path="../../${12}"

if [ "$spd" == "" ]; then
  spd=200
fi
if [ "$ms_level" == "" ]; then
  ms_level=2
fi
if [ "$experiment" == "" ]; then
  experiment="old_data"
fi
if [ "$split_data" == "" ]; then
  split_data=0
fi
if [ "$feature_selection" == "" ]; then
  feature_selection="mutual_info_classif"
fi
if [ "$run_name" == "" ]; then
  run_name="eco,sag,efa,kpn,blk,pool"
fi
if [ "$test" == "" ]; then
  test=0
fi
# if [ "$berm" == "" ]; then
#   berm='none'
# fi

if [ "$split_data" == 1 ]; then
  if [ "$ms_level" == 1 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3.8 msml/preprocess/make_tensors_ms1_split.py --resources_path=$resources_path --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
  fi
  if [ "$ms_level" == 2 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3.8 msml/preprocess/make_tensors_ms2_split.py --resources_path=$resources_path --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
  fi
fi
if [ "$split_data" == 0 ]; then
  if [ "$ms_level" == 1 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3.8 msml/preprocess/make_tensors_ms1.py --resources_path=$resources_path --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
  fi
  if [ "$ms_level" == 2 ]; then
    echo "$experiment" --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
    python3.8 msml/preprocess/make_tensors_ms2.py --resources_path=$resources_path --experiment=$experiment --test_run=$test --run_name=$run_name --feature_selection=$feature_selection --mz_bin_post=$mz_bin_post --rt_bin_post=$rt_bin_post --mz_bin=$mz_bin --rt_bin=$rt_bin
  fi
fi

