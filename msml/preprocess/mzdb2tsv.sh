#!/bin/bash
START_TIME=$SECONDS
mz_bin="$1"
rt_bin="$2"
spd="$3"
ms_level="$4"
experiment="$5"
split_data="$6"
resources_path="../../$7"  # Need to go back 2 levels to go back to root dir
cd msml/mzdb2tsv || exit
nprocs=$(nproc --all)
if [ "$split_data" == "0" ]; then
  echo "Processing mzdb to tsv using $nprocs processes. Split data: 0"
  if [ "$ms_level" == "1" ]; then
    find "$resources_path"/"$experiment"/mzdb/"$spd"spd/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram_ms1.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
    wait
  elif [ "$ms_level" == "2" ]; then
    find "$resources_path"/"$experiment"/mzdb/"$spd"spd/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
    wait
  fi
  # cd $resources_path/mzdb/"$spd"spd/"$group" || exit
  mkdir -p "$resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/all/"
  for input in *.tsv
  do
      id="${input%.*}"
      echo "Moving $id to $resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/all/"
      mv "$id.tsv" "$resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/all/"
  done
else
  echo "Processing mzdb to tsv using $nprocs processes. Split data: 1"
  declare -a StringArray=('train' 'valid' 'test' )
  for group in "${StringArray[@]}";
  do
    if [ "$ms_level" == "1" ]; then
      find "$resources_path"/"$experiment"/mzdb/"$spd"spd/"$group"/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram_ms1.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
      wait
    elif [ "$ms_level" == "2" ]; then
      find "$resources_path"/"$experiment"/mzdb/"$spd"spd/"$group"/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
      wait
    fi
    # cd $resources_path/mzdb/"$spd"spd/"$group" || exit
    mkdir -p "$resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/"$group"/"
    for input in *.tsv
    do
        id="${input%.*}"
        echo "Moving $id to $resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/"$group"/"
        mv "$id.tsv" "$resources_path/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/ms"$ms_level"/"$group"/"
    done
  done
fi
ELAPSED_TIME=$(($SECONDS - $START_TIME))
eval "echo mzdb2tsv Elapsed time : $(date -ud "@$ELAPSED_TIME" +'$((%s/3600/24)) days %H hr %M min %S sec')"
# cd ../../.. || exit
# python3 msml/make_images.py --run_name="$now" --on=all --remove_zeros_on=all --test_run="$test" --mz_bin="$mz_bin" --rt_bin="$rt_bin" --spd="$spd"
