#!/bin/bash
spd="$1"
experiment="$2"
pass_exist="$3"
split_data="$4"
resources_path="$5"

for ms in 2 1
do
  for mz in 10 1 0.1 0.01
  do
    for rt in 10 1 0.1
    do
      path="${resources_path}/$experiment/tsv/mz${mz}/rt${rt}/${spd}spd/ms${ms}"
      if ! [[ -e $path ]] || [ $pass_exist == FALSE ]; then
        # We only do retention time equal or higher than the mz
        if (( $(echo "$rt >= $mz" |bc -l) )); then
          # echo $path
          echo "mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
          echo $path
          bash msml/preprocess/mzdb2tsv.sh $mz $rt $spd $ms $experiment $split_data $resources_path
        else
          echo "$rt < $mz"
        fi
      else
        echo "Already exists: mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
      fi
    done
  done
done

