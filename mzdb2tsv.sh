#!/bin/bash
spd="$1"
# experiment="$2"
pass_exist="$2"
split_data="$3"
resources_path="$4"
for experiment in 'B15-06-29-2024'
do
  for ms in 2
  do
    for mz in 10 0.1
    do
      for rt in 10
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
done

