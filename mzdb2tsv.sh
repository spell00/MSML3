#!/bin/bash
spd="$1"
# experiment="$2"
pass_exist="$2"
split_data="$3"
resources_path="$4"
for experiment in 'B1-02-02-2024' 'B2-02-21-2024' 'B3-02-29-2024' 'B4-03-01-2024' 'B5-03-13-2024' 'B6-03-29-2024' 'B7-04-03-2024' 'B8-04-15-2024' 'B9-04-22-2024' 'B10-05-03-2024' 'B11-05-24-2024' 'B12-05-31-2024' 'B13-06-05-2024' 'B14-06-10-2024' 'B15-06-29-2024' 'BPatients-03-14-2025'
do
  for ms in 2
  do
    for mz in 10 2
    do
      for rt in 320
      do
        path="${resources_path}/$experiment/tsv/mz${mz}/rt${rt}/${spd}spd/ms${ms}"
        if ! [[ -e $path ]] || [ $pass_exist == FALSE ]; then
          echo "mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
          echo $path
          bash msml/preprocess/mzdb2tsv.sh $mz $rt $spd $ms $experiment $split_data $resources_path
        else
          echo "Already exists: mz${mz} rt${rt} ${spd}spd ms${ms} ${experiment}"
        fi
      done
    done
  done
done
