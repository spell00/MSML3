#!/bin/bash
echo $(which python)
mz_bin="$1"
rt_bin="$2"
spd="$3"
group="$4"
if [ $# -ge 1 ]
then
    if [ "$5" == "test" ]
    then
        now='test'
        test=1
    else
        now=$(date +"%Y%m%d_%H%M")
        test=0
    fi
else
    now=$(date +"%Y%m%d_%H%M")
    test=0
fi

cd msml/mzdb2tsv || exit
nprocs=$(nproc --all)
echo "Processing mzdb to tsv using $nprocs processes"
find ../../../../resources/20220706_Data_ML02/mzdb/"$spd"spd/"$group"/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
wait
# cd ../../../../resources/mzdb/"$spd"spd/"$group" || exit
mkdir -p "../../../../resources/20220706_Data_ML02/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/"$group"/"
for input in *.tsv
do
    id="${input%.*}"
    echo "Moving $id data"
    mv "$id.tsv" "../../../../resources/20220706_Data_ML02/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/"$group"/"
done

# cd ../../../../.. || exit
# python3 msml/make_images.py --run_name="$now" --on=all --remove_zeros_on=all --test_run="$test" --mz_bin="$mz_bin" --rt_bin="$rt_bin" --spd="$spd"
