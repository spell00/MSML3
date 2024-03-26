#!/bin/bash
echo $(which python)
mz_bin="$1"
rt_bin="$2"
spd="$3"
experiment="$4"
# group="$5"
if [ $# -ge 1 ]
then
    if [ "$6" == "test" ]
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
find ../../../../resources/"$experiment"/mzdb/"$spd"spd/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram_ms1.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
wait
# cd ../../../../resources/mzdb/"$spd"spd/"$group" || exit
mkdir -p "../../../../resources/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/"
for input in *.tsv
do
    id="${input%.*}"
    echo "Moving $id data"
    mv "$id.tsv" "../../../../resources/"$experiment"/tsv/mz$mz_bin/rt$rt_bin/"$spd"spd/"
done

# cd ../../../../.. || exit
# python3 msml/make_images.py --run_name="$now" --on=all --remove_zeros_on=all --test_run="$test" --mz_bin="$mz_bin" --rt_bin="$rt_bin" --spd="$spd"
