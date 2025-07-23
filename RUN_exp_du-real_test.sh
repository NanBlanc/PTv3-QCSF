#!/bin/sh

rename_if_dir_exist(){
    local counter=0
    local dir_to_check="$1" # Take the directory name as an argument

    while [ -d "${dir_to_check}_$counter" ]; do
        counter=$((counter + 1))
    done

    if [ -d "$dir_to_check" ]; then
        mv "$dir_to_check" "${dir_to_check}_$counter"
        echo "Renamed '$dir_to_check' to '${dir_to_check}_$counter'"
    else
        echo "Directory '$dir_to_check' is free."
    fi
}

root="/home/ostocker/travaux/ptv3/Pointcept/exp/qcsf"
exp_list=("du50" "du20" "du10" "du5")

for exp in "${exp_list[@]}"; do
    result_dir="$root/run_$exp/result"
    rename_if_dir_exist "$result_dir" 
    sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-$exp-rt -n run_$exp
done

#test single
#sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-int_trans-rt -n run_int_trans
