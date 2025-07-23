#!/bin/sh

exp_list=("du50" "du20" "du10" "du5")

for exp in "${exp_list[@]}"; do
    sh scripts/train.sh -g 1 -d qcsf -c semseg-ptv3-$exp -n run_$exp
    sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-$exp -n run_$exp
done