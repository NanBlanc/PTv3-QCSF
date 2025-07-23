#!/bin/sh
sh scripts/train.sh -g 1 -d qcsf -c semseg-ptv3-int_trans -n run_int_trans
sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-int_trans -n run_int_trans

sh scripts/train.sh -g 1 -d qcsf -c semseg-ptv3-int_notrans -n run_int_notrans
sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-int_notrans -n run_int_notrans

sh scripts/train.sh -g 1 -d qcsf -c semseg-ptv3-noint_trans -n run_noint_trans
sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-noint_trans -n run_noint_trans

sh scripts/train.sh -g 1 -d qcsf -c semseg-ptv3-noint_notrans -n run_noint_notrans
sh scripts/test.sh -g 1 -d qcsf -c semseg-ptv3-noint_notrans -n run_noint_notrans