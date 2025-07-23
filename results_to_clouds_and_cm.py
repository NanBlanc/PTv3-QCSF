#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:04:14 2025

@author: ostocker
"""

import OSToolBox as ost
from tqdm import tqdm
import numpy as np
import multiprocessing


#FOR PTV3
split="test"

# exp_list=["run_int_notrans","run_noint_trans","run_noint_notrans","run_int_trans"]
# exp_list=["run_du10","run_du20","run_du5","run_du50"]
# exp_list=["run_du1"]
# base="/home/ostocker/travaux/data/compact"
# outdir_name="res_simulation"

exp_list=["run_int"]
base="/home/ostocker/travaux/data/ALSlike_p12train"
outdir_name="res_p3"

# base="/home/ostocker/travaux/data/ALSlike_full"
# outdir_name="res_real"

for exp in exp_list:
    print("Processing",exp)
    outdir=ost.createDir("/home/ostocker/travaux/ptv3/Pointcept/exp/qcsf/"+exp+"/"+outdir_name)
    expdir="/home/ostocker/travaux/ptv3/Pointcept/exp/qcsf/"+exp+"/result"
    
    cm=ost.ConfusionMatrix(4, ["sol","ab","pm","vb"],99)
    predlist=ost.getFileBySubstr(expdir, "npy")
    for file in tqdm(predlist):
        tag=file.split("/")[-1].split(".")[0]
        pred=np.load(file).reshape([-1,1])
        try :
            # data,f=ost.readPly("/home/ostocker/travaux/data/QCSF_sample/"+split+"/sequences/td_"+tag.split("_")[0]+"/ab_"+tag.split("_")[1]+"/"+tag+".ply",fields_names=True)
            # data,f=ost.readPly(base+"/"+split+"/td_"+tag.split("_")[0]+"/ab_"+tag.split("_")[1]+"/"+tag+".ply",fields_names=True)
            data,f=ost.readPly(ost.getFileBySubstr(base, ost.pathLeaf(tag))[0],fields_names=True)
        except :
            continue
        cm.add_batch(data[:,4], pred)
        f.append("pred")
        ost.writePly(outdir+"/"+tag+".ply", np.hstack((data,pred)), f)
    cm.printPerf(ost.createDir(outdir+"/metrics"))