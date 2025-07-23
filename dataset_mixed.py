#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:45:07 2025

@author: ostocker
"""

import OSToolBox as ost
from tqdm import tqdm
import numpy as np
import multiprocessing
import shutil
import os 


data_path="/home/ostocker/travaux/data/compact/train"
tmp_path_files=ost.getFileBySubstr(data_path,'.ply')
out="/home/ostocker/travaux/data/mixed_real_sim"

index_file="/home/ostocker/travaux/data/compact/Index_selection.npy"
n=500
ind_full=np.load("/home/ostocker/travaux/data/compact/Index_selection.npy",allow_pickle=True)
ind_selected=ind_full[:n]
data_list = [tmp_path_files[i] for i in ind_selected]  
print(len(data_list))
verif=[]



for i,d in enumerate(data_list[:n]):
    
    if "test" in d:
        print("ALERT")
        verif.append(d)
        
    ost.createDir(out+"/train")
    shutil.copyfile(d, out+"/train/"+ost.pathLeafExt(d))
    ost.createDir(out+"/val")
    shutil.copyfile(d, out+"/val/"+ost.pathLeafExt(d))

print(len(verif))




