"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np

import OSToolBox as ost

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class QCSFDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, info=None, **kwargs):
        self.intensity_max=info.intensity_max
        self.dataset_usage=info.dataset_usage
        try: 
            self.mixed=info.mixed
        except:
            self.mixed=False
        super().__init__(ignore_index=ignore_index, **kwargs)


    def get_data_list(self):
        data_path=os.path.join(self.data_root,self.split)
        tmp_path_files=ost.getFileBySubstr(data_path,'.ply')

        if self.dataset_usage is None or self.dataset_usage==1 or self.split!= "train" :
            data_list = tmp_path_files
        else :
            print("=> INFO : dataset_usage detected, only using :", self.dataset_usage*100,"% of",len(tmp_path_files),"founded files")
            ind_full=np.load(ost.getFileBySubstr(self.data_root,'index_selection')[0],allow_pickle=True)
            ind_selected=ind_full[:int(ind_full.shape[0]*self.dataset_usage)]
            data_list = [tmp_path_files[i] for i in ind_selected]  
            # data_list=[tmp_path_files[i] for i in np.arange(int(len(tmp_path_files)*self.dataset_usage))] #for debug

        print("=> USING",len(data_list),"files, from",data_path)
        return data_list
        
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data, fields = ost.readPly(data_path,fields_names=True)
        data=data.astype(np.float32)
        coord = data[:, :3]

        if self.mixed:
            if "als" in ost.pathLeaf(data_path):
                int_norm=self.intensity_max[0]
            else :
                int_norm=self.intensity_max[1]
        else :
            int_norm=self.intensity_max
        strength = data[:, 3].reshape([-1, 1]) / int_norm  # scale intensity to [0, 1]
        
        if "class" in fields:
            segment = data[:, fields.index("class")].astype(np.int32)
        else:
            segment = np.ones((data.shape[0],), dtype=np.int64) * self.ignore_index

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        #print(data_dict)
        return data_dict
