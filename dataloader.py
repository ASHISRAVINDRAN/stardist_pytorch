# SPDX-FileCopyrightText: 2022 Ashis Ravindran <ashis(dot)r91(at)gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
import numpy as np
from stardist import star_dist,edt_prob
from csbdeep.utils import normalize


class DSB2018Dataset(Dataset):

    def __init__(self, root_dir, n_rays,max_dist=None,transform=None,target_transform= None):
       
        self.raw_files = os.listdir(os.path.join(root_dir,'images'))
        self.target_files = os.listdir( os.path.join(root_dir,'masks'))
        self.raw_files.sort()
        self.target_files.sort()
        self.root_dir = root_dir
        self.transform = transform
        self.n_rays = n_rays
        self.target_transform = target_transform
        self.max_dist = max_dist

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):
        assert self.raw_files[idx] == self.target_files[idx]
        img_name = os.path.join(self.root_dir,'images',self.raw_files[idx])
        image = io.imread(img_name)
        image = normalize(image,1,99.8,axis = (0,1))
        image = np.expand_dims(image,0)
        target_name = os.path.join(self.root_dir,'masks',self.target_files[idx])
        target = io.imread(target_name)
        distances = star_dist(target,self.n_rays,opencl=False)
        if self.max_dist:
            distances[distances>self.max_dist] = self.max_dist
        distances = np.transpose(distances,(2,0,1))
        obj_probabilities = edt_prob(target)
        obj_probabilities = np.expand_dims(obj_probabilities,0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            distances = self.target_transform(distances)
            obj_probabilities = self.target_transform(obj_probabilities)
        return image,obj_probabilities,distances
   
def getDataLoaders(n_rays,max_dist,root_dir):
    trainset = DSB2018Dataset(root_dir=root_dir+'/train/',n_rays=n_rays,max_dist=max_dist)
    testset = DSB2018Dataset(root_dir=root_dir+'/test/',n_rays=n_rays,max_dist=max_dist)
    
    trainloader = DataLoader(trainset, batch_size=1,shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)
    return trainloader,testloader