# SPDX-FileCopyrightText: 2022 Ashis Ravindran <ashis(dot)r91(at)gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

﻿import os
print('Working dir',os.getcwd())
from load_save_model import save_model
from train import Trainer
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unet import UNetStar as UNet
from distance_loss import MyL1BCELoss
import dataloader

IN_CHANNELS = 1
LOAD_CHECKPOINT = False
DSB2018_PATH = '<Specify path to DSB2018 dataset directory>'
LOG_DIRECTORY = '<Specify path to tensorboard log directory>'
N_RAYS = 32

Trainloader,Testloader = dataloader.getDataLoaders(N_RAYS,max_dist=65,root_dir= DSB2018_PATH)
TARGET_LABELS = N_RAYS
model = UNet(IN_CHANNELS,TARGET_LABELS)

model_name='UNet2D'
print('model='+model_name)
dataset='DSB2018'
print('dataset='+dataset)
train_mode='StarDist'
print('No.of rays',N_RAYS)

init_lr=1e-4
kwargs={}
additional_notes= 'Star Distance training with distances capped to 65.Rays'+str(N_RAYS)+'.\
Probability and Distances output Final activation ReLU and Sigmoid for distances.\
Corrected Loss to Loss = L1loss*Prob+BCEloss.'
kwargs['additional_notes'] = additional_notes
SAVE_PATH = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/'
kwargs['save_path'] = SAVE_PATH
RESULTS_DIRECTORY = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/plots/'
loss = MyL1BCELoss()
trainer = Trainer(loss, None, LOG_DIRECTORY, validate_every= 2)

if LOAD_CHECKPOINT:
    weights_path =SAVE_PATH+model_name+'_'+train_mode+'_'+dataset+'.t7'
    path_checkpoint = os.getcwd()+'/CHECKPOINT/checkpoint_'+model_name+'_'+train_mode+'_'+dataset+'/CHECKPOINT.t7'
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print('Loaded saved weights')
    elif os.path.isfile(path_checkpoint):
        model.load_state_dict(torch.load(path_checkpoint))
        print('Loaded checkpoint')
    else:
        print('Couldnt load checkpoint')

optimizer=torch.optim.Adam(model.parameters(), lr=init_lr,betas=(0.9, 0.999), eps=1e-08,weight_decay=5e-5)
scheduler=ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True,patience=6,eps=1e-8,threshold=1e-20)
print ('Starting Training')
trainloss_to_file,testloss_to_file,trainMetric_to_file,testMetric_to_file,Parameters= trainer.Train(model,optimizer,
                                                                                    Trainloader,Testloader,epochs=None,Train_mode=train_mode,
                                                                                  Model_name=model_name,
                                                                                  Dataset=dataset,scheduler=scheduler)
print('Saving model...')
save_model(model,trainMetric_to_file,testMetric_to_file,trainloss_to_file,testloss_to_file,Parameters,model_name,train_mode,dataset,plot=False,**kwargs)
