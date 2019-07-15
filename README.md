# Cell Detection with Star-convex Polygons: PyTorch implementation
This is the pytorch implementation of the paper:

Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.
Refer the [StarDist](https://github.com/mpicbg-csbd/stardist) repo before you continue.

### Installation/Requirements
All the necessary packages to generate ground truth can be found in the *StarDist* official repository.
The code uses the deep learning library: PyTorch, instead of Keras/Tensorflow.  
Optional: [TensorboardX](https://github.com/lanpa/tensorboardX)

### Documentation
- `main.py`: Start file for training the network. Specify path to dataset, tensorboard log directory etc here.
- `dataloader.py`: Simple basic pytorch dataloader for DSB2018.
- `distance_loss.py` : Loss function class, as defined in the paper.
- `train.py`: Trainer file. Code to load and save checkpoints in accordance with loss and learning rate.
- `load_save_model.py`: Boilerplate assisting code for the Trainer class.Used for loading and saving models.
- `predict.py`: Script for test set prediction. Specify path to test set and pretrained weights here. User can also change the probability threshold for NMS.
- `metric.py`: Unofficial evaluation metric script for calculating average precision of IoUs.

### Evaluation
The performance of the model was evaluated on DSB2018 data set.

### Notes
UNet code adapted from: https://github.com/milesial/Pytorch-UNet
