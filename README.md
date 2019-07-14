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

### Evaluation
The performance of the model was evaluated on DSB2018 data set.

### Notes
UNet code adapted from here: https://github.com/milesial/Pytorch-UNet
