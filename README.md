# RestoreGAN
A pytorch implementation of RestoreGAN modified to be used to dejitter 2D images taken at the P06 experiment at PETRA III

## Setup

### Prerequisits

This implementation was tested using a NVIDEA GPU + CUDA CuDNN. While this implementation should work on a CPU with no modifications to the sourcecode, this is as of yet untested.

### Getting Started

Install the required dependancies

~~~ bash
pip3 install torch torchvision torchaudio tqdm numpy scipy 
~~~

#### Note that code has been written such that it expects a single input images
from the dataset. This would mean that the left image is the Jittered truth image
while the right image is the unjittered ground truth
