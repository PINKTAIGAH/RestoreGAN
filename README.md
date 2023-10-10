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

Clone this repository

~~~bash
git clone https://github.com/PINKTAIGAH/RestoreGAN.git
cd RestoreGAN
~~~

Download desired dataset form Kaggle here (**Remember to add link**)

Define the location of the training and validation dataset by modifying the following two variables is scr/config.py

~~~python
TRAIN_DIR = "path/to/train/data/dir"
VAL_DIR = "path/to/validation/data/dir"
~~~

Other hyperparameters can also be modified in scr/config.py. Hyperparameters can also be modified from the command line by overwriting a variable's value. Below is an example of modifying TRAIN_DIR

~~~bash
echo "TRAIN_DIR = /new/path/to/dir" >> scr/config.py
~~~

## Training Model

Once the desired hyperparameters have been set, the generator and discriminator can be trained by running src/train.py 

~~~bash
python3 src/train.py
~~~

Using bash scripts located in scr/bash (**Remember to actually include**), we can preform crude hyperparameter optimisation. For example, to train model using varying lambda values for loss function

~~~bash
bash src/test_lambda_content.sh
~~~

To save weights of model while training, modify the value of the following bool in src/config.py. The weights are saved in models/

~~~python
SAVE_MODEL = True
# Modify the name of the file to which weights are saved to
CHECKPOINT_DISC_SAVE = "../models/disc.pth.tar"  # Discriminator weights
CHECKPOINT_GEN_SAVE = "../models/gen.pth.tar"    # Generator weights
~~~

To train using a pretrained model, we can load in pretrained weights by modifying the following variables

~~~python
LOAD_MODEL = True
# Modify the name of the file to which weights are saved to
CHECKPOINT_DISC_LOAD = "../models/disc.pth.tar"  # Discriminator weights
CHECKPOINT_GEN_LOAD = "../models/gen.pth.tar"    # Generator weights
~~~

