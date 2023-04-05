Official PyTorch implementation of our CVPRW2023 paper: Asymmetric Color Transfer with Consistent Modality Learning.

-------------------------------------------------
**We provide a simple training and testing process as follows:**

-------------------------------------------------
**Dependencies**
* Python 3.8
* PyTorch 1.10.0+cu113

-------------------------------------------------
**Train**

Setup the datasets paths in the configure file.

python train.py --config 1

The checkpoints and log file are saved in *output*.

-------------------------------------------------
**Test**

Pretrained models are placed in *ckp*, and test samples are placed in *images\in*.

python test.py --config 1 --ckp setup1.pth

Finally, the results can be found in *images\out*.

-------------------------------------------------
**Results**
![show](https://github.com/keviner1/imgs/blob/main/SICNet-comp.png?raw=true)



