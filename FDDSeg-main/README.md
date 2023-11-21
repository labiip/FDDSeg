# FDDSeg
* This project is the official implementation for FDDSeg.

# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* The MSCMR dataset with annotations can be downloaded from: [CycleMix](https://github.com/BWGZK/CycleMix).

# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python >= 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone this project
```
git clone https://github.com/labiip/FDDSeg
cd FDDSeg
```
2. Data pre-processing
```
cd code
python dataloaders/acdc_data_processing.py
```
3. Train the model
```
cd code
python train_fddseg.py
```

4. Test the model
```
python test_fddseg_acdc.py
```

# Acknowledgement
* The codebase is adapted from the work [WSL4MIS](https://github.com/HiLab-git/WSL4MIS).

