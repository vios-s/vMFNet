# vMFNet: Compositionality Meets Domain-generalised Segmentation
![model](figures/model.png)

This repository contains the official Pytorch implementation of [vMFNet: Compositionality Meets Domain-generalised Segmentation](https://arxiv.org/abs/2206.14538) accepted by [MICCAI 2022](https://conferences.miccai.org/2022/en/)).

The repository is created by [Xiao Liu](https://github.com/xxxliu95), [Spyridon Thermos](https://github.com/spthermo), [Pedro Sanchez](https://vios.science/team/sanchez), [Alison O'Neil](https://vios.science/team/oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/). You are welcome to visit our group website: [vios.s](https://vios.science/)

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* SciPy 1.5.2 or higher
* CUDA toolkit 10 or newer
* Nibabel
* Pillow
* CV2
* Scikit-image
* TensorBoard
* Tqdm


# Datasets
We used two datasets in the paper: [Multi-Centre, Multi-Vendor & Multi-Disease
Cardiac Image Segmentation Challenge (M&Ms) datast](https://www.ub.edu/mnms/) and [Spinal cord grey matter segmentation challenge dataset](http://niftyweb.cs.ucl.ac.uk/challenge/index.php). The dataloader in this repo is only for M&Ms dataset.

# Preprocessing

You need to first change the dirs in the scripts of preprocess folder. Download the M&Ms data and run ```split_MNMS_data.py``` to split the original dataset into different domains. Then run ```save_MNMS_2D.py``` to save the original 4D data as 2D numpy arrays. Finally, run ```save_MNMS_re.py``` to save the resolution of each datum. 

# Pre-training a UNet for image reconstruction
```
python pretrain.py -e 50 -bs 4 -c xxx/cp_unet_100_tvA/ -t A -w UNet_tvA -g 0
```

# Cluster the feature vectors to initilize the vMF kernels
```
python vMF_clustering.py -c xxx/cp_unet_100_tvA/ -t A -g 0
```

# Training
Note that the hyperparameters in the current version are tuned for BCD to A cases. For other cases, the hyperparameters and few specific layers of the model are slightly different. To train the model with 5% labeled data, run:
```
python train.py -e 1200 -bs 4 -c cp_vmfnet_5_tvA/ -enc xxx/cp_unet_100_tvA/UNet.pth -t A -w vmfnet_12_p5_tvA -g 0
```
Here the default learning rate is 1e-4. You can change the learning rate by adding ```-lr xe-x``` .

To train the model with 100% labeled data, try to change the training parameters to:
```
k_un = 1
k1 = 40
k2 = 4
```
The first parameter controls how many iterations you want the model to be trained with unlabaled data for every iteration of training. ```k1 = 40``` means the learning rate will start to decay after 40 epochs and ```k2 = 4``` means it will check if decay learning every 4 epochs.

Also, change the ratio ```k=0.05``` (line 148) to ```k=1``` in ```mms_dataloader_dg_aug.py```.

Then, run:
```
python train.py -e 200 -bs 4 -c cp_vmfnet_100_tvA/ -enc xxx/cp_unet_100_tvA/UNet.pth -t A -w vmfnet_12_p100_tvA -g 0
```
Finally, when training the model, changing the ```resampling_rate=1.2``` (line 50) in ```mms_dataloader_dg_aug.py``` to 1.1 - 1.3 may give better results. This will change the rescale ratio when preprocessing the images, which will affect the size of the anatomy of interest.

# Inference
After training, you can test the model:
```
python inference.py -bs 1 -c cp_vmfnet_2_tvA/ -enc xxx/cp_unet_100_tvA/UNet.pth -t A -g 0
```
This will output the DICE and Hausdorff results as well as the standard deviation. Similarly, changing the ```resampling_rate=1.2``` (line 47) in ```mms_dataloader_dg_aug_test.py``` to 1.1 - 1.3 may give better results.

# Examples of kernel activations:
![kernels](figures/kernels.png)

# Citation
```
@inproceedings{liu2022vmfnet,
  title={vMFNet: Compositionality Meets Domain-generalised Segmentation},
  author={Liu, Xiao and Thermos, Spyridon and Sanchez, Pedro and O’Neil, Alison and Tsaftaris, Sotirios A.},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```

# Acknowlegement
Part of the code is based on [SDNet](https://github.com/spthermo/SDNet), [DGNet](https://github.com/vios-s/DGNet), [CompositionalNets](https://github.com/AdamKortylewski/CompositionalNets) and [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).


# License
All scripts are released under the MIT License.
