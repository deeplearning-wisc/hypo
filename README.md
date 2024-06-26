# HYPO: Hyperspherical Out-of-Distribution Generalization

This codebase provides a Pytorch implementation for the paper: [HYPO: Hyperspherical Out-of-Distribution Generalization](https://openreview.net/pdf?id=VXak3CZZGC) by Haoyue Bai*, Yifei Ming*, Julian Katz-Samuels, and Yixuan Li, .

**Remarks**: The current codebase is available for preview purposes only and is still under development. We are actively working on eliminating hard-coded links, removing unused arguments, and streamlining the processes for loading models and datasets. Please stay tuned for forthcoming updates.

### Abstract

Out-of-distribution (OOD) generalization is critical for machine learning models deployed in the real world.  However, achieving this can be fundamentally challenging, as it requires the ability to learn invariant features across different domains or environments. In this paper, we propose a novel framework HYPO (HYPerspherical OOD generalization) that provably learns domain-invariant representations in a hyperspherical space. In particular, our hyperspherical learning algorithm is guided by intra-class variation and inter-class separation principles---ensuring that features from the same class (across different training domains) are closely aligned with their class prototypes, while different class prototypes are maximally separated. We further provide theoretical justifications on how our prototypical learning objective improves the OOD generalization bound. Through extensive experiments on challenging OOD benchmarks, we demonstrate that our approach outperforms competitive baselines and achieves superior performance.


## Quick Start

### Data Preparation
In this work, we evaluate the OOD generalization performance over a range of environmental discrepancies such as domains, image corruptions, and perturbations. 

**OOD generalization across domains**: The default root directory for ID and OOD datasets is `datasets/`. We consider [PACS](https://arxiv.org/abs/1710.03077), [Office-Home](https://arxiv.org/abs/1706.07522), [VLCS](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf), [Terra Incognita](https://arxiv.org/abs/1807.04975). You may use `scripts/download.py` (from [DomainBed](https://github.com/facebookresearch/DomainBed)) to download and prepare the datasets for domain generalization.

**OOD generalization across common corruptions**: The default root directory for ID and OOD datasets is `datasets/`. We consider 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) & [CIFAR-10-C](https://arxiv.org/abs/1903.12261) and ImageNet-100 & [ImageNet-100-C](https://arxiv.org/abs/1903.12261).
In alignment with prior works on the [ImageNet-100](https://github.com/deeplearning-wisc/MCM/tree/main) subset, the script for generating the subset is provided [here](https://github.com/deeplearning-wisc/MCM/blob/main/create_imagenet_subset.py).

#### CIFAR-10 & CIFAR-10-C

- Create a folder named `cifar-10/` and a folder `cifar-10-c/` under `$datasets`.
- Download the dataset from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) and extract the training and validation sets to `$DATA/cifar-10/`.
- Download the dataset from the [CIFAR-10-C](https://arxiv.org/abs/1903.12261) and extract the training and test sets to `$DATA/cifar-10-c/`. The directory structure should look like
```
cifar-10/
|–– train/ 
|–– val/
cifar-10-c/
|–– CorCIFAR10_train/ 
|–– CorCIFAR10_test/
```

#### ImageNet-100 & ImageNet-100-C

- Create a folder named `imagenet-100/` and a folder `imagenet-100-c/` under `$datasets`.
- Create `images/` under `imagenet-100/` and `imagenet-100-c/.
- Download the dataset from the [ImageNet-100](https://image-net.org/index.php](https://github.com/deeplearning-wisc/MCM/tree/main) and extract the training and validation sets to `$DATA/imagenet-100/images`.
- Download the dataset from the [ImageNet-100-C](https://arxiv.org/abs/1903.12261) and extract the training and validation sets to `$DATA/imagenet-100-c/images`. The directory structure should look like
```
imagenet-100/
|–– images/
|   |–– train/ # contains 100 folders like n01440764, n01443537, etc.
|   |–– val/
imagenet-100-c/
|–– images/
|   |–– train/ 
|   |–– val/
```

## Training and Evaluation 

### Model Checkpoints

**Evaluate pre-trained checkpoints** 

Our checkpoints can be downloaded [here](https://drive.google.com/file/d/1nflCX3YUTwX54QR_jiLlPq9q6Hni2YMe/view?usp=drive_link). Create a directory named `checkpoints/[ID_DATASET]` in the root directory of the project and put the downloaded checkpoints here. For example, for CIFAR-10 and PACS:

```
checkpoints/
---CIFAR-10/	 	
------checkpoint_hypo_resnet18_cifar10.pth.tar
---PACS/	 	
------checkpoint_hypo_resnet50_td_photo.pth.tar
------checkpoint_hypo_resnet50_td_cartoon.pth.tar
------checkpoint_hypo_resnet50_td_sketch.pth.tar
------checkpoint_hypo_resnet50_td_art_painting.pth.tar
```

The following scripts can be used to evaluate the OOD detection performance:

```
sh scripts/eval_ckpt_cifar10.sh ckpt_c10 #for CIFAR-10
sh scripts/eval_ckpt_pacs.sh ckpt_pacs # for PACS
```



**Evaluate custom checkpoints** 

If the default directory to save checkpoints is not `checkpoints`, create a softlink to the directory where the actual checkpoints are saved and name it as `checkpoints`. For example, checkpoints for CIFAR-100 (ID) are structured as follows: 

```python
checkpoints/
---CIFAR-100/
------name_of_ckpt/
---------checkpoint_500.pth.tar
```


**Train from scratch** 

We provide sample scripts to train from scratch. Feel free to modify the hyperparameters and training configurations.

```
sh scripts/train_hypo_cifar10.sh
sh scripts/train_hypo_dg.sh
```

**Fine-tune from ImageNet pre-trained models** 

We also provide fine-tuning scripts on large-scale datasets such as ImageNet-100.

```
sh scripts/train_hypo_imgnet100.sh 
```



### Citation

If you find our work useful, please consider citing our paper:
```
@inproceedings{
hypo2024,
title={Provable Out-of-Distribution Generalization in Hypersphere},
author={Haoyue Bai and Yifei Ming and Julian Katz-Samuels and Yixuan Li},
booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
year={2024},
}
```


### Further discussions
For more discussions on the method and extensions, feel free to drop an email at hbai39@wisc.edu or ming5@wisc.edu
