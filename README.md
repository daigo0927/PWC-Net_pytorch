# Acknowledges
- Thanks [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) for data transformers
- Thanks [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard) for Tensorboard logger
- Thanks [sksq96/pytorch-summary](https://github.com/sksq96/pytorch-summary) for model summary similar to `model.summary()` in Keras

# PWC-Net
**Still in Progress**
- [ ] EPE compute on train_batch & test_batch
- [ ] predict & test codes
- [ ] Support to FlyingThings dataset.
- [ ] Support to KITTI dataset.
- [ ] Load official Caffe weights. (After the official Caffe implementation is released.)


This is an unofficial pytorch implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**.

Resources:  
[arXiv](https://arxiv.org/abs/1709.02371) | [Caffe](https://github.com/deqings/PWC-Net)(official)


# Usage
- Requirements
    - Python 3.6
    - PyTorch 0.3.1


- Get Started with Demo
```
python3 main.py predict --load models/best.model -i example/1.png example/2.png -o example/output.flo
```

- Prepare Datasets
    - Download [FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip) for training  
        When setting `--dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ├── 00001_flow.flo
        ├── 00001_img1.ppm
        ├── 00001_img2.ppm
        ...
        ```
    - Download [FlyingThings](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2) for fine-tuning  
        When setting `--dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ```
    - Download [MPI-Sintel](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) for fine-tuning if you want to validate on MPI-Sintel  
        When setting `--dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ├── training
        |   ├── final
        |   ├── clean
        |   ├── flow
        |   ...
        ├── test
        ...
        ```
    - Download [KITTI](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) for fine-tuning if you want to validate on KITTI  
        When setting `--dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ├── training
        ├── testing
        ```

- Train
```
python3 main.py train --dataset <DATASET_NAME> --dataset_dir <DIR_NAME>
```


# Details
## Network Parameters
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 192, 224]             448
         LeakyReLU-2         [-1, 16, 192, 224]               0
            Conv2d-3         [-1, 16, 192, 224]            2320
         LeakyReLU-4         [-1, 16, 192, 224]               0
            Conv2d-6          [-1, 32, 96, 112]            4640
         LeakyReLU-7          [-1, 32, 96, 112]               0
            Conv2d-8          [-1, 32, 96, 112]            9248
         LeakyReLU-9          [-1, 32, 96, 112]               0
           Conv2d-11           [-1, 64, 48, 56]           18496
        LeakyReLU-12           [-1, 64, 48, 56]               0
           Conv2d-13           [-1, 64, 48, 56]           36928
        LeakyReLU-14           [-1, 64, 48, 56]               0
           Conv2d-16           [-1, 96, 24, 28]           55392
        LeakyReLU-17           [-1, 96, 24, 28]               0
           Conv2d-18           [-1, 96, 24, 28]           83040
        LeakyReLU-19           [-1, 96, 24, 28]               0
           Conv2d-21          [-1, 128, 12, 14]          110720
        LeakyReLU-22          [-1, 128, 12, 14]               0
           Conv2d-23          [-1, 128, 12, 14]          147584
        LeakyReLU-24          [-1, 128, 12, 14]               0
           Conv2d-26            [-1, 192, 6, 7]          221376
        LeakyReLU-27            [-1, 192, 6, 7]               0
           Conv2d-28            [-1, 192, 6, 7]          331968
        LeakyReLU-29            [-1, 192, 6, 7]               0
           Conv2d-32         [-1, 16, 192, 224]             448
        LeakyReLU-33         [-1, 16, 192, 224]               0
           Conv2d-34         [-1, 16, 192, 224]            2320
        LeakyReLU-35         [-1, 16, 192, 224]               0
           Conv2d-37          [-1, 32, 96, 112]            4640
        LeakyReLU-38          [-1, 32, 96, 112]               0
           Conv2d-39          [-1, 32, 96, 112]            9248
        LeakyReLU-40          [-1, 32, 96, 112]               0
           Conv2d-42           [-1, 64, 48, 56]           18496
        LeakyReLU-43           [-1, 64, 48, 56]               0
           Conv2d-44           [-1, 64, 48, 56]           36928
        LeakyReLU-45           [-1, 64, 48, 56]               0
           Conv2d-47           [-1, 96, 24, 28]           55392
        LeakyReLU-48           [-1, 96, 24, 28]               0
           Conv2d-49           [-1, 96, 24, 28]           83040
        LeakyReLU-50           [-1, 96, 24, 28]               0
           Conv2d-52          [-1, 128, 12, 14]          110720
        LeakyReLU-53          [-1, 128, 12, 14]               0
           Conv2d-54          [-1, 128, 12, 14]          147584
        LeakyReLU-55          [-1, 128, 12, 14]               0
           Conv2d-57            [-1, 192, 6, 7]          221376
        LeakyReLU-58            [-1, 192, 6, 7]               0
           Conv2d-59            [-1, 192, 6, 7]          331968
        LeakyReLU-60            [-1, 192, 6, 7]               0
  CostVolumeLayer-63             [-1, 81, 6, 7]               0
  CostVolumeLayer-65           [-1, 81, 12, 14]               0
  CostVolumeLayer-67           [-1, 81, 24, 28]               0
  CostVolumeLayer-69           [-1, 81, 48, 56]               0
  CostVolumeLayer-71          [-1, 81, 96, 112]               0
  CostVolumeLayer-73         [-1, 81, 192, 224]               0
================================================================
Total params: 2044320
Trainable params: 2044320
Non-trainable params: 0
----------------------------------------------------------------

```
