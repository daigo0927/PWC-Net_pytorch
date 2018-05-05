# Acknowledgments
- [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): framework, data transformers, loss functions, and many details about flow estimation.
- [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard): Tensorboard logger
- [sksq96/pytorch-summary](https://github.com/sksq96/pytorch-summary): model summary similar to `model.summary()` in Keras

# PWC-Net
**Resources**  [arXiv](https://arxiv.org/abs/1709.02371) | [Caffe](https://github.com/deqings/PWC-Net)(official)
![](![example/flow.png])

**Still in Progress. Much appreciated if you have any advice.**  
This is an unofficial pytorch implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**.




# Usage
- Requirements
    - Python 3.6+
    - **PyTorch 0.4.0**


- Get Started with Demo
```
python3 main.py predict --load models/best.model -i example/1.png example/2.png -o example/output.flo
```

- Prepare Datasets
    - Download [FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip) for training  
        When setting `--dataset FlyingChairs --dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ├── 00001_flow.flo
        ├── 00001_img1.ppm
        ├── 00001_img2.ppm
        ...
        ```
    - Download [FlyingThings](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2) for fine-tuning  
        When setting `--dataset FlyingThings --dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ```
    - Download [MPI-Sintel](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) for fine-tuning if you want to validate on MPI-Sintel  
        When setting `--dataset Sintel --dataset_dir <DIR_NAME>`, your file tree should be like this
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
        When setting `--dataset KITTI --dataset_dir <DIR_NAME>`, your file tree should be like this
        ```
        <DIR_NAME>
        ├── training
        |   ├── image_2
        |   ├── image_3
        |   ...
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
            Conv2d-5          [-1, 32, 96, 112]            4640
         LeakyReLU-6          [-1, 32, 96, 112]               0
            Conv2d-7          [-1, 32, 96, 112]            9248
         LeakyReLU-8          [-1, 32, 96, 112]               0
            Conv2d-9           [-1, 64, 48, 56]           18496
        LeakyReLU-10           [-1, 64, 48, 56]               0
           Conv2d-11           [-1, 64, 48, 56]           36928
        LeakyReLU-12           [-1, 64, 48, 56]               0
           Conv2d-13           [-1, 96, 24, 28]           55392
        LeakyReLU-14           [-1, 96, 24, 28]               0
           Conv2d-15           [-1, 96, 24, 28]           83040
        LeakyReLU-16           [-1, 96, 24, 28]               0
           Conv2d-17          [-1, 128, 12, 14]          110720
        LeakyReLU-18          [-1, 128, 12, 14]               0
           Conv2d-19          [-1, 128, 12, 14]          147584
        LeakyReLU-20          [-1, 128, 12, 14]               0
           Conv2d-21            [-1, 192, 6, 7]          221376
        LeakyReLU-22            [-1, 192, 6, 7]               0
           Conv2d-23            [-1, 192, 6, 7]          331968
        LeakyReLU-24            [-1, 192, 6, 7]               0
FeaturePyramidExtractor-25            [-1, 192, 6, 7]               0
           Conv2d-26         [-1, 16, 192, 224]             448
        LeakyReLU-27         [-1, 16, 192, 224]               0
           Conv2d-28         [-1, 16, 192, 224]            2320
        LeakyReLU-29         [-1, 16, 192, 224]               0
           Conv2d-30          [-1, 32, 96, 112]            4640
        LeakyReLU-31          [-1, 32, 96, 112]               0
           Conv2d-32          [-1, 32, 96, 112]            9248
        LeakyReLU-33          [-1, 32, 96, 112]               0
           Conv2d-34           [-1, 64, 48, 56]           18496
        LeakyReLU-35           [-1, 64, 48, 56]               0
           Conv2d-36           [-1, 64, 48, 56]           36928
        LeakyReLU-37           [-1, 64, 48, 56]               0
           Conv2d-38           [-1, 96, 24, 28]           55392
        LeakyReLU-39           [-1, 96, 24, 28]               0
           Conv2d-40           [-1, 96, 24, 28]           83040
        LeakyReLU-41           [-1, 96, 24, 28]               0
           Conv2d-42          [-1, 128, 12, 14]          110720
        LeakyReLU-43          [-1, 128, 12, 14]               0
           Conv2d-44          [-1, 128, 12, 14]          147584
        LeakyReLU-45          [-1, 128, 12, 14]               0
           Conv2d-46            [-1, 192, 6, 7]          221376
        LeakyReLU-47            [-1, 192, 6, 7]               0
           Conv2d-48            [-1, 192, 6, 7]          331968
        LeakyReLU-49            [-1, 192, 6, 7]               0
FeaturePyramidExtractor-50            [-1, 192, 6, 7]               0
     WarpingLayer-51            [-1, 192, 6, 7]               0
      Correlation-52             [-1, 81, 6, 7]               0
           Conv2d-53            [-1, 128, 6, 7]          316928
        LeakyReLU-54            [-1, 128, 6, 7]               0
           Conv2d-55            [-1, 128, 6, 7]          147584
        LeakyReLU-56            [-1, 128, 6, 7]               0
           Conv2d-57             [-1, 96, 6, 7]          110688
        LeakyReLU-58             [-1, 96, 6, 7]               0
           Conv2d-59             [-1, 64, 6, 7]           55360
        LeakyReLU-60             [-1, 64, 6, 7]               0
           Conv2d-61             [-1, 32, 6, 7]           18464
        LeakyReLU-62             [-1, 32, 6, 7]               0
           Conv2d-63              [-1, 2, 6, 7]             578
OpticalFlowEstimator-64              [-1, 2, 6, 7]               0
     WarpingLayer-65          [-1, 128, 12, 14]               0
      Correlation-66           [-1, 81, 12, 14]               0
           Conv2d-67          [-1, 128, 12, 14]          243200
        LeakyReLU-68          [-1, 128, 12, 14]               0
           Conv2d-69          [-1, 128, 12, 14]          147584
        LeakyReLU-70          [-1, 128, 12, 14]               0
           Conv2d-71           [-1, 96, 12, 14]          110688
        LeakyReLU-72           [-1, 96, 12, 14]               0
           Conv2d-73           [-1, 64, 12, 14]           55360
        LeakyReLU-74           [-1, 64, 12, 14]               0
           Conv2d-75           [-1, 32, 12, 14]           18464
        LeakyReLU-76           [-1, 32, 12, 14]               0
           Conv2d-77            [-1, 2, 12, 14]             578
OpticalFlowEstimator-78            [-1, 2, 12, 14]               0
     WarpingLayer-79           [-1, 96, 24, 28]               0
      Correlation-80           [-1, 81, 24, 28]               0
           Conv2d-81          [-1, 128, 24, 28]          206336
        LeakyReLU-82          [-1, 128, 24, 28]               0
           Conv2d-83          [-1, 128, 24, 28]          147584
        LeakyReLU-84          [-1, 128, 24, 28]               0
           Conv2d-85           [-1, 96, 24, 28]          110688
        LeakyReLU-86           [-1, 96, 24, 28]               0
           Conv2d-87           [-1, 64, 24, 28]           55360
        LeakyReLU-88           [-1, 64, 24, 28]               0
           Conv2d-89           [-1, 32, 24, 28]           18464
        LeakyReLU-90           [-1, 32, 24, 28]               0
           Conv2d-91            [-1, 2, 24, 28]             578
OpticalFlowEstimator-92            [-1, 2, 24, 28]               0
     WarpingLayer-93           [-1, 64, 48, 56]               0
      Correlation-94           [-1, 81, 48, 56]               0
           Conv2d-95          [-1, 128, 48, 56]          169472
        LeakyReLU-96          [-1, 128, 48, 56]               0
           Conv2d-97          [-1, 128, 48, 56]          147584
        LeakyReLU-98          [-1, 128, 48, 56]               0
           Conv2d-99           [-1, 96, 48, 56]          110688
       LeakyReLU-100           [-1, 96, 48, 56]               0
          Conv2d-101           [-1, 64, 48, 56]           55360
       LeakyReLU-102           [-1, 64, 48, 56]               0
          Conv2d-103           [-1, 32, 48, 56]           18464
       LeakyReLU-104           [-1, 32, 48, 56]               0
          Conv2d-105            [-1, 2, 48, 56]             578
OpticalFlowEstimator-106            [-1, 2, 48, 56]               0
    WarpingLayer-107          [-1, 32, 96, 112]               0
     Correlation-108          [-1, 81, 96, 112]               0
          Conv2d-109         [-1, 128, 96, 112]          132608
       LeakyReLU-110         [-1, 128, 96, 112]               0
          Conv2d-111         [-1, 128, 96, 112]          147584
       LeakyReLU-112         [-1, 128, 96, 112]               0
          Conv2d-113          [-1, 96, 96, 112]          110688
       LeakyReLU-114          [-1, 96, 96, 112]               0
          Conv2d-115          [-1, 64, 96, 112]           55360
       LeakyReLU-116          [-1, 64, 96, 112]               0
          Conv2d-117          [-1, 32, 96, 112]           18464
       LeakyReLU-118          [-1, 32, 96, 112]               0
          Conv2d-119           [-1, 2, 96, 112]             578
OpticalFlowEstimator-120           [-1, 2, 96, 112]               0
          Conv2d-121         [-1, 128, 96, 112]           39296
       LeakyReLU-122         [-1, 128, 96, 112]               0
          Conv2d-123         [-1, 128, 96, 112]          147584
       LeakyReLU-124         [-1, 128, 96, 112]               0
          Conv2d-125         [-1, 128, 96, 112]          147584
       LeakyReLU-126         [-1, 128, 96, 112]               0
          Conv2d-127          [-1, 96, 96, 112]          110688
       LeakyReLU-128          [-1, 96, 96, 112]               0
          Conv2d-129          [-1, 64, 96, 112]           55360
       LeakyReLU-130          [-1, 64, 96, 112]               0
          Conv2d-131          [-1, 32, 96, 112]           18464
       LeakyReLU-132          [-1, 32, 96, 112]               0
          Conv2d-133           [-1, 2, 96, 112]             578
  ContextNetwork-134           [-1, 2, 96, 112]               0
================================================================
Total params: tensor(5.2958e+06)
Trainable params: tensor(5.2958e+06)
Non-trainable params: tensor(0)
----------------------------------------------------------------
```
