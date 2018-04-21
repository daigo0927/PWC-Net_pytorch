# PWC-Net
Still in progress, even no guarantee of a complete version, so feel free to copy/fork/PR/..., do anything you want.

**News**
- Offical repo [deqings/PWC-Net](https://github.com/deqings/PWC-Net) is released.
- Training code for FlyingChairs & Sintel is released.
    - Tested with CUDA 8.0, PyTorch 0.3, CentOS 7.
    - Due to my bad cost volume layer, batchsize=8 will occupy 9000MB+ memory on each GPU.

**Still in Progress**
- [ ] EPE compute on train_batch & test_batch
- [ ] predict & test codes
- [ ] Support to FlyingThings dataset.
- [ ] Support to KITTI dataset.
- [ ] Load official Caffe weights. (After the official Caffe implementation is released.)


This is an unofficial pytorch implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**.

Resources:  
[arXiv](https://arxiv.org/abs/1709.02371) | [offical caffe](https://github.com/deqings/PWC-Net)


# Usage
- Requirements
    - Python 3.6
    - PyTorch 0.3.1


- Get Started with Demo
```
python3 main.py --predict --load models/best.model -i example/1.png example/2.png -o example/output.flo
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
        ```

- Train
```
python3 main.py --train --dataset <DATASET_NAME> --dataset_dir <DIR_NAME>
```

- Validate