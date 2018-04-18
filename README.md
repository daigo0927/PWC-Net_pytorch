# PWC-Net
Still in progress, even no guarantee of a complete version, so feel free to copy/fork/PR/..., do anything you want.

**TODOs**  
- [ ] Support to FlyingThings dataset.
- [ ] Support to KITTI dataset.
- [ ] Load official Caffe weights. (After the official Caffe implementation is released.)


This is an unofficial pytorch implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**.

Resources:  
[arXiv](https://arxiv.org/abs/1709.02371)


# Usage

- Requirements
    - Python3.6
    - PyTorch


- Get Started with Demo
```
python3 main.py --predict --load models/best.model -i example/1.png example/2.png -o example/output.flo
```

- Prepare Datasets
    - Download [FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip) for training
        ```
        <DIR_NAME>
        ```
    - Download [FlyingThings](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2) for fine-tuning
        ```
        <DIR_NAME>
        ```
    - Download [MPI-Sintel](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) for fine-tuning if you want to validate on MPI-Sintel
        ```
        <DIR_NAME>
        ```
    - Download [KITTI](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) for fine-tuning if you want to validate on KITTI
        ```
        <DIR_NAME>
        ```

- Train
```
python3 main.py --train
```

- Validate