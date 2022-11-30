# Unsupervised Point Cloud Completion and Segmentation by Generative Adversarial Autoencoding Network

This repository contains PyTorch implementation for **Unsupervised Point Cloud Completion and Segmentation by Generative Adversarial Autoencoding Network** (NeurIPS 2022).


## Start
### Requirements

```
CUDA                            10.2    ~   11.1
python                          3.7
torch                           1.8.0   ~   1.9.0
numpy
lmdb
msgpack-numpy
ninja                              
termcolor
tqdm
open3d                          0.9.0 
h5py
```
We successfully build the [pointnet2 operation lib](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib) with `CUDA 10.2 + torch 1.9.0` and `CUDA 11.1 + torch 1.8.0`, separately. It should work with PyTorch `1.9.0+`.

## Install 
```
cd util/pointnet2_ops_lib
python setup.py install
```

## Pretrained Models
Download ([NJU BOX](https://box.nju.edu.cn/d/69371034cf00461cbd52/) code:ugaan, [Baidu Yun](https://pan.baidu.com/s/1jB_LL0NTq8QWjU9pNevwHw?pwd=d5ye) code:d5ye) and extract our pretrained models to the `log` folder.
The `log` folder should be 
```
log
├── scannet
│   ├── bookshelf
│   │   ├── model-240.pkl
│   │   └── model-480.pkl
│   ├── chair
|   |   └── ...
│   └── ...
├── scannet_scanobj
│   └── ...
└── scanobj
    └── ...
```

## Datasets
Download ([NJU Box](https://box.nju.edu.cn/d/4855967efb2642909bc3/) code:ugaan, [Baidu Yun](https://pan.baidu.com/s/1uVx1PYaNlZju-qmb04mJwg?pwd=9wle) code:9wle) and extract our dataset, s3dis, and scanobjectnn to the `data` folder. The `data` folder should be
```
data
├── modelnet
|   └── ...
├── s3dis_coseg
├── scanobjectnn
├── shapenet
├── us_gt
└── ws
```

## Evaluation
Evaluate segmentation results on our dataset.
```
python test_scannet.py --cate chair
```
Evaluate segmentation results on S3DIS using the weights trained on our dataset.
```
python test_s3dis.py --cate chair
```
Evaluate segmentation results on ScanObjectNN using the weights trained on our dataset.
```
python test_scannet_scanobj.py --cate chair
```
Evaluate segmentation results on ScanObjectNN using the weights trained on ScanObjectNN.
```
python test_scanobj.py --cate chair
```
Evaluate completion results on our dataset.
```
python test_scannet_com.py --cate chair
```

## Train
Train on our dataset.
```
python train_scannet.py --cate chair
```
Train on our dataset for ScanObjectNN.
```
python train_scannet_scanobj.py --cate chair
```
Train on ScanObjectNN.
```
python train_scanobj.py --cate chair
```

## License
MIT License

## Acknowledgements
[pointnet2 operation lib](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib)

[Scan2CAD](https://github.com/skanti/Scan2CAD)

[ScanNet](https://github.com/ScanNet/ScanNet)

[S3DIS](https://ieeexplore.ieee.org/document/7780539/)

[ShapeNet](https://shapenet.org/)

[ModelNet](https://modelnet.cs.princeton.edu/)

[ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{ma2022ugaan,
  title={Unsupervised Point Cloud Completion and Segmentation by Generative Adversarial Autoencoding Network},
  author={Ma, Changfeng and Yang, Yang and Guo, Jie and Pan, Fei and Wang, Chongjun and Guo, Yanwen},
  booktitle={NeurIPS},
  year={2022}
}
```