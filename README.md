# Re3 in PyTorch
[Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects](https://danielgordon10.github.io/pdfs/re3.pdf)

<img src="/demo/sample_1.gif" height="300"/> <img src="/demo/sample_2.gif" height="300"/>

This is the Official Repository for Re3 in PyTorch. However, it has some significant differences between it and the [TensorFlow repository](https://github.com/danielgordon10/re3-tensorflow).
1. Due to PyTorch's dynamic graph construction, the network can now be trained one unroll at a time. This means less time preprocessing the images, but also it doesn't parallelize as well.
1. The simulation code is removed as it does not work with this one unroll setup.
1. The code is Python 3 compatible and encouraged.

## First Time Setup:
I have switched from virtualenv to Anaconda because it allows for easier CUDA integration. The setup is much the same as in the TensorFlow version.
```bash
conda create -n re3-pytorch-env python=3.6.8
conda env update -n re3-pytorch-env -f env.yml
```

## Model:
The model weights we used in our paper were ported from Caffe to Tensorflow and then to PyTorch. There may be slight differences in performance from the original paper.
Weights can be downloaded by running `sh download_weights_large.sh`

A smaller network trained in PyTorch is also available. Its performance is worse, but it is significantly faster.
These weights can be downloaded by running `sh download_weights_small.sh`.
Additionally, set `USE_SMALL_NET = True` in [constants.py](constants.py).

## Run the Demo:
1. [Download the pretrained weights](#model)
1. Run `python demo/image_demo.py`


## Folders and Files:
### Most important for using Re3 in a new project:
1. [tracker/re3_tracker.py](tracker/re3_tracker.py) - The tracker file for actually tracking objects.
1. [tracker/network.py](tracker/network.py) - The network file specifying the layout of Re3.
1. [constants.py](constants.py) - A place to put constants that are used in multiple files such as image size and log location.

### Most important for (re)training Re3 on new data:
1. [Training Readme](training/README.md)
1. [Dataset Readme](training/datasets/README.md)
1. [training/unrolled_solver.py](training/unrolled_solver.py)

Re3 is released under the GPL V3.

Please cite Re3 in your publications if it helps your research:
```
@article{gordon2018re3,
     title={Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects},
     author={Gordon, Daniel and Farhadi, Ali and Fox, Dieter},
     journal={IEEE Robotics and Automation Letters},
     volume={3},
     number={2},
     pages={788--795},
     year={2018},
     publisher={IEEE}
}
```
