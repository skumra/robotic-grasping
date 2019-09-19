# Antipodal Robotic Grasping
We present a novel generative residual convolutional neural network based model architecture which detects objects in the camera’s field of view and predicts a suitable antipodal grasp configuration for the objects in the image.

This repository contains the implementation of the Generative Residual Convolutional Neural Network (GR-ConvNet) from the paper:

### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry:

```
@article{kumra2019antipodal,
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network},
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  journal={arXiv preprint arXiv:1909.04810},
  year={2019}
}
```

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation

```
git clone git@github.com:skumra/robotic-grasping.git
cd robotic-grasping
pip install -r requirements.txt
```
