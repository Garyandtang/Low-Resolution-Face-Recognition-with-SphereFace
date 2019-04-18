# Low-Resolution Face-Recognition

The Python implementation of low-resolution face recognition project. The backbones of the face recognition system are **Alexnet** and **ShereFace**. To let the system fit the low resolution condtion, we guide the training of low-resolution model by considering the difference between  LR image's features and HR image's features.
* Alexnet is a famous CNN model for image classification. We used the pretrained Pythorch model and fine-turned it with Casia-Webface. The details of Alexnet is shown in this [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* ShereFace is the start-of-the-art face recognition algorithm. They proposed the **A-Softmax Loss** to improve the performance of face recogniton. The details of ShereFace is shown in this [paper](https://arxiv.org/pdf/1704.08063.pdf). The implementation of ShereFace in Pytorch is Credit to @aaronzguan.

This is part of my final-year project at The Hong Kong Polytechinic University.

## Some Description
* Deep-learning framework: Pytorch 
* OS: Linux
* Cuda toolkit: v9.0
* GPU: NVIDIA GeForce GTX 1080ti
* Training set: Casia-Webface
* Testing set: LFW Benchmark (View 2)
* Downsampling method: Bicubic Interpolation

## Result
