# Neurartist

A ready-to-use implementation of various Artistic Deep Learning Algorithms.

* *Image Style Transfer Using Convolutional Neural Networks*, Gatys et. al, 2016

# Usage

```
python -m neurartist
```

Make sure that the libraries in `requirements.txt` are installed in your current environment. See [PyTorch website](https://pytorch.org/get-started/locally/) for instructions on PyTorch installation depending of your platform and package manager. Anaconda is strongly recommended:

```
conda create python=3.7 --name neurartist_env
conda activate neurartist_env

# with gpu
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install --file requirements.txt

# with cpu
conda install pytorch-cpu torchvision-cpu -c pytorch
conda install --file requirements_cpu.txt
```

Please note that the a GPU is recommended, as CNN computations are pretty slow on a CPU.

## TODO

* CLI interface.
* Implement the remaining parts of the jupyter notebook.
* [Semantic segmentation as described in this article as to limit spillovers](https://arxiv.org/pdf/1703.07511.pdf): different approach than guided gram matrices, but same idea of using spatial guidance channels that describe a semantic segmentation of our images.
* More deep-artistic algorithms.
