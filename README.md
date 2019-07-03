# Neurartist

A ready-to-use implementation of various Artistic Deep Learning Algorithms.

* *Image Style Transfer Using Convolutional Neural Networks*, Gatys et. al, 2016

# Installation

```
# It is recommended to install torch/torchvision manually before this command, according to your hardware configuration (see below)
pip install neurartist
```

Please note that the use of a GPU is recommended, as CNN computations are pretty slow on a CPU.

NB for GPU users: pip ships `torch`/`torchvision` with the Cuda Toolkit 9.0. If you use a more recent version of the Cuda Toolkit, see the [PyTorch website](https://pytorch.org/get-started/locally/) for instructions on PyTorch installation with another version of the toolkit.

# Usage

## Console entrypoint

```
# Then see the builtin help for usage details
neurartist --help
```

## Library

```
import neurartist
```

To be added.

# Development

Anaconda is strongly recommended:

```
conda create python=3.7 --name neurartist_env
conda activate neurartist_env

# with gpu
conda install pytorch torchvision cudatoolkit=<your cudatoolkit version> -c pytorch
conda install --file requirements.txt

# with cpu
conda install pytorch-cpu torchvision-cpu -c pytorch
conda install --file requirements.txt
```

You can then run the main entrypoint directly using:

```
python -m neurartist --help
```

Or build and install the wheel file with the `--editable` flag.

## TODO

* Documentation.
* Implement the remaining parts of the jupyter notebook.
* [Semantic segmentation as described in this article as to limit spillovers](https://arxiv.org/pdf/1703.07511.pdf): different approach than guided gram matrices, but same idea of using spatial guidance channels that describe a semantic segmentation of our images.
* More deep-artistic algorithms.
