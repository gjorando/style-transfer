# Neurartist

A ready-to-use implementation of various Artistic Deep Learning Algorithms.

* *Image Style Transfer Using Convolutional Neural Networks*, Gatys et. al, 2016
* *Controlling Perceptual Factors in Neural Style Transfer*, Gatys et. al, 2016

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

See the examples below for the most common use cases.

## Library

```
import neurartist
```

To be added.

# Examples

* Basic usage: apply the style of an image to a content image, while preserving the semantic content.
```
neurartist -c content.jpg -s style.jpg
```
* Color control: apply a style, but preserve the color of the content image.
```
# Luminance only
neurartist -c content.jpg -s style.jpg --color-control luminance_only
# Luminance only, luma normalized
neurartist -c content.jpg -s style.jpg --color-control luminance_only --cc-luminance-only-normalize
# Color histogram matching
neurartist -c content.jpg -s style.jpg --color-control histogram_matching
```
* Style mixin: mix the coarse scale information of style1 (higher layers) with the fine scale information of style2 (lower layers), to create a mixed style to apply to a content image.
```
neurartist -c style1.jpg -s style2.jpg -o mixed.png --content-layers [22,29] --style-layers [1,6]
neurartist -c content.jpg -s mixed.png
```
* Efficient high resolution: first pass is a low resolution style transfer that efficiently catches coarse scale style features, second pass is a high resolution style transfer that upscales the result of the first pass and fills the lost fine information using fine scale style features.
```
neurartist -c content.jpg -s style.jpg -o lowres.png -S 500
neurartist -c content.jpg -s style.jpg -o highres.png -S 1000 --init-image-path lowres.png
```
* Spatial control: Guided gram matrices with guidance channels. Guidance paths should contain black and white guidance images (with the same size ratio as content and style images), defining the boundaries of semantic regions of each image. Style guidance images and content guidance images should have the same name, in correspondance to a semantic region. Segmentation of the image should be exhaustive.
```
neurartist -c content.jpg -s style.jpg --content-guidance content_image_guidance_path/ --style-guidance style_image_guidance_path/ --guidance-propagation-method inside
```

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

# TODO

* Documentation.
* Implement total variation loss (see [this](https://www.tensorflow.org/beta/tutorials/generative/style_transfer)).
* Implement guided sums.
* Examine if we need to add a fallback global guidance channel for pixels that aren't covered by any channel (briefly mentioned in the article but very vague). Short answer: yes, it is needed, with bigger kernel sizes the style transfer isn't performed at the boundary.
* For guidance channels propagation: investigate what they mean by "erosion" (see supplementary material document linked in the original article).
* [Semantic segmentation as described in this article as to limit spillovers](https://arxiv.org/pdf/1703.07511.pdf): different approach than guided gram matrices, but same idea of using spatial guidance channels that describe a semantic segmentation of our images.
* More deep-artistic algorithms.
