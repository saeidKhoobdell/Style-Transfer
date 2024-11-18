
# Neural Style Transfer with PyTorch

This project demonstrates how to implement the Neural Style Transfer algorithm using PyTorch, inspired by the research paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al. Neural Style Transfer aims to apply the artistic style of one image to the content of another by optimizing an input image to minimize both content and style loss.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Detailed Steps](#detailed-steps)
   - [1. Importing Packages](#1-importing-packages)
   - [2. Loading Images](#2-loading-images)
   - [3. Loss Functions](#3-loss-functions)
   - [4. Style Loss & Gram Matrix](#4-style-loss--gram-matrix)
   - [5. Model Import](#5-model-import)
   - [6. Gradient Descent](#6-gradient-descent)
   - [7. Running the Algorithm](#7-running-the-algorithm)
6. [References](#references)

---

## Introduction

Neural Style Transfer involves taking a content image and a style image and blending them so that the resulting image maintains the content of the original but mimics the artistic style of the style image. This project demonstrates the original optimization-based approach to style transfer.

## Project Overview

This notebook covers:
1. Importing necessary packages and setting up the device (CPU/GPU).
2. Loading and preprocessing content and style images.
3. Implementing content and style loss functions.
4. Calculating the Gram matrix for style representation.
5. Importing a pre-trained model (VGG) and modifying it for style transfer.
6. Optimizing the input image to minimize content and style loss.

## Setup and Installation

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- matplotlib

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/style-transfer.git
   cd style-transfer
   ```
2. Install the required packages using `pip`:
   ```bash
   pip install torch torchvision pillow matplotlib
   ```

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook styleTransfer.ipynb
   ```
2. Follow the steps outlined in the notebook to perform style transfer using your content and style images.

## Detailed Steps

### 1. Importing Packages
Necessary packages such as PyTorch, torchvision models, and image processing libraries are imported. The model relies on deep learning packages for style transfer.

### 2. Loading Images
The content and style images are loaded, resized to the same dimensions, and transformed into tensors with values normalized between 0 and 1.

### 3. Loss Functions
Content and style loss functions are defined to measure the differences between images. The optimization algorithm minimizes these loss functions.

### 4. Style Loss & Gram Matrix
The Gram matrix is used to represent the style of an image. The matrix encodes correlations between different feature maps.

### 5. Model Import
A pre-trained VGG model is imported and modified for calculating content and style representations.

### 6. Gradient Descent
The input image is optimized using gradient descent to minimize the content and style losses.

### 7. Running the Algorithm
The complete style transfer process is run to generate a stylized image.

## References
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) - Gatys et al.
