import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 3.1: Understanding Data as Tensors\n",
    "\n",
    "**Course**: Machine Learning\n",
    "**Topic**: Multidimensional Tensors for Images and Videos\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part A: Vectors and Matrices\n",
    "\n",
    "#### Step 1: Create a Vector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "vector = np.array([10, 20, 30, 40])\n",
    "\n",
    "print(vector)\n",
    "print(\"Shape:\", vector.shape)\n",
    "print(\"Number of dimensions:\", vector.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 2: Create a Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "matrix = np.array([\n",
    "    [10, 20, 30],\n",
    "    [40, 50, 60],\n",
    "    [70, 80, 90]\n",
    "])\n",
    "\n",
    "print(matrix)\n",
    "print(\"Shape:\", matrix.shape)\n",
    "print(\"Number of dimensions:\", matrix.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 3: Connect Matrix to Tabular Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "tabular_data = np.array([\n",
    "    [2, 60, 50],\n",
    "    [5, 80, 75],\n",
    "    [8, 95, 90]\n",
    "])\n",
    "\n",
    "print(tabular_data)\n",
    "print(\"Shape:\", tabular_data.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part B: Single Grayscale Image as a 2D Tensor\n",
    "\n",
    "#### Step 4: Create a Small Grayscale Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "gray_image = np.array([\n",
    "    [0,   0,   0,   0,   0,   0,   0,   0],\n",
    "    [0, 255, 255, 255, 255, 255, 255,   0],\n",
    "    [0, 255,   0,   0,   0,   0, 255,   0],\n",
    "    [0, 255,   0, 255, 255,   0, 255,   0],\n",
    "    [0, 255,   0, 255, 255,   0, 255,   0],\n",
    "    [0, 255,   0,   0,   0,   0, 255,   0],\n",
    "    [0, 255, 255, 255, 255, 255, 255,   0],\n",
    "    [0,   0,   0,   0,   0,   0,   0,   0]\n",
    "])\n",
    "\n",
    "print(gray_image)\n",
    "print(\"Shape:\", gray_image.shape)\n",
    "print(\"Number of dimensions:\", gray_image.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 5: Display the Grayscale Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.imshow(gray_image, cmap=\"gray\")\n",
    "plt.title(\"Single Grayscale Image\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part C: Flattening an Image\n",
    "\n",
    "#### Step 6: Flatten the Grayscale Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "flat_image = gray_image.flatten()\n",
    "\n",
    "print(flat_image)\n",
    "print(\"Original shape:\", gray_image.shape)\n",
    "print(\"Flattened shape:\", flat_image.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part D: Multiple Grayscale Images as a 3D Tensor\n",
    "\n",
    "#### Step 8: Create Multiple Grayscale Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "image_1 = gray_image\n",
    "image_2 = np.flipud(gray_image)\n",
    "image_3 = np.fliplr(gray_image)\n",
    "\n",
    "gray_batch = np.array([image_1, image_2, image_3])\n",
    "\n",
    "print(\"Batch shape:\", gray_batch.shape)\n",
    "print(\"Number of dimensions:\", gray_batch.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 9: Display Multiple Grayscale Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(gray_batch.shape[0]):\n",
    "    plt.imshow(gray_batch[i], cmap=\"gray\")\n",
    "    plt.title(f\"Grayscale Image {i + 1}\")\n",
    "    plt.axis(\"off\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part E: Single Color Image as a 3D Tensor\n",
    "\n",
    "#### Step 10: Create a Small RGB Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "height = 8\n",
    "width = 8\n",
    "channels = 3\n",
    "\n",
    "rgb_image = np.zeros((height, width, channels), dtype=np.uint8)\n",
    "\n",
    "# Red square in top-left\n",
    "rgb_image[0:4, 0:4, 0] = 255\n",
    "\n",
    "# Green square in top-right\n",
    "rgb_image[0:4, 4:8, 1] = 255\n",
    "\n",
    "# Blue square in bottom-left\n",
    "rgb_image[4:8, 0:4, 2] = 255\n",
    "\n",
    "# White square in bottom-right\n",
    "rgb_image[4:8, 4:8, :] = 255\n",
    "\n",
    "print(\"RGB image shape:\", rgb_image.shape)\n",
    "print(\"Number of dimensions:\", rgb_image.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 11: Display RGB Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.imshow(rgb_image)\n",
    "plt.title(\"Single RGB Image\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 12: Inspect RGB Channels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "red_channel = rgb_image[:, :, 0]\n",
    "green_channel = rgb_image[:, :, 1]\n",
    "blue_channel = rgb_image[:, :, 2]\n",
    "\n",
    "print(\"Red channel shape:\", red_channel.shape)\n",
    "print(\"Green channel shape:\", green_channel.shape)\n",
    "print(\"Blue channel shape:\", blue_channel.shape)\n",
    "\n",
    "plt.imshow(red_channel, cmap=\"gray\")\n",
    "plt.title(\"Red Channel\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()\n",
    "\n",
    "plt.imshow(green_channel, cmap=\"gray\")\n",
    "plt.title(\"Green Channel\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()\n",
    "\n",
    "plt.imshow(blue_channel, cmap=\"gray\")\n",
    "plt.title(\"Blue Channel\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part F: Flattening a Color Image\n",
    "\n",
    "#### Step 13: Flatten RGB Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "flat_rgb_image = rgb_image.flatten()\n",
    "\n",
    "print(\"Original RGB shape:\", rgb_image.shape)\n",
    "print(\"Flattened RGB shape:\", flat_rgb_image.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part G: Video as a 4D Tensor\n",
    "\n",
    "#### Step 14: Create a Simple Video Tensor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "frames = []\n",
    "\n",
    "for i in range(5):\n",
    "    frame = np.zeros((8, 8, 3), dtype=np.uint8)\n",
    "    # Moving red square\n",
    "    frame[2:5, i:i+3, 0] = 255\n",
    "    frames.append(frame)\n",
    "\n",
    "video_tensor = np.array(frames)\n",
    "\n",
    "print(\"Video tensor shape:\", video_tensor.shape)\n",
    "print(\"Number of dimensions:\", video_tensor.ndim)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 15: Display Video Frames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(video_tensor.shape[0]):\n",
    "    plt.imshow(video_tensor[i])\n",
    "    plt.title(f\"Video Frame {i + 1}\")\n",
    "    plt.axis(\"off\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part H: Batch of Videos\n",
    "\n",
    "#### Step 16: Create Batch of Videos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_1 = video_tensor\n",
    "video_2 = np.flip(video_tensor, axis=2)\n",
    "video_batch = np.array([video_1, video_2])\n",
    "\n",
    "print(\"Video batch shape:\", video_batch.shape)\n",
    "print(\"Number of dimensions:\", video_batch.ndim)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Lab_3_1_Understanding_Data_as_Tensors.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
