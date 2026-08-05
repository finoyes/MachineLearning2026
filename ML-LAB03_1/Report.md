# Lab 3.1: Understanding Data as Tensors

## Part K: Student Tasks Outputs

### Vector and Matrix Creation
**Vector**
```text
[10 20 30 40]
Shape: (4,)
Number of dimensions: 1
```

**Matrix**
```text
[[10 20 30]
 [40 50 60]
 [70 80 90]]
Shape: (3, 3)
Number of dimensions: 2
```

### Grayscale Image Creation and Display
**Grayscale Image (8x8)**
```text
[[  0   0   0   0   0   0   0   0]
 [  0 255 255 255 255 255 255   0]
 [  0 255   0   0   0   0 255   0]
 [  0 255   0 255 255   0 255   0]
 [  0 255   0 255 255   0 255   0]
 [  0 255   0   0   0   0 255   0]
 [  0 255 255 255 255 255 255   0]
 [  0   0   0   0   0   0   0   0]]
Shape: (8, 8)
Number of dimensions: 2
```
![Grayscale Image](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/grayscale_image.png)

### Flattening Grayscale Image
**Flattened Grayscale Vector**
```text
[  0   0   0   0   0   0   0   0   0 255 255 255 255 255 255   0   0 255
   0   0   0   0 255   0   0 255   0 255 255   0 255   0   0 255   0 255
 255   0 255   0   0 255   0   0   0   0 255   0   0 255 255 255 255 255
 255   0   0   0   0   0   0   0   0   0]
Original shape: (8, 8)
Flattened shape: (64,)
```

### Batch of Grayscale Images
**Shape of batch of 3 images**
```text
Batch shape: (3, 8, 8)
Number of dimensions: 3
```
![Batch Image 1](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/grayscale_batch_1.png)
![Batch Image 2](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/grayscale_batch_2.png)
![Batch Image 3](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/grayscale_batch_3.png)

### RGB Image Creation and Display
**RGB Image Tensor Info**
```text
RGB image shape: (8, 8, 3)
Number of dimensions: 3
```
![RGB Image](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/rgb_image.png)

### RGB Channel Extraction
**Channel Shapes**
```text
Red channel shape: (8, 8)
Green channel shape: (8, 8)
Blue channel shape: (8, 8)
```
![Red Channel](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/red_channel.png)
![Green Channel](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/green_channel.png)
![Blue Channel](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/blue_channel.png)

### Flattening RGB Image
```text
Original RGB shape: (8, 8, 3)
Flattened RGB shape: (192,)
```

### Video Tensor Creation and Display
**Video Tensor Shape**
```text
Video tensor shape: (5, 8, 8, 3)
Number of dimensions: 4
```
![Video Frame 1](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/video_frame_1.png)
![Video Frame 2](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/video_frame_2.png)
![Video Frame 3](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/video_frame_3.png)
![Video Frame 4](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/video_frame_4.png)
![Video Frame 5](file:///e:/sem5/prac/MLlabs/ML-LAB03_1/figures/video_frame_5.png)


## Tensor Shape Summary Table

| Data | Tensor Shape | Meaning |
|---|---|---|
| One tabular record | `(features,)` | One row of features |
| Tabular dataset | `(samples, features)` | Many rows and columns |
| One grayscale image | `(height, width)` | 2D pixel grid |
| Multiple grayscale images | `(images, height, width)` | Batch of grayscale images |
| One RGB image | `(height, width, channels)` | Color image |
| Multiple RGB images | `(images, height, width, channels)` | Batch of color images |
| One video | `(frames, height, width, channels)` | Sequence of RGB frames |
| Multiple videos | `(videos, frames, height, width, channels)` | Batch of videos |

## Part L: Reflection Questions

1. **Why is a grayscale image represented as a 2D tensor?**
   A grayscale image consists of a grid of pixels where each pixel has a single intensity value. A 2D array naturally models this grid with dimensions of `(height, width)`.

2. **Why is an RGB image represented as a 3D tensor?**
   An RGB image has a 2D spatial grid `(height, width)`, but each pixel is made up of three color values (Red, Green, Blue) representing color intensity. The third dimension stores these color channels, resulting in a `(height, width, 3)` tensor.

3. **Why is a video represented as a 4D tensor?**
   A video is a sequence of RGB images over time (frames). Since a single RGB image is a 3D tensor `(height, width, channels)`, adding a temporal dimension for frames results in a 4D tensor with dimensions `(frames, height, width, channels)`.

4. **What happens when we flatten an image?**
   Flattening converts the multi-dimensional structure (the 2D grid) into a single, continuous 1D array or vector.

5. **Why can flattening be useful for classical ML?**
   Classical Machine Learning algorithms (like Support Vector Machines, Logistic Regression, or Random Forests) are inherently designed to take a 1D feature vector for each sample as input. 

6. **Why can flattening be harmful for image understanding?**
   Flattening an image destroys the inherent spatial structure and 2D local dependencies (e.g., edges, textures, shapes). 

7. **Why are CNNs better suited for image data than simple flattened input?**
   Convolutional Neural Networks (CNNs) process images while retaining their original 2D or 3D structure. By using convolutional filters, they are able to learn local patterns, shapes, and spatial hierarchies efficiently without losing context.

8. **What does the channel dimension mean in an RGB image?**
   The channel dimension represents the separate color components (Red, Green, and Blue) that mix together to determine the visual color of a pixel.

9. **What does the frame dimension mean in a video?**
   The frame dimension represents the temporal axis—or time. Each index along this dimension points to a distinct, complete image representing a single moment in the video.

10. **How is tabular data different from image data?**
    Tabular data has explicitly defined features (columns) where each column represents a different semantic attribute, and order generally doesn't define spatial correlation. In contrast, image data consists of homogeneous values (pixel intensities) that have strict spatial and structural relationships; neighboring pixels are highly correlated and their order is essential to the meaning of the data.
