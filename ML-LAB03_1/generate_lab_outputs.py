import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
print("Part A: Vectors and Matrices")
vector = np.array([10, 20, 30, 40])
print(vector)
print("Shape:", vector.shape)
print("Number of dimensions:", vector.ndim)

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(matrix)
print("Shape:", matrix.shape)
print("Number of dimensions:", matrix.ndim)

tabular_data = np.array([
    [2, 60, 50],
    [5, 80, 75],
    [8, 95, 90]
])
print(tabular_data)
print("Shape:", tabular_data.shape)

print("\nPart B: Single Grayscale Image as a 2D Tensor")
gray_image = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0],
    [0, 255, 255, 255, 255, 255, 255,   0],
    [0, 255,   0,   0,   0,   0, 255,   0],
    [0, 255,   0, 255, 255,   0, 255,   0],
    [0, 255,   0, 255, 255,   0, 255,   0],
    [0, 255,   0,   0,   0,   0, 255,   0],
    [0, 255, 255, 255, 255, 255, 255,   0],
    [0,   0,   0,   0,   0,   0,   0,   0]
])
print(gray_image)
print("Shape:", gray_image.shape)
print("Number of dimensions:", gray_image.ndim)

plt.imshow(gray_image, cmap="gray")
plt.title("Single Grayscale Image")
plt.axis("off")
plt.savefig("figures/grayscale_image.png")
plt.close()

print("\nPart C: Flattening an Image")
flat_image = gray_image.flatten()
print(flat_image)
print("Original shape:", gray_image.shape)
print("Flattened shape:", flat_image.shape)

print("\nPart D: Multiple Grayscale Images as a 3D Tensor")
image_1 = gray_image
image_2 = np.flipud(gray_image)
image_3 = np.fliplr(gray_image)
gray_batch = np.array([image_1, image_2, image_3])
print("Batch shape:", gray_batch.shape)
print("Number of dimensions:", gray_batch.ndim)

for i in range(gray_batch.shape[0]):
    plt.imshow(gray_batch[i], cmap="gray")
    plt.title(f"Grayscale Image {i + 1}")
    plt.axis("off")
    plt.savefig(f"figures/grayscale_batch_{i+1}.png")
    plt.close()

print("\nPart E: Single Color Image as a 3D Tensor")
height = 8
width = 8
channels = 3

rgb_image = np.zeros((height, width, channels), dtype=np.uint8)

# Red square in top-left
rgb_image[0:4, 0:4, 0] = 255

# Green square in top-right
rgb_image[0:4, 4:8, 1] = 255

# Blue square in bottom-left
rgb_image[4:8, 0:4, 2] = 255

# White square in bottom-right
rgb_image[4:8, 4:8, :] = 255

print("RGB image shape:", rgb_image.shape)
print("Number of dimensions:", rgb_image.ndim)

plt.imshow(rgb_image)
plt.title("Single RGB Image")
plt.axis("off")
plt.savefig("figures/rgb_image.png")
plt.close()

red_channel = rgb_image[:, :, 0]
green_channel = rgb_image[:, :, 1]
blue_channel = rgb_image[:, :, 2]

print("Red channel shape:", red_channel.shape)
print("Green channel shape:", green_channel.shape)
print("Blue channel shape:", blue_channel.shape)

plt.imshow(red_channel, cmap="gray")
plt.title("Red Channel")
plt.axis("off")
plt.savefig("figures/red_channel.png")
plt.close()

plt.imshow(green_channel, cmap="gray")
plt.title("Green Channel")
plt.axis("off")
plt.savefig("figures/green_channel.png")
plt.close()

plt.imshow(blue_channel, cmap="gray")
plt.title("Blue Channel")
plt.axis("off")
plt.savefig("figures/blue_channel.png")
plt.close()

print("\nPart F: Flattening a Color Image")
flat_rgb_image = rgb_image.flatten()
print("Original RGB shape:", rgb_image.shape)
print("Flattened RGB shape:", flat_rgb_image.shape)

print("\nPart G: Video as a 4D Tensor")
frames = []
for i in range(5):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    # Moving red square
    frame[2:5, i:i+3, 0] = 255
    frames.append(frame)

video_tensor = np.array(frames)
print("Video tensor shape:", video_tensor.shape)
print("Number of dimensions:", video_tensor.ndim)

for i in range(video_tensor.shape[0]):
    plt.imshow(video_tensor[i])
    plt.title(f"Video Frame {i + 1}")
    plt.axis("off")
    plt.savefig(f"figures/video_frame_{i+1}.png")
    plt.close()

print("\nPart H: Batch of Videos")
video_1 = video_tensor
video_2 = np.flip(video_tensor, axis=2)
video_batch = np.array([video_1, video_2])

print("Video batch shape:", video_batch.shape)
print("Number of dimensions:", video_batch.ndim)

with open('output.txt', 'w') as f:
    f.write("Generated figures and output successfully.")
