import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.metrics import mean_squared_error

# Load image
image_path = './l.jpg' 
img = Image.open(image_path)

# Convert image to RGB (if it's not already in RGB)
img_rgb = img.convert('RGB')

# Convert the image into a 3D numpy array (height x width x channels)
img_array = np.array(img_rgb)

# Calculate the original file size
original_file_size = os.path.getsize(image_path)
original_file_size_kb = original_file_size / 1024  # KB
original_file_size_mb = original_file_size / (1024 ** 2)  # MB

print(f"Original Image File Size: {original_file_size_kb:.2f} KB / {original_file_size_mb:.2f} MB")

# Calculate the number of pixels before PCA compression
original_pixels = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]


plt.figure(figsize=(10, 6))

# Loop over each channel (R, G, B)
colors = ['red', 'green', 'blue']
labels = ['Red Channel', 'Green Channel', 'Blue Channel']

for i in range(3): 
    # Flatten one channel
    channel_example = img_array[:, :, i].reshape(-1, img_array.shape[1])

    # Apply PCA to the channel
    pca_example = PCA()
    pca_example.fit(channel_example)

    # Plot cumulative explained variance for the current channel
    plt.plot(
        np.cumsum(pca_example.explained_variance_ratio_) * 100, 
        label=labels[i], 
        color=colors[i],
        marker='o',
        markersize=3
    )


plt.title("Scree Plot: Cumulative Explained Variance for RGB Channels")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.xlim(0, 500)  # Set x-axis range to 0-800
plt.ylim(0, 100)  # Ensure the y-axis is 0-100%
plt.grid()
plt.legend()
plt.show()


# Apply PCA to each channel (R, G, B)
channels = []
mse_per_channel = []
n_components = 100

for i in range(3):  # Loop over R, G, B channels
    channel = img_array[:, :, i]  # Extract channel
    channel_flattened = channel.reshape(-1, channel.shape[1])  # Flatten spatially (rows x cols)

    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(channel_flattened)
    reconstructed = pca.inverse_transform(compressed)  # Reconstruct
    mse = mean_squared_error(channel_flattened, reconstructed)
    mse_per_channel.append(mse)

    # Calculate and print cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Cumulative Explained Variance for Channel {i+1}: {cumulative_variance[-1]:.2f}")

    # Reshape back to original dimensions
    channels.append(reconstructed)

# Stack the reconstructed channels back into a single image
img_reconstructed = np.stack(channels, axis=2).astype(np.uint8)

# Save the compressed image
compressed_image_path = 'compressed_image1.jpeg'
compressed_image = Image.fromarray(img_reconstructed)
compressed_image.save(compressed_image_path)

# Calculate the compressed file size
compressed_file_size = os.path.getsize(compressed_image_path)
compressed_file_size_kb = compressed_file_size / 1024  # KB
compressed_file_size_mb = compressed_file_size / (1024 ** 2)  # MB

print(f"Compressed Image File Size: {compressed_file_size_kb:.2f} KB / {compressed_file_size_mb:.2f} MB")

# Calculate the number of pixels after PCA compression
pca_pixels = img_reconstructed.shape[0] * img_reconstructed.shape[1] * img_reconstructed.shape[2]

print(f"Number of Pixels Before PCA Compression: {original_pixels}")
print(f"Number of Pixels After PCA Compression: {pca_pixels}")

# Calculate overall MSE and error percentage
overall_mse = np.mean(mse_per_channel)
error_percentage = (np.sqrt(overall_mse) / 255) * 100
print(f"Mean Squared Error (MSE) Between Original and Reconstructed Image: {overall_mse:.2f}")
print(f"Error Percentage: {error_percentage:.2f}%")

# Plot original and compressed images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array)
plt.axis('off')

# Compressed image
plt.subplot(1, 2, 2)
plt.title(f"Compressed Image (PCA, {n_components} PCs)")
plt.imshow(img_reconstructed.astype(np.uint8))
plt.axis('off')

plt.show()
