import matplotlib.pyplot as plt

# Ensure the model is in evaluation mode
model.eval()

# Get the original image (normalized for the model)
image_tensor = torch.FloatTensor(image / 255.0).unsqueeze(0).unsqueeze(0)

# Get the reconstructed image
with torch.no_grad():
    reconstructed_tensor = model(image_tensor)

reconstructed_image = reconstructed_tensor.squeeze().numpy()

# Calculate the pixel-wise absolute error
error_map = np.abs(reconstructed_image - (image / 255.0))

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original Image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Reconstructed Image
axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

# Error Map
# Use a color map that highlights differences, like 'viridis' or 'hot'
error_img = axes[2].imshow(error_map, cmap='hot', interpolation='nearest')
axes[2].set_title('Pixel-wise Reconstruction Error')
axes[2].axis('off')
fig.colorbar(error_img, ax=axes[2], orientation='vertical')

plt.tight_layout()
plt.show()