import os
import numpy as np
from PIL import Image

# Load the .npy file
images = np.load('./octmnist/val_images.npy')
labels = np.load('./octmnist/val_labels.npy')

# Create directories for each label
output_dir = "images_by_label_octmnist"
os.makedirs(output_dir, exist_ok=True)

# Save images into folders based on labels
for i, (image, label) in enumerate(zip(images, labels)):
    label_dir = os.path.join(output_dir, str(label[0]))
    os.makedirs(label_dir, exist_ok=True)
    image_path = os.path.join(label_dir, f"image_{i}.png")  # Adjust file format as needed
    # image = (image * 255).astype(np.uint8)  # Convert image to uint8 if necessary
    Image.fromarray(image).save(image_path)

print("Images saved into folders based on labels.")