import os
import numpy as np
from PIL import Image

list_of_datasets = ['organsmnist', 'pneumoniamnist', 'retinamnist']

for dataset in list_of_datasets:

    for sub_dataset in ['train', 'test', 'val']:
            # Load the .npy file
        images = np.load(f'./DATA/{dataset}/{sub_dataset}_images.npy')
        labels = np.load(f'./DATA/{dataset}/{sub_dataset}_labels.npy')

        # Create directories for each label
        output_dir = dataset + "/" + sub_dataset
        os.makedirs(output_dir, exist_ok=True)

        # Save images into folders based on labels
        for i, (image, label) in enumerate(zip(images, labels)):
            label_dir = os.path.join(output_dir, str(label[0]))
            os.makedirs(label_dir, exist_ok=True)
            image_path = os.path.join(label_dir, f"image_{i}.png")  # Adjust file format as needed
            # image = (image * 255).astype(np.uint8)  # Convert image to uint8 if necessary
            Image.fromarray(image).save(image_path)

        print("Images saved into folders based on labels.")