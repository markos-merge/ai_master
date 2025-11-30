import os
import numpy as np
from PIL import Image
from open_file import load_CIFAR10, load_label_names

def save_all_cifar10_images(data_folder, output_dir):
    """
    Loads the CIFAR-10 dataset and saves all images to disk in labeled folders.
    """
    # 1. Load the dataset and label names
    print("Loading CIFAR-10 data...")
    cifar10_data = load_CIFAR10(data_folder)
    label_names_bytes = load_label_names(data_folder)
    label_names = [name.decode('utf-8') for name in label_names_bytes]
    print(f"Found labels: {label_names}")

    # 2. Create the main output directory and subdirectories for each label
    os.makedirs(output_dir, exist_ok=True)
    for name in label_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    # 3. Keep track of image counts for unique filenames
    image_counters = {name: 0 for name in label_names}
    total_saved = 0

    # 4. Iterate through each data batch
    for batch in cifar10_data:
        images = batch[b'data']
        labels = batch[b'labels']

        # 5. Iterate through each image in the batch
        for i in range(len(images)):
            # Reshape and transpose the image from (3072,) to (32, 32, 3)
            img_array = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img_array, 'RGB')

            # Get the corresponding label and increment counter
            label_name = label_names[labels[i]]
            image_counters[label_name] += 1

            # Construct filename and save the image
            filename = f"{label_name}_{image_counters[label_name]:04d}.png"
            save_path = os.path.join(output_dir, label_name, filename)
            img.save(save_path)
            total_saved += 1

    print(f"\nFinished saving. Total images saved: {total_saved}")
    print(f"Images are located in the '{output_dir}' directory.")

if __name__ == "__main__":
    data_source_folder = './cifar-10'
    output_image_folder = 'cifar10_images'
    save_all_cifar10_images(data_source_folder, output_image_folder)