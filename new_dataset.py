import os
import shutil
import random

def create_subset_dataset(original_dir, subset_dir, num_images):
    # Create directories if they don't exist
    os.makedirs(subset_dir, exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'val', 'labels'), exist_ok=True)

    # Get list of image files in train and val directories
    train_images = os.listdir(os.path.join(original_dir, 'train', 'images'))
    val_images = os.listdir(os.path.join(original_dir, 'val', 'images'))

    # Randomly select images
    train_images_subset = random.sample(train_images, min(num_images, len(train_images)))
    val_images_subset = random.sample(val_images, min(num_images // 4, len(val_images)))  # Divide by 4 because there are 4 times fewer images in val

    # Copy selected images and labels to subset directory
    for img_file in train_images_subset:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        shutil.copyfile(os.path.join(original_dir, 'train', 'images', img_file),
                        os.path.join(subset_dir, 'train', 'images', img_file))
        shutil.copyfile(os.path.join(original_dir, 'train', 'labels', label_file),
                        os.path.join(subset_dir, 'train', 'labels', label_file))

    for img_file in val_images_subset:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        shutil.copyfile(os.path.join(original_dir, 'val', 'images', img_file),
                        os.path.join(subset_dir, 'val', 'images', img_file))
        shutil.copyfile(os.path.join(original_dir, 'val', 'labels', label_file),
                        os.path.join(subset_dir, 'val', 'labels', label_file))

# Example usage:
original_dataset_dir = 'datasets/17R_dataset'
subset_dataset_dir = 'datasets/17R_dataset_small'
num_images_to_take = 50  # Change this according to your preference

create_subset_dataset(original_dataset_dir, subset_dataset_dir, num_images_to_take)
