import cv2
import os
import random
import glob

def select_images(dataset_dir, save_dir, num_images=100):
    image_files = glob.glob(os.path.join(dataset_dir, "*.png"))

    print(f"Found {len(image_files)} images in {dataset_dir}")

    random.shuffle(image_files)

    for image_file in image_files[:num_images]:
        image = cv2.imread(image_file)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_file)), image)

if __name__ == "__main__":
    dataset_dir = "dataset/licenseplates/images/train"
    save_dir = "dataset/sampleplates"

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist")
        exit(1)

    os.makedirs(save_dir, exist_ok=True)

    num_images = 10
    select_images(dataset_dir, save_dir, num_images)

    print(f"Selected {num_images} images from {dataset_dir} and saved to {save_dir}")