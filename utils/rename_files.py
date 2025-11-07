import cv2
import os

def rename_files(directory):
    count = 0
    for file in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, file))
        if image is None:
            continue
        
        new_name = f"{count:04d}.png"
        cv2.imwrite(os.path.join(directory, new_name), image)
        os.remove(os.path.join(directory, file))
        count += 1
    print(f"Renamed {count} files")

if __name__ == "__main__":
    directory = "dataset/motorbike/"
    rename_files(directory)