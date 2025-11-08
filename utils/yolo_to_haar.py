import cv2
import os
import glob

def get_haar_annotations(image_path, label_path):
    image = cv2.imread(image_path)
    label = open(label_path, 'r').read()

    class_bboxes = {}

    for line in label.split('\n'):
        if not line:
            continue

        data = line.split()
        class_id = int(data[0])

        if class_id not in class_bboxes:
            class_bboxes[class_id] = []

        coords = []
        for i in range(1, len(data), 2):
            x, y = float(data[i]) * image.shape[1], float(data[i+1]) * image.shape[0]
            coords.append((x, y))

        min_x = min(coords[0][0], coords[2][0]) - 2
        min_y = min(coords[0][1], coords[2][1]) - 2
        max_x = max(coords[1][0], coords[3][0]) + 2
        max_y = max(coords[1][1], coords[3][1]) + 2
        width = max_x - min_x
        height = max_y - min_y

        class_bboxes[class_id].append(f"{int(min_x)} {int(min_y)} {int(width)} {int(height)}")

    return class_bboxes

def write_haar_data(haar_data, output_path):
    with open(output_path, 'a') as f:
        for data in haar_data:
            f.write(f"{data}\n")

def convert_yolo_to_haar_dataset(image_dir, label_dir, output_path, positive_sample=True):
    image_files = glob.glob(os.path.join(image_dir, "*.png"))

    dataset = []

    for image_file in image_files:
        label_file = os.path.join(label_dir, os.path.basename(image_file).replace('.png', '.txt'))
        if positive_sample and os.path.exists(label_file):
            dataset.append((image_file, label_file))
        else:
            dataset.append((image_file, None))

    print(f"Found {len(dataset)} images in {image_dir}")

    for image_file, label_file in dataset:
        if not positive_sample:
            output_file = os.path.join(output_path, "negative.dat")
            negative_line = []
            negative_line.append(image_file)
            write_haar_data(negative_line, output_file)
            continue

        class_bboxes = get_haar_annotations(image_file, label_file)
        for class_id, bboxes in class_bboxes.items():
            output_file = os.path.join(output_path, "positive_{}.dat".format(class_id))
            positive_line = []
            positive_line.append("{} {} {}".format(image_file, len(bboxes), " ".join(bboxes)))
            write_haar_data(positive_line, output_file)

    print("Done converting YOLO to Haar dataset")

if __name__ == "__main__":
    image_dir = "dataset/motorbike"
    label_dir = "dataset/motorbike"
    output_path = "dataset/licenseplates"
    convert_yolo_to_haar_dataset(image_dir, label_dir, output_path, positive_sample=False)