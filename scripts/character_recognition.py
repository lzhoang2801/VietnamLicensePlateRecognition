import cv2
import numpy as np
from scripts.image_preprocessing import deskew
from scripts.feature_extraction import extract_hog_features

import pickle
model = pickle.load(open("models/hog_svm.pkl", "rb"))

def class_to_label(x):
    if 0 <= x <= 9:
        return str(x)
    elif 10 <= x <= 35:
        return chr(ord('A') + x - 10)
    else:
        return str(x)

def predict_characters(char_images):
    if isinstance(char_images, str):
        char_images = [char_images]
    characters = []
    for char_image in char_images:
        char_image = cv2.resize(char_image, (28, 28))
        char_image = deskew(char_image, (28, 28))
        features = extract_hog_features(char_image)
        prediction = model.predict([features])
        characters.append(class_to_label(prediction[0]))
    return characters

def segment_characters(plate_region):
    thresh = cv2.adaptiveThreshold(plate_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_candidates = []
    plate_height, _ = plate_region.shape[:2]

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        height_ratio = h / float(plate_height)
        
        if 0.15 < aspect_ratio < 1.2 and 0.3 < height_ratio < 0.95:
            char_candidates.append((x, y, w, h))

    return sorted(char_candidates, key=lambda b: (b[0], b[1]))

def format_character(char_crop, target_dim=28):
    _, char_binary = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    h, w = char_binary.shape
    
    pad_color = 0
    
    if w > h:
        new_w = target_dim - 8
        new_h = int(h * (new_w / w))
    else:
        new_h = target_dim - 8
        new_w = int(w * (new_h / h))
        
    resized = cv2.resize(char_binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.full((target_dim, target_dim), pad_color, dtype=np.uint8)
    
    pad_y = (target_dim - new_h) // 2
    pad_x = (target_dim - new_w) // 2
    
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return padded

def character_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    char_bounding_boxes = segment_characters(gray_image)
    
    if not char_bounding_boxes:
        return image, []

    annotated_plate = image.copy()
    segmented_characters = []

    for (x, y, w, h) in char_bounding_boxes:
        cv2.rectangle(annotated_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
        char_crop = gray_image[y:y+h, x:x+w]
        formatted_char = format_character(char_crop)
        segmented_characters.append(formatted_char)
        
    return annotated_plate, segmented_characters