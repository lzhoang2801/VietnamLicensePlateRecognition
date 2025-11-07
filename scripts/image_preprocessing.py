import cv2
import numpy as np

def find_plate_candidates(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_yellow = np.array([0, 133, 77])
    upper_yellow = np.array([255, 173, 127])
    
    yellow_mask = cv2.inRange(ycrcb_image, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / float(h)
        
        min_aspect_ratio = 2.5
        max_aspect_ratio = 5.5
        min_area = 1500
        
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio and w * h > min_area:
            candidate_regions.append(image[y:y+h, x:x+w])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return candidate_regions, image