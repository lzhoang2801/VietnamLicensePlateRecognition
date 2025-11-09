import cv2
import numpy as np

def deskew(image, image_size: tuple[int, int]) -> np.ndarray:
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()

    skew = m['mu11'] / m['mu02']

    M = np.float32([[1, skew, -0.5*image_size[0]*skew], [0, 1, 0]])

    image = cv2.warpAffine(image, M, image_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return image

def contrast_enhancement(image):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rect_kernel)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rect_kernel)
    
    return cv2.subtract(cv2.add(image, tophat), blackhat)