import cv2

def extract_hog_features(image):
    hog = cv2.HOGDescriptor(
        _winSize=(28, 28),
        _blockSize=(14, 14),
        _blockStride=(7, 7),
        _cellSize=(14, 14),
        _nbins=9,
        _gammaCorrection=True,
        _signedGradient=True
    )

    return hog.compute(image)