import cv2
import numpy as np

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = cv2.rotate(arr, angle)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Skew correction
    angle, rotated = correct_skew(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Median blur
    filter1 = cv2.medianBlur(gray, 5)

    # Gaussian blur
    filter2 = cv2.GaussianBlur(filter1, (5, 5), 0)

    # Denoising
    dst = cv2.fastNlMeansDenoising(filter2, None, 17, 7, 21)

    # Binarization using Otsu's thresholding
    _, binarized = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the processed image
    preprocessed_path = 'uploads/preprocessed_image.png'
    cv2.imwrite(preprocessed_path, binarized)

    return preprocessed_path
