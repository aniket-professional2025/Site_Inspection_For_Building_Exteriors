# Importing required Packages
import numpy as np
import cv2

# Define function to pre-process an image
def preprocess_image(image_path):

    # Load the Image
    try:
        image = cv2.imread(image_path)
    except Exception as e:
        raise FileNotFoundError(f"Image Not Found In {image_path}")

    # Convert to Gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Blurred Operation
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    return image, gray, blurred

# Detecting Flaking and Peeling as a criteria to judge adhesion
def segment_defects(blurred_image, min_defect_area: int):

    # Apply Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Performing Morphological Operations: Opening and Closing
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1)

    # Find Contours (Potential Defect Area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the final, filtered defects
    defect_mask = np.zeros_like(mask)

    # Filter out very small contours (noise)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_defect_area:
            cv2.drawContours(defect_mask, [contour], -1, 255, -1)

    return defect_mask

# Function to Quantification and Scoring
def calculate_adhesion_score(image_bgr, defect_mask):
    
    # Total area of the wall section being analyzed (in pixels)
    total_area = image_bgr.shape[0] * image_bgr.shape[1]

    # Calculate the total defect area (sum of white pixels in the mask)
    defect_area = np.sum(defect_mask > 0)

    # Calculate the percentage of the area that is defective
    defect_percentage = (defect_area / total_area) * 100

    # Map the defect percentage to an estimated adhesion score (0 to 5)
    if defect_percentage <= 2: 
        score = 5
    elif defect_percentage <= 5:
        score = 4
    elif defect_percentage <= 15:
        score = 3
    elif defect_percentage <= 35:
        score = 2
    elif defect_percentage <= 65:
        score = 1
    else:
        score = 0

    return defect_percentage, score

# The Complete Main Function
def estimate_adhesion_score_from_image(image_path, min_area: int = 50):
    
    try:
        # Preprocess
        img_bgr, _ , img_blurred = preprocess_image(image_path)

        # 2. Segment Defects
        defect_mask = segment_defects(img_blurred, min_defect_area = min_area)

        # 3. Quantification and Scoring
        percentage, score = calculate_adhesion_score(img_bgr, defect_mask)

        # Visualize the detected defects
        cv2.imshow('Defect Mask', defect_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"--- Estimated Adhesion Result ---")
        print(f"Defective Area: {percentage:.2f}%")
        print(f"Estimated Adhesion Score (0-5): {score}")
        print("---------------------------------")
        return score, percentage

    except FileNotFoundError as e:
        print(e)
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example Usage:
if __name__ == "__main__":
    image_path = r""
    estimated_score, area_percent = estimate_adhesion_score_from_image(image_path = image_path, min_area = 100)
    print("The Estimated Score is:", estimated_score)
    print("The Area Percentage is:", area_percent)