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
    
    # Resize the Image
    image = cv2.resize(image, (400,400))

    # Convert to Gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Blurred Operation
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    return image, gray, blurred

# Detecting Flaking and Peeling as a criteria to judge adhesion
def segment_defects(image_bgr, min_defect_area: int):

    # Apply HSV color model
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Saturation/Value should be high.
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for the intact blue paint
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # The mask will have white where paint is missing (defects) and black where paint is present (blue_mask).
    initial_defect_mask = cv2.bitwise_not(blue_mask)

    # Visualize the Intial Defect Mask
    cv2.imshow("Initial Defect Mask", initial_defect_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Refine Performing Morphological Operations: Opening and Closing
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(initial_defect_mask, cv2.MORPH_CLOSE, kernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Find Contours (Potential Defect Area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the final, filtered defects
    defect_mask = np.zeros_like(mask)

    # Filter out very small contours (noise)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_defect_area:
            cv2.drawContours(defect_mask, [contour], -1, 255, -1)

    # Visualize the Defected Mask
    cv2.imshow("Defect Mask", defect_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return defect_mask

# Function to Quantification and Scoring
def calculate_adhesion_score(image_bgr, defect_mask):
    
    # Total area of the wall section being analyzed (in pixels)
    total_area = image_bgr.shape[0] * image_bgr.shape[1]

    # Calculate the total defect area (sum of white pixels in the mask)
    defect_area = np.sum(defect_mask > 0)

    # Calculate the percentage of the area that is defective
    defect_percentage = (defect_area / total_area) * 100

    # Calculate the Adhesion Percentage Score
    adhesion_percentage_score = 100.0 - defect_percentage

    return np.round(total_area, 2), np.round(defect_area,2),  np.round(defect_percentage,2), np.round(adhesion_percentage_score,2)

# The Complete Main Function
def estimate_adhesion_score_from_image(image_path, output_path: str, min_area: int = 50):
    
    try:
        # Preprocess
        image_bgr, image_gray , image_blurred = preprocess_image(image_path)

        # Visualize the Original Image
        cv2.imshow("Original Image", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Visualize the Gray Scale Image
        cv2.imshow("Gray Scale Image", image_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Visualize the Blurred Image
        cv2.imshow("Blurred Image", image_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 2. Segment Defects
        defect_mask = segment_defects(image_bgr, min_defect_area = min_area)

        # Store the Detected_mask
        cv2.imwrite(output_path, defect_mask)

        # 3. Quantification and Scoring
        total_area, defetcted_area, percentage, score = calculate_adhesion_score(image_bgr, defect_mask)

        
        print(f"Total Image Area is {total_area}")
        print(f"Estimated Detected Region Area is {defetcted_area}")

        print(f"--- Estimated Adhesion Result ---")
        print(f"Defective Area: {percentage:.2f}%")
        print(f"Estimated Adhesion Score (0-100): {score}")
        print("---------------------------------")
        return total_area, defetcted_area, percentage, score

    except FileNotFoundError as e:
        print(e)
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example Usage:
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\Adhesion_Test_Image.png"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\OutputImages\Adhesion_Test_Image_Defected_Mask.png"
    defected_area, area_percent, estimated_score = estimate_adhesion_score_from_image(image_path = image_path, output_path = output_path, min_area = 100)
    print("Defected Area is:", defected_area)
    print("The Area Percentage is:", area_percent)
    print("The Estimated Adhesion Score is:", estimated_score)
    