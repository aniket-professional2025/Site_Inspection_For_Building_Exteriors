# Importing Required Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the function to Detect, Segment and Calculate Area of Fungul growth
def get_fungus_segmentation(image_path: str, min_area: float = 500, output_path: str = None):

    # Load the image
    try:
        image_bgr = cv2.imread(image_path)
    except Exception as e:
        raise FileNotFoundError(f"No Image Found in {image_path}")
    
    # Make a copy for drawing the final result
    img_result = image_bgr.copy()

    # Convert the image into gray scale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur on the Gray image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply Thresholding on blured image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perorming Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Display the intermediate binary mask
    cv2.imshow('Binary Mask (White=Fungus)', mask)

    # Finding the Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Configuring the fungul area
    total_fungal_area = 0.0

    # 7. Process Contours
    for i, contour in enumerate(contours):
        # Calculate the area of the current contour
        area = cv2.contourArea(contour)

        # Filter out contours that are too small (likely noise)
        if area > min_area:
            total_fungal_area += area

            # Draw the contour on the result image for visualization
            cv2.drawContours(img_result, contours, i, (0, 255, 0), 4)
    
    # --- Analysis and Output ---
    print(f"--- Fungal Growth Analysis Results ---")
    
    # Image dimensions and total pixel count
    height, width, _ = image_bgr.shape
    total_pixels = height * width
    print(f"Image Dimensions: {width} x {height} pixels")
    print(f"Total Wall Surface Area (in pixels): {total_pixels}")
    
    # Total Fungal Area
    print(f"Total Fungal Area (in pixels): {int(total_fungal_area)}")
    
    # Calculate percentage of coverage
    percentage = (total_fungal_area / total_pixels) * 100
    print(f"Approximate Wall Coverage: {percentage:.2f}%")

    # Save the image in output directory
    if output_path is None:
        pass
    else:
        cv2.imwrite(output_path, img_result)

    # Visualize the Results
    cv2.imshow('Segmented Fungus', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Fungal Segmentation complete.")

# Example Usage
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\FungusImage.png"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\OutputImages\Detected_Fungus.png"
    get_fungus_segmentation(image_path = image_path, min_area = 500, output_path = output_path)