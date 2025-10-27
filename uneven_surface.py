# Importing Required Packages
import numpy as np
import cv2

# Define function to detect Uneven Surface, detect, segment and calculate area
def find_uneven_surface(image_path, seg_save_path: str, bound_save_path: str):

    # Load the image
    try:
        image = cv2.imread(image_path)
        cv2.imshow("Original Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        raise FileNotFoundError(f"Image not Found in {image_path}")

    # Converting the Image into gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Scale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Applying Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imshow("Blurred Image", blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("Canny Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Contour Finding and Segmentation
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the Number of Detected Contours
    num_contours = len(contours)
    print(f"Total {num_contours} Contours are Detected in the Image")

    # Initialize variables for total area
    total_area = 0

    # Create a mask to segment ALL areas
    mask = np.zeros_like(gray, dtype = np.uint8)
    
    # Iterate through ALL found contours to fill the mask and sum the areas: Check if any contours were found
    if contours:

        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

        for contour in contours:
            total_area += cv2.contourArea(contour)
        
        segmented_image = cv2.bitwise_and(image, image, mask=mask)

        print("===============================================================") 
        print(f"All Uneven Surfaces Detected and Segmented successfully.")
        print(f"Calculated Total Area of Uneven Surfaces (in pixel units): {total_area:.2f}")
        print("===============================================================")

        # Display the segmented image
        print("Segmented Uneven Surface (All Regions):")

        cv2.imshow("Segmented Region", segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Saving the Segmented Image
        cv2.imwrite(seg_save_path, segmented_image)

        # Display the boundary drawn on the original image (Optional)
        boundary_image = image.copy()
    
        cv2.drawContours(boundary_image, contours, -1, (0, 255, 0), 2) 
        print("Detected Boundary (All Regions):")

        cv2.imshow("Boundary Image", boundary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the Boundary Image
        cv2.imwrite(bound_save_path, boundary_image)

    else:
        print("Could not detect any significant contours. Adjust parameters (e.g., Canny thresholds, blur kernel).")

# Example Usage
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\Uneven_Surface_Image.png"
    seg_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\OutputImages\Uneven_Surface_Image_Segment.png"
    bound_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\OutputImages\Uneven_Surface_Image_Boundary.png"
    find_uneven_surface(image_path, seg_path, bound_path)