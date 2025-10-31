# Importing Required Packages
from objects_area import detect_objects
from building_area import building_information
import cv2
import numpy as np

# Define the function to get the complete results
def measure_paintable_area(image_path: str, building_threshold: float = 0.4, object_threshold: float = 0.4, real_pipe_height: float = 20, real_building_height: float = 40.0, real_height_window_feet: float = 3.5, real_height_door_feet: float = 7.0, output_path = None):

    # The Function to Fetch the Building Information
    height_ft, width_ft, area_ft = building_information(image_path = image_path, detection_threshold = building_threshold, real_building_height = real_building_height)
    print(f"The Height of the Building is {height_ft} Feet, The Width is {width_ft} Feet and Area is {area_ft} Square Feet")

    # The function to Fetch the Area of Objects (Widnows and Doors)
    window_area, door_area, pipe_area, object_area = detect_objects(image_path = image_path, real_height_pipes_feet = real_pipe_height, real_height_window_feet = real_height_window_feet, real_height_door_feet = real_height_door_feet, threshold = object_threshold)
    print(f"Window Area is: {window_area} Sq.Feet, Door Area is: {door_area} Sq.Feet, Total Area is: {object_area} Sq.Feet")

    # Calculate Total Paintable Area
    paintable_area = np.round((area_ft - object_area), 2)

    # Annotate and Save Final Image
    print("\nAnnotating and Saving Final Image")
    try:
        img_np = cv2.imread(image_path)
        if img_np is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception:
        raise FileNotFoundError("Could not load image for final annotation.")
    
    # Define Annotation Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255) # Red color for text
    thickness = 2
    
    # Text to be written on the image
    text_lines = [
        f"Estimated Height: {height_ft} Feets",
        f"Estimated Width: {width_ft} Feets",
        f"Total Building Area: {area_ft} Sq. Feet",
        f"Total Window Area: {window_area} Sq. Feet",
        f"Total Door Area: {door_area} Sq. Feet",
        f"Total Pipe Area: {pipe_area} Sq. Feet", 
        f"Total Objects Area: {object_area} Sq. Feet",
        f"Paintable Area: {paintable_area} Sq. Feet"
    ]
    
    # Put text on image (defining the Coordinates)
    y_start = 40
    line_spacing = 30
    for i, line in enumerate(text_lines):
        cv2.putText(img_np, line, (10, y_start + i * line_spacing), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the final image
    if output_path:
        cv2.imwrite(output_path, img_np)
        print(f"Final annotated image saved to: {output_path}")
        
    # Display the image (optional/comment out if running in headless environment)
    cv2.imshow("Paintable Area Result", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return Results
    return height_ft, width_ft, area_ft, window_area, door_area, pipe_area, object_area, paintable_area

# Inference on the Function
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\OrgImages\Image_21.jpg"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\Modify\Image_21_Final_Result.jpg"
    height_ft, width_ft, area_ft, window_area, door_area, pipe_area, object_area, paintable_area = measure_paintable_area(image_path = image_path, building_threshold = 0.4, object_threshold = 0.31, real_pipe_height = 20, real_building_height = 23, real_height_window_feet = 3.5, real_height_door_feet = 7, output_path = output_path)

    print(f"The Actual Height is: {height_ft} Feet")
    print(f"The Actual Width is: {width_ft} Feet")
    print(f"The Total Building Area is: {area_ft} Sq. Feet")
    print(f"Total Windows Area is: {window_area}")
    print(f"Total Doors Area is: {door_area}")
    print(f"Total Objects Area is: {object_area}")
    print(f"Total Paintable Area is: {paintable_area}")