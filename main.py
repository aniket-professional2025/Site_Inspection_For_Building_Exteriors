# Importing Required Packages
import numpy as np
import cv2
from PIL import Image
from building_area import building_information
from objects_area import detect_objects
from defect_detection import get_defects

# Define Function to Calculate Building Paintable Area
def calculate_paintable_area(image1_path: None, image2_path: None, image3_path: None, image4_path: None, 
                             building_threshold: float = 0.4, 
                             object_threshold: float = 0.4,
                             real_building_height: float = 40.0,
                             real_window_height: float = 3.5,
                             real_door_height: float = 7.0,
                             real_pipe_height: float = 20.0):

    # Getting the Height Width Area of the Buildings
    h1, w1, a1 = building_information(image1_path, detection_threshold = building_threshold, real_building_height = real_building_height)
    h2, w2, a2 = building_information(image2_path, detection_threshold = building_threshold, real_building_height = real_building_height)
    h3, w3, a3 = building_information(image3_path, detection_threshold = building_threshold, real_building_height = real_building_height)
    h4, w4, a4 = building_information(image4_path, detection_threshold = building_threshold, real_building_height = real_building_height)

    # Getting the Final Height, Width and Area of the Buildings
    final_h = real_building_height
    final_w = np.median([w1, w2, w3, w4])
    final_a = np.median([a1, a2, a3, a4])
   
    # Getting the area of window, door, pipe and total area
    window_area_1, door_area_1, pipe_area_1, total_area_1, wc1, dc1, pc1 = detect_objects(image1_path, real_height_pipes_feet = real_pipe_height, real_height_window_feet = real_window_height, real_height_door_feet = real_door_height, threshold = object_threshold)
    window_area_2, door_area_2, pipe_area_2, total_area_2, wc2, dc2, pc2 = detect_objects(image2_path, real_height_pipes_feet = real_pipe_height, real_height_window_feet = real_window_height, real_height_door_feet = real_door_height, threshold = object_threshold)
    window_area_3, door_area_3, pipe_area_3, total_area_3, wc3, dc3, pc3 = detect_objects(image3_path, real_height_pipes_feet = real_pipe_height, real_height_window_feet = real_window_height, real_height_door_feet = real_door_height, threshold = object_threshold)
    window_area_4, door_area_4, pipe_area_4, total_area_4, wc4, dc4, pc4 = detect_objects(image4_path, real_height_pipes_feet = real_pipe_height, real_height_window_feet = real_window_height, real_height_door_feet = real_door_height, threshold = object_threshold)

    # Getting the Final Area of Windows, Doors, Pipes and Total area
    final_window_area = np.median([window_area_1, window_area_2, window_area_3, window_area_4])
    final_door_area = np.median([door_area_1, door_area_2, door_area_3, door_area_4])
    final_pipe_area = np.median([pipe_area_1, pipe_area_2, pipe_area_3, pipe_area_4])
    final_wdp_area = np.median([total_area_1, total_area_2, total_area_3, total_area_4])
    window_count = np.median([wc1, wc2, wc3, wc4])
    door_count = np.median([dc1, dc2, dc3, dc4])
    pipe_count = np.median([pc1, pc2, pc3, pc4])

    # The total paintable area
    final_paintable_area = final_a - final_wdp_area

    print("======================== The Function Runs Completely ==========================")

    # Returning the outcomes
    return final_h, final_w, window_count, door_count, pipe_count, final_a, final_wdp_area, final_paintable_area

# Function to detect the defects on the building wall images
def find_defects(image1_path: None, image2_path: None, image3_path: None, image4_path: None, defect_threshold: float = 0.35):
    
    # Store all image paths in a list for iteration
    image_paths = [image1_path, image2_path, image3_path, image4_path]
    all_labels = []

    for idx, path in enumerate(image_paths, 1):
        if path:  # Ensure path exists
            print(f"\n[INFO] Processing Image {idx}: {path}")
            labels = get_defects(image_path=path, detection_threshold=defect_threshold) or []
            print(f"[INFO] Detected in Image {idx}: {labels}")
            all_labels.extend(labels)
        else:
            print(f"[WARNING] Image {idx} path not provided. Skipping.")

    # Filter out empty or invalid labels
    all_labels = [lbl.strip().lower() for lbl in all_labels if isinstance(lbl, str) and lbl.strip()]

    # Deduplicate labels while preserving order
    final_labels = list(dict.fromkeys(all_labels))

    print(f"\n[RESULT] Final Combined Defect Labels: {final_labels}")
    return final_labels

# def find_defects(image1_path, image2_path, image3_path, image4_path, defect_threshold: float = 0.3):

#     # Getting the defection labels
#     labels_1 = get_defects(image1_path, detection_threshold = defect_threshold) or []
#     labels_2 = get_defects(image2_path, detection_threshold = defect_threshold) or []
#     labels_3 = get_defects(image3_path, detection_threshold = defect_threshold) or []
#     labels_4 = get_defects(image4_path, detection_threshold = defect_threshold) or []

#     # Making the final defects label
#     final_labels = labels_1 + labels_2 + labels_3 + labels_4

#     # Return the final labels
#     final_labels = [lbl for lbl in final_labels if isinstance(lbl, str) and lbl.strip()]

#     return final_labels

# # Inference on the above function
# if __name__ == "__main__":
#     image1 = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\Images\OrgImages\Image_3.jpg"
#     image2 = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\Images\OrgImages\Image_3.jpg"
#     image3 = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\Images\OrgImages\Image_3.jpg"
#     image4 = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\Images\OrgImages\Image_3.jpg"
#     bh, bw, wc, dc, pc, total_building_area, total_wdp_area, paintable_area = calculate_paintable_area(image1_path = image1, 
#                                                           image2_path = image2, 
#                                                           image3_path = image3, 
#                                                           image4_path = image4,
#                                                           building_threshold = 0.4, 
#                                                           object_threshold = 0.3,
#                                                           real_building_height = 25.0,
#                                                           real_window_height = 3.5,
#                                                           real_door_height = 7.0,
#                                                           real_pipe_height = 20.0)
    
#     print(f"The Height of the building is {bh} Sq. Feet")
#     print(f"The Width of the building is {bw} Sq. Feet")
#     print(f"Total {wc} Windows are detcetd")
#     print(f"Total {dc} Doors are detected")
#     print(f"Total {pc} Pipes are detected")
#     print(f"The building has total area of {total_building_area} Sq. Feet")
#     print(f"Total Area of Doors, Windows and Pipes are {total_wdp_area} Sq. Feet")
#     print(f"Total Paintable Area is {paintable_area} Sq. Feet")