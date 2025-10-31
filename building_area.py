# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

# Setting the Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Function to detect windows using image_path and threshold
def building_information(image_path: str, detection_threshold: float, real_building_height: int = 40, output_path = None):

    print("[DEBUG] Inside the Main Function to Building Detection")
    
    # Reading the Image
    try:
        image = Image.open(image_path).convert("RGB")
        # print("[DEBUG] Inside the Function: Image Loaded Successfully")
    except Exception:
        raise FileNotFoundError("No Image found")
    
    # Setting the Text Labels
    text_labels = [["building"]]

    # Feeding the Inputs to the Model
    inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Generating the results
    results = processor.post_process_grounded_object_detection(outputs, threshold = detection_threshold, target_sizes = [(image.height, image.width)])

    # Extracting the first result
    result = results[0]
    # print("[DEBUG] The results are Achieved")

    # Getting the number of detections
    detections = len(result["boxes"])
    # print(f"[DEBUG] Total {detections} buildings are detected using threshold {detection_threshold}")

    # Fallbacks if no window detected
    if detections == 0:
        print("[DEBUG] No Building detected. Exiting the Function")
        return

    # Drawing the Detections on that Image
    draw = ImageDraw.Draw(image)

    # Get the box coordinate and its height and width
    box = result['boxes'][0].tolist()
    height_px = abs(box[3] - box[1])
    width_px = abs(box[2] - box[0])

    # The detected box on the image
    draw.rectangle(box, outline = 'red', width = 3)
    
    # Calculate the scale factor
    scale_factor = real_building_height / height_px

    # Calculate the Actual Height and Actual Width
    height_ft = real_building_height
    width_ft = scale_factor * width_px

    # Calculate the Area of the building in feet
    area_ft = height_ft * width_ft

    # Putting Text on the Image
    image_np = np.array(image)
    cv2.putText(image_np, f"Height : {height_ft} Ft", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_np, f"Width : {np.round(width_ft, 2)} Ft", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_np, f"Area : {np.round(area_ft, 2)} Sq. Ft", (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Printing the results
    # print("The bounding Box Coordinates are:", result['boxes'].tolist()[0])
    # print(f"The Height and Width of the Buidling are: {height_ft} feet and {width_ft} feet")
    # print(f"The Area of the Building is: {area_ft} sq. feet")

    # Converting the Image back in PIL format and showing it
    image = Image.fromarray(image_np)

    # Saving the image in the specified output path
    if output_path is None:
        pass
    else: 
        image.save(output_path)
        print(f"The Annotated Image is Saved in: {output_path}")

    # Showing the Image
    # image.show()

    # Return the required building information
    return height_ft, np.round(width_ft, 2) , np.round(area_ft, 2)

# # Inference
# image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\OrgImages\Image_21.jpg"
# output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Height_Width\Images\Modify\Image_21_Building_Area.jpg"
# building_information(image_path, detection_threshold = 0.4, real_building_height = 23, output_path = output_path)