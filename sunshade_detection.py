# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from torchvision.ops import nms

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"
print(f"========== The model: {model_id} is Set Successfully =============")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=========== The {device} is The Suitable Device ==================")

processor = AutoProcessor.from_pretrained(model_id)
print("[DEBUG] The processor is loaded Successfully")

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("[DEBUG] The Model is Loaded Successfully")

# Define functions to get Filtered Results
def filter_by_area(result, area_range_factor = (0.5, 2.0)):

    # Fetching the Boxes, Scores and Labels
    boxes = result['boxes']
    scores = result['scores']
    labels = result['labels']

    # Check for Number of Boxes
    if len(boxes) == 0:
        return result
    
    # Calculate area of all boxes
    box_array = np.array([box.tolist() for box in boxes])
    widths = box_array[:, 2] - box_array[:, 0]
    heights = box_array[:, 3] - box_array[:, 1]
    areas = widths * heights

    # Calculate the median area of all detections
    median_area = np.median(areas)

    # Determine the acceptable range
    min_area = median_area * area_range_factor[0]
    max_area = median_area * area_range_factor[1]

    # Create a mask for boxes within the acceptable area range
    area_mask = (areas >= min_area) & (areas <= max_area)

    # Apply the mask to filter results
    filtered_result = {
        "boxes": [boxes[i] for i, mask in enumerate(area_mask) if mask],
        "scores": [scores[i] for i, mask in enumerate(area_mask) if mask],
        "labels": [labels[i] for i, mask in enumerate(area_mask) if mask]
    }

    # Print statistics for debugging
    print(f"[DEBUG] Median Area: {median_area:.2f}. Range: {min_area:.2f} to {max_area:.2f}")
    print(f"[DEBUG] Filtered {len(boxes) - len(filtered_result['boxes'])} boxes.")
    
    return filtered_result

# Define functions to get the detections
def get_detections(image_path, output_path = None, detection_threshold: float = 0.5):

    print("[DEBUG] Inside the Main Function")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Image not Found in {image_path}")
    
    # Setting the object name to find
    text_labels = [["window and sun shade"]] 

    # Creating the complete Input
    inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    print("[DEBUG] Generating the Results")

    results = processor.post_process_grounded_object_detection(outputs, threshold = detection_threshold, target_sizes = [(image.height, image.width)])
    result = results[0]

    # Apply Filtering (optional, but keep it for robustness)
    if len(result['boxes']) > 1:
        result = filter_by_area(result, area_range_factor = (0.4, 3.0))

    detected_count = len(result["boxes"])
    if detected_count == 0:
        print("No objects detected after filtering. Exiting.")
        return
    
    # ====================================================================
    #           NEW STEP: MANIPULATE SCORES TO FAVOR LARGER BOXES
    # ====================================================================
    
    # Convert boxes and scores to Tensors
    boxes_tensor = torch.stack(result["boxes"]).to(device)
    scores_tensor = torch.tensor(result["scores"]).to(device)

    # 1. Calculate area of each box
    # Boxes are [x_min, y_min, x_max, y_max]
    widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
    heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
    areas = widths * heights

    # 2. Normalize areas to create a weight (e.g., scale between 0.1 and 1.0)
    # Adding a small constant to prevent division by zero or extreme scaling
    min_area = torch.min(areas)
    max_area = torch.max(areas)
    
    # Create a scaling factor that is larger for larger boxes
    # This factor will range from 1.0 (for min area) up to a max_factor (for max area)
    max_factor = 3.5 # A subtle boost of 5% is usually enough
    
    # Linear scaling: 
    # factor = 1.0 + (areas - min_area) / (max_area - min_area) * (max_factor - 1.0)
    # Using a simple multiplicative factor based on normalized area:
    area_weights = 1.0 + (areas - min_area) / (max_area - min_area) * (max_factor - 1.0)
    
    # 3. Apply the area weight to the original scores
    adjusted_scores = scores_tensor * area_weights
    
    print(f"[DEBUG] Applied area-based score boost to favor outer boxes (max factor: {max_factor}).")
    # ====================================================================

    # Define Intersection over Union (IoU) threshold
    # Use the value that worked for you (0.2) or slightly lower for strict filtering
    iou_threshold = 0.2

    # Apply NMS using the ADJUSTED scores
    # This forces the NMS to select the larger box among highly overlapping candidates
    keep = nms(boxes_tensor, adjusted_scores, iou_threshold) 

    # Filter the results using the indices from NMS
    nms_result = {
        # Note: We keep the original scores for display/logging, but NMS used adjusted_scores
        "boxes": [result["boxes"][i] for i in keep],
        "scores": [result["scores"][i] for i in keep],
        "labels": [result["labels"][i] for i in keep]
    }

    print(f"Total {detected_count} objects detected after filtering.")
    print(f"[DEBUG] Reduced from {detected_count} to {len(nms_result['boxes'])} boxes after NMS with score boost.")

    # Use nms_result from here on
    final_result = nms_result

    final_count = len(final_result["boxes"])
    if final_count == 0:
        print("No objects remaining after NMS. Exiting.")
        return
    
    print(f"Total {final_count} unique objects detected after NMS.")

    print("[DEBUG] Getting the Detections")

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)

    for box, score, label in zip(final_result["boxes"], final_result["scores"], final_result["labels"]):
        box_coords = [round(x, 2) for x in box.tolist()]
        score_val = round(score.item(), 3)
        label_text = f"window_w_shade ({score_val})" 

        print(f"Detected {label_text} with confidence {score_val} at location {box_coords}")

        # Draw rectangle and label
        draw.rectangle(box_coords, outline = "red", width = 3) 
        draw.text((box_coords[0] + 5, box_coords[1] + 5), label_text, fill = "red")

    # Save the Image in Mentioned output path
    if output_path is None:
        pass
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"The detected Image is Saved at {output_path}")

    # Showing the image
    image.show()

# Inference on the Function
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Paintable_Area_Calculation\Images\OrgImages\Image_8.jpg"
    output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Paintable_Area_Calculation\Images\Modify\Advanced_Detected_SunShade.jpg"
    get_detections(image_path = image_path, output_path = output_path, detection_threshold = 0.1)
