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

# Define Function to get different detections
def detect_objects(image_path: str, real_height_pipes_feet: float = 20, real_height_window_feet: float = 3.5, real_height_door_feet: float = 7.0, threshold: float = 0.4, output_path = None):

    
    print("[DEBUG] Inside the Main Function for Objects Detection")
    
    
    # --- Load Image ---
    try:
        image = Image.open(image_path).convert("RGB")
        # print("Image Loaded Successfully")
    except Exception:
        raise FileNotFoundError("No Image found at:", image_path)

    # --- Prepare Labels ---
    text_labels = [["windows", "doors", "pipes"]]
    # print("[DEBUG] Using text labels:", text_labels)

    # --- Model Input ---
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # --- Postprocess Detections ---
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
    )[0]

    detected_count = len(results["boxes"])
    if detected_count == 0:
        print("No objects detected. Exiting.")
        return

    # print(f"Total {detected_count} objects detected at threshold {threshold}")

    draw = ImageDraw.Draw(image)

    # --- Initialize Accumulators ---
    total_area_all = 0.0
    object_areas = {"windows": 0.0, "doors": 0.0, "pipes": 0.0}
    object_counts = {"windows": 0, "doors": 0, "pipes": 0}
    scale_factors = {"windows": None, "doors": None, "pipes": None}
    real_heights = {"windows": real_height_window_feet, "doors": real_height_door_feet, "pipes": real_height_pipes_feet}

    # --- Process Each Detection ---
    for i, (box, score, label_id) in enumerate(zip(results["boxes"], results["scores"], results["labels"]), 1):
        box = [float(x) for x in box.tolist()]

        # Convert or clean label
        if isinstance(label_id, str):
            label_text = label_id.strip().lower()
        elif isinstance(label_id, int) and  0 <= label_id < len(text_labels[0]):
            label_text = text_labels[0][label_id].lower()
        else:
            label_text = ""

        # Skip if label not one of expected
        if not label_text or label_text not in object_areas:
            continue

        confidence = float(score.item())

        # print(f"\nDetected {label_text} with confidence {confidence:.2f} at {box}")

        # Pixel dimensions
        x_min, y_min, x_max, y_max = box
        width_px = abs(x_max - x_min)
        height_px = abs(y_max - y_min)

        # Compute scale factor once per object type
        if scale_factors[label_text] is None:
            scale_factor = real_heights[label_text] / height_px
            scale_factors[label_text] = scale_factor
            # print(f"[INFO] Scale factor for {label_text}: {scale_factor:.5f} ft/pixel")

        scale_factor = scale_factors[label_text]
        width_ft = width_px * scale_factor
        height_ft = height_px * scale_factor
        area_ft = width_ft * height_ft

        object_areas[label_text] += area_ft
        object_counts[label_text] += 1
        total_area_all += area_ft

        # print(f"{label_text.capitalize()} {i}: {width_ft:.2f} ft x {height_ft:.2f} ft = {area_ft:.2f} sq.ft")

        # Draw bounding boxes and labels
        # color = (255, 0, 0) if label_text == "windows" else (0, 0, 255)
        outline_color = "red" if label_text == "windows" else ("blue" if label_text == "doors" else "green")
        draw.rectangle(box, outline = outline_color, width = 3)
        text_position = (x_min, y_min - 25 if y_min > 25 else y_min + 5)
        draw.text(text_position, f"{label_text} {confidence:.2f}",
                  fill="red" if label_text == "windows" else "blue")

    ## --- Print Summary ---
    # print("\n[RESULT] Object-wise total areas (sq.ft):")
    # for obj, area in object_areas.items():
    #     print(f"  {obj}: {area:.2f}")
    # print(f"[RESULT] Combined Total Area: {total_area_all:.2f} sq.ft")

    # # Print Summary
    # print("\n[RESULT] Object-wise total areas (sq.ft) and counts:")
    # for obj in object_areas.keys():
    #     print(f"  {obj}: {object_areas[obj]:.2f} sq.ft ({object_counts[obj]} detected)")
    # print(f"[RESULT] Combined Total Area: {total_area_all:.2f} sq.ft")

    # --- Annotate Summary on Image ---
    image_np = np.array(image)
    y_offset = 40
    for obj, area in object_areas.items():
        cv2.putText(image_np, f"{obj.capitalize()} Area: {area:.2f} sq.ft",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        y_offset += 30

    cv2.putText(image_np, f"Total Area: {total_area_all:.2f} sq.ft",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Save & Display ---
    if output_path:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr) 
        print(f"Image with detections saved to: {output_path}")

    # Convert the Numpy Image 
    image = Image.fromarray(image_np)

    # Showing the Image
    # image.show()

    # Creating the Json for storing the Final result
    result_json =  {
        "windows_area": np.round(object_areas["windows"], 2),
        "doors_area": np.round(object_areas["doors"], 2),
        "pipes_area": np.round(object_areas["pipes"], 2),
        "total_area": np.round(total_area_all, 2),
        "windows_count": object_counts["windows"],
        "doors_count": object_counts["doors"],
        "pipes_count": object_counts["pipes"],
        "total_area": total_area_all
    }

    # Accessing the area of each object
    window_area = result_json['windows_area']
    door_area = result_json['doors_area']
    pipe_area = result_json['pipes_area']
    total_area = np.round(result_json['total_area'],2)
    window_count = result_json['windows_count']
    door_count = result_json['doors_count']
    pipe_count = result_json['pipes_count']

    # Returning the areas
    return window_area, door_area, pipe_area, total_area, window_count, door_count, pipe_count

# # Inference on the Modified Function
# if __name__ == "__main__":
#     image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Paintable_Area_Calculation\Images\OrgImages\PipeTestImage.png"
#     output_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Building_Paintable_Area_Calculation\Images\Modify\PipeTestImage_Objects_Area.jpg"

#     window, door, pipe, total, window_count, door_count, pipe_count = detect_objects(image_path = image_path, real_height_pipes_feet = 20, real_height_window_feet = 3.5, real_height_door_feet = 7.0, threshold = 0.2, output_path = output_path)

#     # print("Window Area:", window)
#     # print("Door Area:", door)
#     # print("Pipe Area:", pipe)
#     # print("Total Area:", total)