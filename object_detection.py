# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"
print(f"========== The model: {model_id} is Set Successfully =============")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=========== The {device} is The Suitable Device ==================")

processor = AutoProcessor.from_pretrained(model_id)
print("[DEBUG] The processor is loaded Successfully")

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("[DEBUG] The Model is Loaded Successfully")

# Define functions to get the detections
def get_detections(image_path, detection_threshold: float = 0.5):

    print("[DEBUG] Inside the Main Function")

    image = Image.open(image_path).convert("RGB")
    text_labels = [["damp"]]
    inputs = processor(images = image, text = text_labels, return_tensors = "pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    print("[DEBUG] Generating the Results")

    results = processor.post_process_grounded_object_detection(outputs, threshold = detection_threshold, target_sizes = [(image.height, image.width)])
    result = results[0]

    # Fallback if nothing is detected
    detected_count = len(result["boxes"])
    if detected_count == 0:
        print("No objects detected. Exiting.")
        return
    
    print(f"Total {detected_count} objects detected at threshold {detection_threshold}")

    print("[DEBUG] Getting the Detections")

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        score_val = round(score.item(), 3)
        label_text = f"{label} ({score_val})"

        print(f"Detected {label} with confidence {score_val} at location {box}")

        # Draw rectangle and label
        draw.rectangle(box, outline = "red", width = 3)
        draw.text((box[0], box[1] - 10), label_text, fill = "red")

    # Display result inline
    image.show()

# Inference on the Function
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\DampImage.png"
    get_detections(image_path = image_path, detection_threshold = 0.1)