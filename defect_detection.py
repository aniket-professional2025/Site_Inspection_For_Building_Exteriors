# Importing Required Packages
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

# Prepare processor and model
model_id = "rziga/mm_grounding_dino_large_all"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Define functions to get the detections
def get_defects(image_path, output_path = None, detection_threshold: float = 0.5):

    print("[DEBUG] Inside the Main Function of Defect Detection")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Image not Found in {image_path}")
    
    # Setting the object name to find
    text_labels = [["crack", "dampness on the wall", "fungal growth", "flaking", "uneven surface", "holes"]] 
    # text_labels = [["crack"]] # With threshold 0.3 
    # text_labels = [["dampness on the wall"]] # With threshold 0.1
    # text_labels = [["fungal growth"]] # With threshold 0.2
    # text_labels = [["flaking"]] # With threshold 0.1
    # text_labels = [["uneven surface"]] # With threshold 0.4
    # text_labels = [["holes"]] # With threshold 0.3

    # Creating the complete Input
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

    # Define the expected object labels
    expected_labels = [lbl.lower() for group in text_labels for lbl in group]

    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        # Convert or clean label
        if isinstance(label, str):
            label_text = label.strip().lower()
        elif isinstance(label, int) and 0 <= label < len(expected_labels):
            label_text = expected_labels[label]
        else:
            label_text = ""

        # Skip if label not valid
        if not label_text or label_text not in expected_labels:
            print(f"[WARNING] Skipping invalid label: {label}")
            continue

        box = [round(x, 2) for x in box.tolist()]
        score_val = round(score.item(), 3)
        label_text = f"{label} ({score_val})"

        print(f"Detected {label} with confidence {score_val} at location {box}")

        # Draw rectangle and label
        draw.rectangle(box, outline = "red", width = 3)
        draw.text((box[0], box[1] - 10), label_text, fill = "red")

    # Display result inline
    image.show()

    # Save the Image in Mentioned output path
    if output_path is None:
        pass
    else:
        image.save(output_path)

    # Confirmation of successful run
    print(f"The detected Image is Saved at {output_path}")

    # Return the labels
    final_labels = [lbl for lbl in result["labels"] if isinstance(lbl, str) and lbl.strip() and lbl.strip().lower() != "wall"]

    final_labels =  list(set(final_labels))

    return final_labels

# Inference on the Function
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\HoleImage.jpg"
    labels = get_defects(image_path = image_path, detection_threshold = 0.3)
    print(labels)