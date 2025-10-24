# Importing Required Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define Function to Detect the Paint Peeling
def get_peeling_segmentation(image_path):

    # Step 1: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Step 2: Convert to HSV for robust color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # NOTE: To display the HSV image visually, we usually convert it back to BGR
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Step 3: Define HSV Masks for Exposed Layers ---

    # Mask 1: Beige/Tan Exposed Plaster
    lower_beige = np.array([10, 20, 100])
    upper_beige = np.array([35, 120, 255])
    mask_beige = cv2.inRange(hsv, lower_beige, upper_beige)

    # Mask 2: Inner Dark Green/Grayish Layer
    lower_green = np.array([45, 60, 60]) # [45, 50, 50]
    upper_green = np.array([80, 200, 150]) # [80, 200, 200]
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine the two masks using bitwise OR
    combined_mask = cv2.bitwise_or(mask_beige, mask_green)

    # --- Step 4: Morphological Cleanup ---
    kernel = np.ones((7, 7), np.uint8) # Closing kernel
    morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    kernel_small = np.ones((3, 3), np.uint8) # Opening kernel
    morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, kernel_small)

    # Step 5: Final Segmentation Mask
    segmentation_mask = morphed_mask.copy()

    # Step 6: Calculate Area and Create Overlay
    peel_area_px = np.sum(segmentation_mask > 0)

    # Create the overlay
    overlay = img.copy()
    red_color_bgr = [0, 0, 255] # BGR for RED
    overlay[segmentation_mask > 0] = red_color_bgr

    # Step 7: Display results in a single row

    images = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original Image"),
        (cv2.cvtColor(hsv_display, cv2.COLOR_BGR2RGB), "HSV Conversion"),
        (morphed_mask, "Morphed Image"), # Result after morphology
        (segmentation_mask, "Segmentation Mask"), # Binary mask
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "Peeling Overlay")
    ]

    plt.figure(figsize=(20, 4)) # Adjusted figsize for 5 images
    num_images = len(images)

    for i, (im, title) in enumerate(images):
        plt.subplot(1, num_images, i + 1)
        plt.title(title, fontsize=10)

        if len(im.shape) == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(im)

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Total Peeling Area (in pixels): {peel_area_px}")

    return peel_area_px

# Example Usage
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\PaintPeelingImage.png"
    area = get_peeling_segmentation(image_path)