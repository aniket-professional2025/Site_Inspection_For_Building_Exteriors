# Importing Required Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define Function to get Hole and its area
def get_hole_segmentation(image_path):

    # Step 1: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply bilateral filter (preserve edges while smoothing)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 4: Edge detection for cracks
    edges = cv2.Canny(blur, 50, 150)
    crack_ratio = np.sum(edges > 0) / edges.size

    # Step 5: HSV conversion for damp region detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask 1: Yellow/Green Stains (Fungal/Mold Growth) # H: 15-40 for Yellow/Green # S: min 40 # V: min 40
    lower_stain = np.array([15, 80, 80]) # [15, 80, 80]
    upper_stain = np.array([60, 255, 255]) # [60, 255, 255]
    mask_stain = cv2.inRange(hsv, lower_stain, upper_stain)

    # Mask 2: Dark/Gray Damp Areas (Low Saturation/Value) # H: Any hue (0-180) # S: Low Saturation (0-80) to catch gray # V: Low Value (0-150) to catch dark areas
    lower_dark_hole = np.array([0, 0, 0]) # [0,0,0]
    upper_dark_hole = np.array([180, 100, 100]) # [180, 100, 100] # best: [180, 100, 100]
    mask_dark = cv2.inRange(hsv, lower_dark_hole, upper_dark_hole)

    # mask_damp = cv2.inRange(hsv, lower_damp, upper_damp)
    mask_hole = cv2.bitwise_or(mask_stain, mask_dark)

    # Step 6: Morphological cleanup of damp mask
    kernel = np.ones((5,5), np.uint8)
    mask_hole = cv2.morphologyEx(mask_hole, cv2.MORPH_CLOSE, kernel)
    mask_hole = cv2.morphologyEx(mask_hole, cv2.MORPH_OPEN, kernel)

    hole_area_px = np.sum(mask_hole > 0)
    hole_ratio = np.sum(mask_hole > 0) / mask_hole.size

    # Step 7: Overlay damp mask in red
    overlay = img.copy()
    overlay[mask_hole > 0] = [0, 0, 255]  # damp in red

    # Step 8: Display results in one row
    images = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Original"),
        (gray, "Grayscale"),
        (edges, "Canny Edges"),
        (mask_hole, "Holes/Uneven Surface Mask"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "Damp Overlay")
    ]

    plt.figure(figsize=(18,4))
    for i, (im, title) in enumerate(images):
        plt.subplot(1, len(images), i+1)
        if len(im.shape) == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(im)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("Crack Ratio:", crack_ratio)
    print("Damp Ratio:", hole_ratio)
    print("Damp Area (pixels):", hole_area_px)

    return hole_ratio, hole_area_px

# Example Usage
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\HoleImage.jpg"
    ratio, area = get_hole_segmentation(image_path)