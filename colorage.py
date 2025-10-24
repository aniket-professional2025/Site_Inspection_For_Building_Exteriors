# Importing Required Packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# The analyze_paint_fading function
def analyze_paint_fading(image_path):

    # Step 1: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Step 2: Convert to HSV for Saturation analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract the Saturation channel (S-channel)
    saturation_channel = hsv[:, :, 1]
    value_channel = hsv[:, :, 2]

    # --- Step 3: Identify a Representative Area of the Paint (Masking) ---
    # Create a mask to isolate the wall surface (V between 50 and 230)
    lower_V = 50
    upper_V = 230
    wall_mask = cv2.inRange(value_channel, lower_V, upper_V)

    # Remove areas of complex texture (edges/cracks)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Final robust mask for the smooth, central wall area:
    texture_mask = cv2.bitwise_and(wall_mask, wall_mask, mask = ~edges_dilated)

    # --- Step 4: Calculate Mean Saturation (The Fading Metric) ---
    # Use the robust mask to calculate the mean saturation only on the clean paint area.
    valid_pixels = saturation_channel[texture_mask > 0]

    if valid_pixels.size == 0:
        print("Warning: Could not isolate a clean paint area for analysis. Returning 0 saturation and 100 fading score.")
        mean_saturation = 0
    else:
        # Saturation values range from 0 (pure gray/white) to 255 (pure color)
        mean_saturation = np.mean(valid_pixels)

    # --- Step 4.5: Calculate Analyzed Area in Pixels ---
    # Count the number of white pixels in the texture_mask
    analyzed_area_pixels = np.sum(texture_mask > 0)

    # --- Step 5: Calculate Fading Score ---
    max_saturation = 200 # Assumed Max Saturation (as per your code)
    if max_saturation <= 0:
        fading_score = 100.0
    else:
        fading_score = 100.0 * (1 - (mean_saturation / max_saturation))

    # Clamp score between 0 and 100
    fading_score = max(0, min(100, fading_score))
    
    # --- Step 6: Display Results ---

    # Create an overlay for the analyzed area
    overlay = img.copy()
    overlay[texture_mask > 0] = [255, 0, 0] # Highlight analyzed area in blue

    images = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Original Image")
    ]

    plt.figure(figsize=(15, 5))
    for i, (im, title) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.title(title, fontsize=10)

        if len(im.shape) == 2:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(im)

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Mean Paint Saturation (S-Channel): {mean_saturation:.2f} (Max 255)")
    print(f"Assumed Max Saturation for Score: {max_saturation}")
    print(f"Estimated Fading Score (0=New, 100=Max Fade): {fading_score:.2f}")
    print(f"Analyzed Area (in pixels): {analyzed_area_pixels}")

    # Determining the Paint Age based on the fading score
    if fading_score <= 0.25:
        print(f"The paint with fading score {fading_score} is New")
    elif fading_score > 0.25 and fading_score <= 0.55:
        print(f"The paint with fading score {fading_score} is Old")
    else:
        print(f"The paint with fading score {fading_score} is Very Old")

    return mean_saturation, fading_score

# Execute the analysis
if __name__ == "__main__":
    image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\GreenWallImage.png"
    mean_sat, fade_score = analyze_paint_fading(image_path)