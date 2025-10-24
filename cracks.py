# Importing Required Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Creating a function to get the crack segmentation
def get_crack_segmentation(image_path):

  # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply smoothing (bilateral filter preserves edges)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 4: Detect edges (potential cracks)
    edges = cv2.Canny(blur, 50, 150)

    # Step 5: Morphological operations to close gaps in cracks
    kernel = np.ones((3,3), np.uint8)
    crack_mask = cv2.dilate(edges, kernel, iterations=1)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)

    # Step 6: Pixel-wise crack area (number of white pixels in mask)
    crack_area_px = np.sum(crack_mask > 0)

    # Step 7: Create overlay for visualization
    overlay = img.copy()
    overlay[crack_mask > 0] = [0, 0, 255]  # Highlight cracks in red

    # Step 8: Show results
    plt.figure(figsize=(16,8))

    plt.subplot(2,3,1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.title("After Morphology (Mask)")
    plt.imshow(crack_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.title("Overlay (Crack Highlighted)")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Crack area (in pixels):", crack_area_px)

    # Applying a Threshold value to Measure it a Crack is High or Low Based on Measurement
    if crack_area_px >= 40000:
      print("The Crack Severity is High")
    elif crack_area_px >= 20000 and crack_area_px < 40000:
      print("The Crack Severity is Medium")
    else:
      print("The Crack Severity is Low")
      
    return crack_area_px

# Example usage
if __name__ == "__main__":
  image_path = r"C:\Users\Webbies\Jupyter_Notebooks\Berger_Site_Inspection_Exterior\InputImages\CrackImage.png"
  area = get_crack_segmentation(image_path)