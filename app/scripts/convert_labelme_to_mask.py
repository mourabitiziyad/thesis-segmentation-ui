import json
import numpy as np
import cv2
from PIL import Image
import os

# Path to your LabelMe JSON file
json_path = "/Users/I752629/Desktop/Reference Thesis Images/Final Set/Set 2/S2L2A_T33UVU-20250404-u04c5816_TCI_out_label.json"

# Load LabelMe JSON
with open(json_path, "r") as f:
    data = json.load(f)

image_shape = (data["imageHeight"], data["imageWidth"])
mask = np.zeros(image_shape, dtype=np.uint8)

for shape in data["shapes"]:
    if shape["shape_type"] != "polygon":
        continue
    points = np.array(shape["points"], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)  # Fill PV area with 1

# Save the binary mask
mask_img = Image.fromarray(mask * 255)  # scale to 0-255 for visibility
mask_img.save("pv_mask.png")
print("Saved binary mask to pv_mask.png")
