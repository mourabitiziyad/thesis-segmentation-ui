import json
import numpy as np
import cv2
from PIL import Image
import os

# Path to your LabelMe JSON file
json_path = "/Users/I752629/Desktop/Reference Thesis Images/Final Set/set 3/satlas_label.json"

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

# Save the binary mask in the same directory as the input JSON file
output_dir = os.path.dirname(json_path)
output_filename = os.path.splitext(os.path.basename(json_path))[0] + ".png"
output_path = os.path.join(output_dir, output_filename)

mask_img = Image.fromarray(mask * 255)  # scale to 0-255 for visibility
mask_img.save(output_path)
print(f"Saved binary mask to {output_path}")
