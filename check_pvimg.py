import cv2
import os

# Check dimensions of specific images
image_paths = [
    'app/old_data/PV01_324958_1203803.png',  # The specific image mentioned
    'app/old_data/PV08_350325_1189776.png',  # A PV08 image for comparison
    'app/data/S2L2A_T32UQD-20240721-u3af099a_TCI.png',         # Regular image
    'app/data/S2L2A_4x_T32UQD-20240721-u3af099a_TCI_stitched.png',  # 4x image
    'app/data/S2L2Ax10_T32UQD-20240721-u3af099a_TCI.png'       # 10x image
]

print("Image dimensions:")
for path in image_paths:
    try:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f'{os.path.basename(path)}: {w}x{h} pixels, {size_mb:.2f} MB')
            else:
                print(f'{os.path.basename(path)}: Failed to load image')
        else:
            print(f'{os.path.basename(path)}: File not found')
    except Exception as e:
        print(f'Error with {os.path.basename(path)}: {e}') 