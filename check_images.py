import cv2
import os

# Check dimensions of images in app/data
print("Images in app/data:")
for img in os.listdir('app/data'):
    if img.endswith('.png'):
        try:
            path = os.path.join('app/data', img)
            img_data = cv2.imread(path)
            if img_data is not None:
                h, w = img_data.shape[:2]
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f'{img}: {w}x{h} pixels, {size_mb:.2f} MB')
        except Exception as e:
            print(f'Error with {img}: {e}')

# Check for other possible data directories
data_dirs = ['data', 'raw_data', 'old_data', 'src/data', 'notebooks/data']
for data_dir in data_dirs:
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        print(f"\nFound directory: {data_dir}")
        for img in os.listdir(data_dir):
            if img.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                try:
                    path = os.path.join(data_dir, img)
                    img_data = cv2.imread(path)
                    if img_data is not None:
                        h, w = img_data.shape[:2]
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        print(f'{img}: {w}x{h} pixels, {size_mb:.2f} MB')
                except Exception as e:
                    print(f'Error with {img}: {e}') 