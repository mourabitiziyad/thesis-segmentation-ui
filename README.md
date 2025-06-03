# Deep Learning for Solar Panel Recognition

This repository contains a deep learning solution for solar panel segmentation and recognition using computer vision techniques.

## Features

- Segmentation of solar panels from satellite or aerial imagery
- Interactive demo app using Streamlit
- Support for high-resolution images with tiled processing
- Multiple visualization options: binary mask, overlay, and bounding boxes
- Area calculations and panel coordinates
- Multi-resolution comparison tool

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Streamlit
- Albumentations
- Segmentation Models PyTorch (SMP)

## Setup

1. **Activate the virtual environment:**
   ```bash
   source new_venv/bin/activate
   ```

2. **Navigate to the app directory:**
   ```bash
   cd app
   ```

## Adding Images for Processing

### Image Directory Structure

Place your images in the `app/data/` directory. The app will automatically detect and display available images in the dropdown menu.

```
app/data/
├── your_image1.jpg
├── your_image2.png
├── satellite_image.tif
└── aerial_photo.jpeg
```

### Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **TIFF** (.tif, .tiff)

### Image Requirements

- **Resolution:** Any resolution supported (optimized for high-res images)
- **Color space:** RGB color images
- **File size:** No strict limit, but larger files will take longer to process

### Ground Truth Labels (Optional)

If you have ground truth segmentation masks, place them in the same directory with the suffix `_label`:

```
app/data/
├── image1.jpg           # Original image
├── image1_label.png     # Ground truth mask (binary: 0=background, 255=solar panel)
├── image2.jpg
└── image2_label.png
```

**Label format requirements:**
- Binary images: 0 (black) for background, 255 (white) for solar panels
- Same filename as original image + `_label` suffix
- Preferably PNG format for lossless compression

### High-Resolution Images

For very large images (>2000px), the app provides tiled processing options:
- **Automatic detection** of high-resolution images
- **Configurable tile size** (default: 512px)
- **Overlap settings** to ensure seamless processing
- **Scale factor adjustment** for memory optimization

## Usage

### Demo Application

Run the main demo application:

```bash
streamlit run demo.py
```

Features:
- Upload or select images from the data folder
- Choose from available pre-trained models
- Adjust processing parameters for high-resolution images
- View segmentation results with various visualizations
- Calculate solar panel areas and export coordinates

### Multi-Resolution Comparison

Run the comparison tool:

```bash
streamlit run compare_resolutions.py
```

Features:
- Compare multiple model predictions
- Side-by-side resolution analysis
- Performance metrics when ground truth is available

## Repository Structure

- `app/`: Contains the main application code
  - `demo.py`: Main Streamlit demo application
  - `compare_resolutions.py`: Multi-resolution comparison tool
  - `utils.py`: Utility functions for image processing
  - `data/`: **Place your images here**
    - Sample images and labels
    - User-uploaded images
- `models/`: Pre-trained deep learning models
- `new_venv/`: Virtual environment with all dependencies

## Tips for Best Results

1. **Lighting:** Avoid heavily shadowed or overexposed images
2. **File Naming:** Use descriptive names to easily identify images in the dropdown
3. **Ground Truth:** Provide labels when available for accuracy assessment
4. **Memory:** For very large images, adjust the scale factor in tiled processing mode
