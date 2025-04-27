# Data Directory

Place your aerial or satellite images for solar panel detection in this directory.

## Supported Image Formats

The application supports the following image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tif, .tiff)

## Naming Convention for Ground Truth Labels

If you have ground truth segmentation masks, name them with the same base filename
plus the `_label` suffix. For example:

- Image: `example.png`
- Mask: `example_label.png`

Ground truth masks should be binary images where white (255) represents solar panels
and black (0) represents background.

## Sample Data

To get started, you can use publicly available datasets like:
- AIRS (Aerial Imagery for Roof Segmentation)
- SolarPanel segmentation dataset from UC Merced
- Open Cities AI Challenge dataset 