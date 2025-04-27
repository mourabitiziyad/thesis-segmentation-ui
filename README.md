# Deep Learning for Solar Panel Recognition

This repository contains a deep learning solution for solar panel segmentation and recognition using computer vision techniques.

## Features

- Segmentation of solar panels from satellite or aerial imagery
- Interactive demo app using Streamlit
- Support for high-resolution images with tiled processing
- Multiple visualization options: binary mask, overlay, and bounding boxes
- Area calculations and panel coordinates

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Streamlit
- Albumentations
- Segmentation Models PyTorch (SMP)

## Usage

Run the demo application:

```bash
cd app
streamlit run demo.py
```

## Repository Structure

- `app/`: Contains the main application code
  - `demo.py`: Streamlit demo application
  - `utils.py`: Utility functions for image processing
- `data/`: Example images
- `models/`: Trained deep learning models
