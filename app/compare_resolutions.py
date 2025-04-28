import os
import torch
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import albumentations as A
from pathlib import Path
import segmentation_models_pytorch as smp
from utils import *
import glob
import re
from PIL import Image
from scipy import ndimage

# Set page configuration
st.set_page_config(
    page_title='Multi-Resolution Comparison',
    page_icon="https://api.iconify.design/openmoji/solar-energy.svg?width=500",
    layout='wide'
)

# Cache helper for image processing
def transpose_and_convert_to_float32(img, **kwargs):
    """Helper function to transpose image and convert to float32 (avoids lambda warning)"""
    return img.transpose(2, 0, 1).astype('float32')

@st.cache_data
def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess an image"""
    try:
        # Try loading with OpenCV first
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL
            image = np.array(Image.open(image_path))
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    original_dimensions = image.shape[:2]
    
    # Resize for display
    display_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return {
        "original": image,
        "display": display_image,
        "dimensions": original_dimensions,
        "path": image_path,
        "filename": os.path.basename(image_path)
    }

@st.cache_data
def segment_image(_model, image, backbone='se_resnext101_32x4d', input_size=(256, 256)):
    """Run segmentation on an image"""
    # Resize for model input
    resized_image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    
    # Get preprocessing function
    preprocess_input = smp.encoders.get_preprocessing_fn(backbone)
    
    # Apply preprocessing
    _transform = [
        A.Lambda(image=preprocess_input),
        A.Lambda(image=transpose_and_convert_to_float32),
    ]
    preprocessing = A.Compose(_transform)
    sample = preprocessing(image=resized_image)
    proc_image = sample['image']
    
    # Convert to tensor and predict
    image_torch = torch.from_numpy(proc_image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        prediction = _model.predict(image_torch)
    
    # Convert prediction to numpy
    mask = prediction.squeeze().cpu().numpy()
    
    # Resize mask back to original image size
    mask = (mask > 0.5).astype(np.float32)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask_resized

@st.cache_data(show_spinner=False)
def find_image_groups(directory, extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
    """Find groups of images that might be of the same location but different resolutions"""
    # Get all image files
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    
    # If no files found, try case-insensitive search
    if not all_files:
        for ext in extensions:
            # Try uppercase extension
            all_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
            # Try mixed case extensions like .Tif, .TIF, etc.
            mixed_ext = ext.replace('.', '.').upper()
            all_files.extend(glob.glob(os.path.join(directory, f"*{mixed_ext}")))
    
    # Filter out label files
    image_files = [f for f in all_files if 'label' not in f.lower()]
    
    # Debug output
    st.sidebar.text(f"Found {len(image_files)} images in {os.path.basename(directory)}")
    
    # If still no files, check if there are any files in the directory
    if not image_files:
        all_dir_files = os.listdir(directory)
        st.warning(f"Found {len(all_dir_files)} files in directory, but none matched image patterns. Files: {', '.join(all_dir_files[:10])}{'...' if len(all_dir_files) > 10 else ''}")
        return {}, []
    
    # Debug: Print all filenames found
    filenames = [os.path.basename(f) for f in image_files]
    st.sidebar.text(f"Files found: {', '.join(filenames)}")
    
    # Special case for data set 4 folder - we know these are all related
    if "data set 4" in directory:
        return {"T32UQD Location Images": image_files}, image_files
    
    # Try to group by location
    location_groups = {}
    
    # First try: Group by geographic identifiers in filename
    # This assumes filenames might contain location codes, coordinates, or other identifiers
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Extract potential location identifiers (customize for your naming convention)
        # Look for patterns like:
        # 1. Location codes: T33UUS, T33UVU, etc. 
        # 2. Remove resolution indicators like 4x, 10x, x4, x10
        # 3. Extract date if present (YYYYMMDD format)
        
        # Try to match location code like T33UUS or T32UQD
        loc_match = re.search(r'T\d+[A-Z]+', filename)
        location_code = loc_match.group(0) if loc_match else ""
        
        # Try to match date like 20250404
        date_match = re.search(r'20\d{6}', filename)
        date_code = date_match.group(0) if date_match else ""
        
        # Try to extract coordinates if present (like numbers in PV files)
        coord_match = re.search(r'_(\d{6})_(\d{6})', filename)
        coord_code = f"{coord_match.group(1)}_{coord_match.group(2)}" if coord_match else ""
        
        # Try to extract unique IDs in S2L2A files (like uf5abf05)
        id_match = re.search(r'-([a-z0-9]{7})', filename)
        unique_id = id_match.group(1) if id_match else ""
        
        # Create a composite key with available identifiers
        composite_key = f"{location_code}_{date_code}_{coord_code}_{unique_id}".strip('_')
        
        # If we couldn't find any identifier, use the base filename without extension and resolution indicators
        if not composite_key:
            # Remove resolution indicators and extension
            base_name = re.sub(r'_\d+x|_x\d+', '', os.path.splitext(filename)[0])
            # Remove file extension numbers/version indicators
            base_name = re.sub(r'_v\d+$|_\d+$', '', base_name)
            # Remove model names (like esrgan, satlas)
            base_name = re.sub(r'^[a-zA-Z]+_', '', base_name)
            composite_key = base_name
        
        # Add to appropriate group
        if composite_key not in location_groups:
            location_groups[composite_key] = []
        location_groups[composite_key].append(img_path)
    
    # Debug: Show all grouping keys found
    st.sidebar.text(f"Grouping keys: {', '.join(location_groups.keys())}")
    
    # Add a fallback for files that might start with the same prefix
    if not any(len(v) > 1 for v in location_groups.values()):
        prefix_groups = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Get first part of filename (before first underscore or first 5 chars)
            if '_' in filename:
                prefix = filename.split('_')[0]
            else:
                prefix = filename[:min(5, len(filename))]
            
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(img_path)
        
        # If we found any groups with multiple images, use those
        prefix_multi_groups = {k: v for k, v in prefix_groups.items() if len(v) > 1}
        if prefix_multi_groups:
            location_groups = prefix_multi_groups
    
    # Filter groups to only include those with multiple images
    multi_image_groups = {k: v for k, v in location_groups.items() if len(v) > 1}
    
    # If no multi-image groups found but we're in data set 4, use all images
    if not multi_image_groups and "data set 4" in directory and image_files:
        multi_image_groups = {"All Images": image_files}
    
    # If no multi-image groups found, create a default group with all images
    if not multi_image_groups and image_files:
        multi_image_groups = {"All Images": image_files}
    
    # Debug: Show final group counts
    for group_name, group_files in multi_image_groups.items():
        st.sidebar.text(f"Group '{group_name}': {len(group_files)} images")
    
    return multi_image_groups, image_files

def extract_resolution_from_filename(filename, full_path=None):
    """Extract resolution information from filename"""
    # Special case for data set 4 files which have different naming patterns
    
    # For files with explicit model prefix like esrgan_4x or satlas_4x
    model_res_match = re.search(r'^([a-zA-Z]+)_(\d+)x', filename)
    if model_res_match:
        model_name = model_res_match.group(1)
        resolution = model_res_match.group(2)
        return f"{model_name}_{resolution}x"
    
    # Look for explicit resolution patterns like 10x, 4x, x10, x4
    match = re.search(r'(\d+)x|x(\d+)', filename)
    if match:
        # Return the first non-None group (either group 1 or 2)
        res = match.group(1) or match.group(2)
        # For S2L2Ax10 type patterns
        if filename.startswith("S2L2Ax"):
            return f"S2L2A_{res}x"
        return res
    
    # For base S2L2A files (without resolution indicator)
    if filename.startswith("S2L2A_") and not any(x in filename for x in ["4x", "10x", "x4", "x10"]):
        return "S2L2A_1x"  # Base resolution
    
    # Look for dimensions in the filename - larger numbers might indicate higher resolution
    # Example: if filename contains dimensions like 1024x1024 or 512x512
    dim_match = re.search(r'(\d{3,4})x(\d{3,4})', filename)
    if dim_match:
        # Use the first dimension as an indicator
        return dim_match.group(1)
    
    # If filename contains "high" or "low" resolution indicators
    if "highres" in filename.lower() or "hi-res" in filename.lower() or "hi_res" in filename.lower():
        return "High"
    if "lowres" in filename.lower() or "lo-res" in filename.lower() or "lo_res" in filename.lower():
        return "Low"
    
    # If filename contains words like "original", "full", etc.
    if "original" in filename.lower() or "full" in filename.lower():
        return "Original"
    
    # If nothing matches, check the file size - this requires the full path
    if full_path:
        try:
            file_size = os.path.getsize(full_path)
            # Very large files might be higher resolution
            if file_size > 5 * 1024 * 1024:  # > 5MB
                return "Large"
            elif file_size > 1 * 1024 * 1024:  # > 1MB
                return "Medium"
            else:
                return "Small"
        except (OSError, IOError):
            pass
    
    return "Unknown"

def create_difference_mask(mask1, mask2):
    """Create a visualization of differences between two masks"""
    # Ensure masks are the same size
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create RGB difference visualization
    diff = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    
    # Areas in both masks (green)
    diff[np.logical_and(mask1 > 0.5, mask2 > 0.5)] = [0, 255, 0]
    
    # Areas only in mask1 (red)
    diff[np.logical_and(mask1 > 0.5, mask2 <= 0.5)] = [255, 0, 0]
    
    # Areas only in mask2 (blue)
    diff[np.logical_and(mask1 <= 0.5, mask2 > 0.5)] = [0, 0, 255]
    
    return diff

def compute_metrics_table(images_data):
    """Compute metrics between all pairs of images"""
    num_images = len(images_data)
    metrics_data = []
    
    for i in range(num_images):
        for j in range(i+1, num_images):
            mask1 = images_data[i]["mask"]
            mask2 = images_data[j]["mask"]
            
            # Resize mask2 to match mask1 dimensions for comparison
            if mask1.shape != mask2.shape:
                mask2_resized = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask2_resized = mask2
            
            # Calculate metrics
            intersection = np.logical_and(mask1 > 0.5, mask2_resized > 0.5).sum()
            union = np.logical_or(mask1 > 0.5, mask2_resized > 0.5).sum()
            
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / ((mask1 > 0.5).sum() + (mask2_resized > 0.5).sum() + 1e-8)
            
            # Calculate area difference
            area1 = (mask1 > 0.5).sum()
            area2 = (mask2_resized > 0.5).sum()
            area_diff = abs(area1 - area2) / max(area1, area2) * 100 if max(area1, area2) > 0 else 0
            
            # Add processing method information
            img1_method = images_data[i].get("processing_method", "standard")
            img2_method = images_data[j].get("processing_method", "standard")
            
            metrics_data.append({
                "Image 1": f"{images_data[i]['filename']} ({img1_method})",
                "Image 2": f"{images_data[j]['filename']} ({img2_method})",
                "IoU": f"{iou:.4f}",
                "Dice": f"{dice:.4f}",
                "Area Diff (%)": f"{area_diff:.2f}%"
            })
    
    return pd.DataFrame(metrics_data)

def create_boundary_visualization(mask, kernel_size=3):
    """Create a visualization of the mask boundary"""
    # Ensure binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Dilate and find boundary via XOR
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    boundary = cv2.bitwise_xor(dilated, binary_mask)
    
    # Create RGB visualization
    viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    viz[boundary > 0] = [255, 255, 0]  # Yellow boundary
    
    return viz

def create_heat_map_visualization(image_1, image_2, mask_1, mask_2):
    """Create a heatmap visualization showing segmentation confidence from both resolutions"""
    # Resize everything to match dimensions
    h, w = image_1.shape[:2]
    image_2_resized = cv2.resize(image_2, (w, h), interpolation=cv2.INTER_AREA)
    mask_2_resized = cv2.resize(mask_2, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create a combined heatmap
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Areas present in both masks (green)
    both = np.logical_and(mask_1 > 0.5, mask_2_resized > 0.5)
    heatmap[both] = [0, 255, 0]
    
    # Areas only in mask_1 (red with 50% opacity)
    only_mask_1 = np.logical_and(mask_1 > 0.5, ~both)
    
    # Areas only in mask_2 (blue with 50% opacity)
    only_mask_2 = np.logical_and(mask_2_resized > 0.5, ~both)
    
    # Blend with the original image
    result = image_1.copy()
    # Add colors for areas only in one mask (with transparency)
    result[only_mask_1] = result[only_mask_1] * 0.5 + np.array([255, 0, 0]) * 0.5
    result[only_mask_2] = result[only_mask_2] * 0.5 + np.array([0, 0, 255]) * 0.5
    # Add colors for areas in both masks (with transparency)
    result[both] = result[both] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    return result.astype(np.uint8)

def create_slider_comparison(image1, image2, mask1, mask2):
    """Create an interactive slider comparison visualization"""
    # Ensure images are the same size
    h, w = image1.shape[:2]
    image2_resized = cv2.resize(image2, (w, h), interpolation=cv2.INTER_AREA)
    mask2_resized = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create mask overlays
    overlay1 = image1.copy()
    overlay1[mask1 > 0.5] = overlay1[mask1 > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3
    
    overlay2 = image2_resized.copy()
    overlay2[mask2_resized > 0.5] = overlay2[mask2_resized > 0.5] * 0.7 + np.array([0, 0, 255]) * 0.3
    
    return overlay1, overlay2

def create_precision_recall_heatmap(mask1, mask2):
    """Create a precision-recall heatmap visualization between two masks"""
    # Ensure masks are binary
    mask1_bin = mask1 > 0.5
    mask2_bin = mask2 > 0.5
    
    # Create a colored image to represent precision-recall relationships
    heatmap = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    
    # True Positives (Green): Areas present in both masks
    heatmap[np.logical_and(mask1_bin, mask2_bin)] = [0, 255, 0]
    
    # False Positives (Red): Areas only in mask1
    heatmap[np.logical_and(mask1_bin, ~mask2_bin)] = [255, 0, 0]
    
    # False Negatives (Blue): Areas only in mask2
    heatmap[np.logical_and(~mask1_bin, mask2_bin)] = [0, 0, 255]
    
    # Calculate metrics
    tp = np.sum(np.logical_and(mask1_bin, mask2_bin))
    fp = np.sum(np.logical_and(mask1_bin, ~mask2_bin))
    fn = np.sum(np.logical_and(~mask1_bin, mask2_bin))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return heatmap, precision, recall, f1

def process_image_in_tiles(model, image, tile_size=512, overlap=128):
    """Process a large image by breaking it into tiles, processing each, and stitching results.
    
    Args:
        model: The segmentation model
        image: Original large image (RGB)
        tile_size: Size of tiles to process
        overlap: Overlap between tiles to avoid boundary artifacts
        
    Returns:
        Full-size mask with segmentation results
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Initialize the output mask
    full_mask = np.zeros((h, w), dtype=np.float32)
    
    # Initialize a weight map for blending
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Compute preprocessing function
    backbone = 'se_resnext101_32x4d'  # Default backbone
    preprocess_input = smp.encoders.get_preprocessing_fn(backbone)
    
    # Define preprocessing
    _transform = [
        A.Lambda(image=preprocess_input),
        A.Lambda(image=transpose_and_convert_to_float32),
    ]
    preprocessing = A.Compose(_transform)
    
    # Calculate steps with overlap
    x_steps = range(0, w - tile_size + 1, tile_size - overlap)
    if len(x_steps) == 0 or x_steps[-1] + tile_size < w:
        x_steps = list(x_steps) + [max(0, w - tile_size)]
    
    y_steps = range(0, h - tile_size + 1, tile_size - overlap)
    if len(y_steps) == 0 or y_steps[-1] + tile_size < h:
        y_steps = list(y_steps) + [max(0, h - tile_size)]
    
    # Process tiles with a progress bar
    total_tiles = len(x_steps) * len(y_steps)
    progress_bar = st.progress(0)
    tile_counter = 0
    
    # Add status message
    status_text = st.empty()
    
    # Adaptive tile processing - for very large tiles, we might need to resize for inference
    model_input_size = 512  # Maximum input size for model to avoid memory issues
    resize_for_model = tile_size > model_input_size
    
    # Create a smoother weight function - Gaussian weights work better than linear distance
    def create_weight_map(size):
        y_grid, x_grid = np.mgrid[0:size, 0:size]
        center_y, center_x = size // 2, size // 2
        # Use Gaussian weighting - smoother transitions
        sigma = size / 6  # Controls the spread of the Gaussian
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        weight = np.exp(-(dist_from_center**2) / (2 * sigma**2))
        # Normalize weights to [0,1]
        weight = (weight - weight.min()) / (weight.max() - weight.min())
        return weight

    status_text.info(f"Processing {total_tiles} tiles (size: {tile_size}px, overlap: {overlap}px)...")
    
    # Show a mini-grid of tile processing status
    mini_grid_cols = min(10, len(x_steps))  # Cap at 10 columns
    mini_grid_size = (len(y_steps), mini_grid_cols)
    mini_grid = np.zeros(mini_grid_size, dtype=np.uint8)
    
    # Create placeholder for mini-grid visualization
    mini_grid_display = st.empty()
    
    for i, y in enumerate(y_steps):
        for j, x in enumerate(x_steps):
            # Update status
            status_text.info(f"Processing tile {tile_counter+1}/{total_tiles} at position ({x}, {y})...")
            
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            
            # Handle tiles smaller than tile_size at edges
            actual_h, actual_w = tile.shape[:2]
            if actual_h != tile_size or actual_w != tile_size:
                # Pad the tile to reach tile_size
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded_tile[:actual_h, :actual_w] = tile
                tile = padded_tile
            
            # Resize for model if tile is large
            if resize_for_model:
                model_tile = cv2.resize(tile, (model_input_size, model_input_size))
            else:
                model_tile = tile
            
            # Preprocess tile
            sample = preprocessing(image=model_tile)
            proc_tile = sample['image']
            
            # Convert to tensor and predict
            tile_tensor = torch.from_numpy(proc_tile).unsqueeze(0).to('cpu')
            with torch.no_grad():
                prediction = model.predict(tile_tensor)
            
            # Convert prediction to numpy
            tile_mask = prediction.squeeze().cpu().numpy()
            
            # Resize prediction back to original tile size if needed
            if resize_for_model:
                tile_mask = cv2.resize(tile_mask, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            
            # Create weight map for this tile - Gaussian weights work better
            weight = create_weight_map(tile_size)
            
            # Handle edge tiles (which may be smaller than tile_size)
            if actual_h != tile_size or actual_w != tile_size:
                # Only use the valid portion of the prediction and weight map
                tile_mask = tile_mask[:actual_h, :actual_w]
                weight = weight[:actual_h, :actual_w]
            
            # Add weighted tile to full mask
            full_mask[y:y_end, x:x_end] += tile_mask * weight
            weight_map[y:y_end, x:x_end] += weight
            
            # Update progress
            tile_counter += 1
            progress_bar.progress(tile_counter / total_tiles)
            
            # Update mini-grid visualization
            if j < mini_grid_cols:
                mini_grid[i, j] = 255  # Mark as processed
                # Create RGB visualization of mini-grid
                grid_viz = np.zeros((len(y_steps)*10, mini_grid_cols*10, 3), dtype=np.uint8)
                for i_viz in range(len(y_steps)):
                    for j_viz in range(min(len(x_steps), mini_grid_cols)):
                        color = (0, 255, 0) if mini_grid[i_viz, j_viz] > 0 else (50, 50, 50)
                        grid_viz[i_viz*10:(i_viz+1)*10, j_viz*10:(j_viz+1)*10] = color
                mini_grid_display.image(grid_viz, caption="Tile Processing Status", use_container_width=True)
    
    # Clean up progress displays
    status_text.success("Processing complete!")
    mini_grid_display.empty()
    
    # Normalize the mask by the weight map (add small epsilon to avoid division by zero)
    final_mask = full_mask / (weight_map + 1e-8)
    
    # Apply median filter to remove noise (optional based on image quality)
    if tile_size > 512:  # Only apply to large tiles which might have more boundary artifacts
        try:
            from scipy import ndimage
            final_mask = ndimage.median_filter(final_mask, size=3)
        except ImportError:
            pass  # Skip if scipy not available
    
    # Threshold to binary
    final_mask = (final_mask > 0.5).astype(np.float32)
    
    return final_mask

# Main application
def main():
    # Title and description
    st.title("🔍 Multi-Resolution Segmentation Comparison")
    st.markdown("""
    Compare segmentation results across images of the same location at different resolutions.
    This tool helps identify how resolution affects segmentation quality and accuracy.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Select directories - dynamically find available data directories
        st.subheader("Image Source")
        
        # Find available data directories
        app_dir = os.path.dirname(os.path.abspath(__file__))
        potential_data_dirs = ["data", "old_data", "test_data", "data set 4"]
        available_data_dirs = []
        
        for dir_name in potential_data_dirs:
            full_path = os.path.join(app_dir, dir_name)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                available_data_dirs.append(dir_name)
        
        if not available_data_dirs:
            st.error("No data directories found. Please create a data directory with images.")
            return
        
        data_dir = st.selectbox(
            "Select image directory",
            options=available_data_dirs,
            index=0
        )
        
        # Select model
        model_dir = os.path.join(os.path.dirname(app_dir), 'models')
        if not os.path.exists(model_dir):
            st.error(f"Model directory not found: {model_dir}")
            return
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') or f.endswith('.pt')]
        
        if not model_files:
            st.error("No model files found in the models directory")
            return
            
        selected_model = st.selectbox(
            "Select segmentation model",
            options=model_files,
            index=0
        )
        
        model_path = os.path.join(model_dir, selected_model)
        
        # Visualization options
        st.subheader("Visualization Options")
        viz_mode = st.radio(
            "Comparison view mode",
            options=["Side-by-side", "Overlay", "Difference map", "Boundary comparison", "Heat Map"],
            index=0
        )
        
        # Display options
        show_metrics = st.checkbox("Show comparison metrics", value=True)
        show_zoom = st.checkbox("Show zoomed region", value=True)
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            display_size = st.slider(
                "Display image size", 
                min_value=256, 
                max_value=1024, 
                value=512, 
                step=64
            )
            
            boundary_thickness = st.slider(
                "Boundary thickness", 
                min_value=1, 
                max_value=10, 
                value=3, 
                step=1
            )
            
            # Add tile processing options for high-res images
            st.subheader("High-Resolution Processing")
            use_tiled_processing = st.checkbox(
                "Use tiled processing for high-res images", 
                value=True,
                help="Process large images in tiles for better detection of details"
            )
            
            # Only show these options if tiled processing is enabled
            if use_tiled_processing:
                col1, col2 = st.columns(2)
                with col1:
                    patch_size = st.slider(
                        "Patch Size (px)",
                        min_value=128,
                        max_value=1024,
                        value=512,
                        step=32,
                        help="Size of patches to process. Larger patches need more memory but may detect larger panels better."
                    )
                
                with col2:
                    overlap_percent = st.slider(
                        "Overlap (%)",
                        min_value=10,
                        max_value=50,
                        value=25,
                        step=5,
                        help="Percentage of overlap between patches. Higher overlap helps reduce boundary artifacts."
                    )
                    # Calculate actual overlap in pixels
                    overlap = int(patch_size * overlap_percent / 100)
                
                scale_factor = st.slider(
                    "Resolution (%)",
                    min_value=10,
                    max_value=100,
                    value=60,
                    step=10,
                    help="Percentage of original image resolution to use for processing."
                ) / 100  # Convert percentage to decimal
                
                st.info(f"Using patches of {patch_size}x{patch_size} pixels with {overlap} pixels ({overlap_percent}%) overlap at {scale_factor*100:.0f}% of original resolution")
    
    # Main content
    # Find image groups in the selected directory
    st.header("Image Groups")
    abs_data_dir = os.path.join(app_dir, data_dir)  # Use app_dir from earlier

    if not os.path.exists(abs_data_dir):
        st.error(f"Data directory not found: {abs_data_dir}")
        st.stop()

    # Check if there are any image files in the directory
    image_files_exist = False
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        if glob.glob(os.path.join(abs_data_dir, f"*{ext}")):
            image_files_exist = True
            break

    if not image_files_exist:
        st.warning(f"No image files found in {data_dir} directory. Please add some images with extensions: .png, .jpg, .jpeg, .tif, or .tiff")
        st.stop()

    image_groups, all_images = find_image_groups(abs_data_dir)
    
    # Add manual grouping option
    with st.expander("Manual Image Selection"):
        st.write("""
        If automatic grouping doesn't work well with your image naming conventions,
        you can manually select images to compare.
        """)
        
        # Get all image files for manual selection
        all_image_names = [os.path.basename(img) for img in all_images]
        
        # Allow manual selection of 2-4 images to compare
        manual_selected_images = st.multiselect(
            "Select 2-4 images to compare (must select at least 2)",
            options=all_image_names,
            default=[]
        )
        
        use_manual_selection = st.checkbox(
            "Use manual selection instead of auto-grouped images", 
            value=False,
            help="Enable this to compare the images you selected manually rather than automatically grouped ones"
        )
        
        if use_manual_selection and len(manual_selected_images) < 2:
            st.warning("Please select at least 2 images to compare")
        
        if use_manual_selection and len(manual_selected_images) >= 2:
            # Create a manual group with full paths
            manual_image_paths = [os.path.join(abs_data_dir, img) for img in manual_selected_images]
            image_groups = {"Manual Selection": manual_image_paths}

    if not image_groups:
        st.warning(f"""
        No groups of related images found in '{data_dir}'. 
        You can try:
        1. Using images with similar naming patterns
        2. Using manual selection above
        3. Using a different data directory
        """)
        st.stop()
    
    # Display group selection
    group_names = list(image_groups.keys())
    selected_group = st.selectbox(
        "Select image group to compare",
        options=group_names,
        index=0
    )
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            # Try different model loading approaches
            try:
                # First try loading with weights_only=False
                model = torch.load(model_path, map_location='cpu', weights_only=False)
            except (RuntimeError, TypeError):
                # If that fails, try with weights_only=True
                try:
                    model = torch.load(model_path, map_location='cpu', weights_only=True)
                except TypeError:
                    # If TypeError (weights_only not supported), try without this parameter
                    model = torch.load(model_path, map_location='cpu')
            
            # Ensure model is in eval mode
            model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please try another model or check that the model file is valid.")
        return
    
    # Load and process images
    image_paths = image_groups[selected_group]
    
    # Progress bar for processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process all images in the group
    images_data = []
    
    for i, img_path in enumerate(image_paths):
        status_text.text(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Load image
        img_data = load_and_preprocess_image(img_path, target_size=(display_size, display_size))
        if img_data is None:
            continue
        
        # Check if image is high-resolution and needs tiled processing
        original_image = img_data["original"]
        is_highres = (max(original_image.shape[0], original_image.shape[1]) > 1000 or 
                    "4x" in img_data["filename"] or "10x" in img_data["filename"])
        
        # Run segmentation
        if is_highres and use_tiled_processing:
            # For tiled processing, first scale the image if needed
            if scale_factor < 1.0:
                h, w = original_image.shape[:2]
                processing_h = int(h * scale_factor)
                processing_w = int(w * scale_factor)
                processing_img = cv2.resize(original_image, (processing_w, processing_h), interpolation=cv2.INTER_AREA)
            else:
                processing_img = original_image
            
            # Process image in tiles
            st.info(f"Processing {img_data['filename']} with tiled approach (size: {processing_img.shape[1]}x{processing_img.shape[0]})")
            mask = process_image_in_tiles(model, processing_img, tile_size=patch_size, overlap=overlap)
            
            # Resize mask back to original size if needed
            if scale_factor < 1.0:
                mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            # Standard processing for non-high-res images
            mask = segment_image(_model=model, image=original_image)
        
        # Add to our dataset
        img_data["mask"] = mask
        img_data["resolution"] = extract_resolution_from_filename(img_data["filename"], img_path)
        img_data["is_highres"] = is_highres
        img_data["processing_method"] = "tiled" if (is_highres and use_tiled_processing) else "standard"
        images_data.append(img_data)
        
        # Update progress
        progress_bar.progress((i + 1) / len(image_paths))
    
    status_text.empty()
    progress_bar.empty()
    
    # If we have at least 2 images, we can do comparison
    if len(images_data) < 2:
        st.error("Need at least 2 related images to compare segmentation results.")
        return
    
    # Sort images by resolution (if possible)
    def get_sort_key(img_data):
        res = img_data["resolution"]
        
        # For specialized model+resolution formats (like esrgan_4x)
        model_res_match = re.match(r'([a-zA-Z]+)_(\d+)x', res)
        if model_res_match:
            res_val = int(model_res_match.group(2))
            # Higher resolution value = higher priority
            return 100 + res_val
        
        # For S2L2A_Nx format
        s2l2a_match = re.match(r'S2L2A_(\d+)x', res)
        if s2l2a_match:
            res_val = int(s2l2a_match.group(1))
            # Higher resolution value = higher priority
            return 100 + res_val
        
        # For numeric resolutions
        if res.isdigit():
            return int(res)
        # For text-based resolutions
        elif res == "High":
            return 1000
        elif res == "Original":
            return 900
        elif res == "Large":
            return 800
        elif res == "Medium":
            return 500
        elif res == "Low" or res == "Small":
            return 100
        # Default for unknown
        return 0

    # Sort by resolution (higher values first)
    images_data.sort(key=get_sort_key, reverse=True)
    
    # Add debug information
    st.sidebar.text("Image resolutions detected:")
    for img in images_data:
        st.sidebar.text(f" - {img['filename']}: {img['resolution']}")

    # Display comparative visualizations
    st.header("Segmentation Comparison")
    
    # Add summary of processing methods
    st.subheader("Processing Configuration")
    processing_summary = {
        "Standard": 0,
        "Tiled": 0
    }

    for img_data in images_data:
        method = img_data.get("processing_method", "standard")
        processing_summary[method.capitalize()] += 1

    # Create processing summary columns
    summary_cols = st.columns(len(processing_summary) + 1)
    with summary_cols[0]:
        st.metric("Total Images", len(images_data))

    for i, (method, count) in enumerate(processing_summary.items(), 1):
        with summary_cols[i]:
            st.metric(f"{method} Processing", count)

    if processing_summary["Tiled"] > 0:
        st.info(f"Tiled processing parameters: {patch_size}px patches with {overlap_percent}% overlap at {int(scale_factor*100)}% resolution")

    # Add explanatory text about the images
    st.markdown(f"""
    Comparing {len(images_data)} images of the same location at different resolutions:
    {', '.join([f"{img['filename']} ({img['dimensions'][1]}×{img['dimensions'][0]}px)" for img in images_data])}
    """)
    
    # Display the images with their segmentation masks
    if viz_mode == "Side-by-side":
        # Create columns based on number of images
        cols = st.columns(len(images_data))
        
        for i, (col, img_data) in enumerate(zip(cols, images_data)):
            with col:
                # Display original image
                st.subheader(f"Image {i+1}: {img_data['resolution']}x")
                st.image(img_data["display"], caption=f"Original {img_data['dimensions'][1]}×{img_data['dimensions'][0]}px")
                
                # Display processing method info
                if img_data.get("is_highres", False):
                    if img_data.get("processing_method") == "tiled":
                        st.info(f"Processing: Tiled ({patch_size}px patches, {overlap_percent}% overlap, {int(scale_factor*100)}% res)")
                    else:
                        st.info("Processing: Standard")
                
                # Display segmentation mask
                display_mask = cv2.resize(img_data["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
                mask_overlay = img_data["display"].copy()
                mask_overlay[display_mask > 0.5] = mask_overlay[display_mask > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3
                st.image(mask_overlay, caption="Segmentation Result")
    
    elif viz_mode == "Overlay":
        # Compare the first (highest resolution) image with each other image
        reference_img = images_data[0]
        st.subheader(f"Reference Image: {reference_img['resolution']}x")
        
        for i, img_data in enumerate(images_data[1:], 1):
            st.markdown(f"#### Comparison with {img_data['resolution']}x image")
            
            # Create overlaid visualization
            heatmap = create_heat_map_visualization(
                reference_img["display"],
                img_data["display"],
                cv2.resize(reference_img["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST),
                cv2.resize(img_data["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(reference_img["display"], caption=f"Reference: {reference_img['resolution']}x")
            with col2:
                st.image(img_data["display"], caption=f"Compare: {img_data['resolution']}x")
            
            st.image(heatmap, caption=f"Overlay (Green: Both, Red: {reference_img['resolution']}x only, Blue: {img_data['resolution']}x only)")
            
            # Add interactive slider comparison
            st.subheader("Interactive Comparison")
            overlay1, overlay2 = create_slider_comparison(
                reference_img["display"],
                img_data["display"],
                cv2.resize(reference_img["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST),
                cv2.resize(img_data["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            )
            
            # Create a container for the slider image
            slider_container = st.container()
            slider_value = st.slider(
                "Slide to compare images",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                key=f"slider_{i}"
            )
            
            # Blend images based on slider position
            blend_factor = slider_value / 100.0
            blended = cv2.addWeighted(overlay1, 1 - blend_factor, overlay2, blend_factor, 0)
            
            # Display the blended image
            with slider_container:
                st.image(blended, caption=f"Comparison slider ({100-slider_value}% reference, {slider_value}% comparison)")
            
            # Add a separator
            if i < len(images_data) - 1:
                st.markdown("---")
    
    elif viz_mode == "Difference map":
        # Create a difference map for all pairs of images
        st.subheader("Segmentation Difference Maps")
        
        for i in range(len(images_data)):
            for j in range(i+1, len(images_data)):
                img1 = images_data[i]
                img2 = images_data[j]
                
                # Create difference visualization
                mask1 = cv2.resize(img1["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
                mask2 = cv2.resize(img2["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
                diff_viz = create_difference_mask(mask1, mask2)
                
                # Display
                st.markdown(f"#### {img1['resolution']}x vs {img2['resolution']}x")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mask1_overlay = img1["display"].copy()
                    mask1_overlay[mask1 > 0.5] = mask1_overlay[mask1 > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3
                    st.image(mask1_overlay, caption=f"{img1['resolution']}x segmentation")
                
                with col2:
                    mask2_overlay = img2["display"].copy()
                    mask2_overlay[mask2 > 0.5] = mask2_overlay[mask2 > 0.5] * 0.7 + np.array([0, 0, 255]) * 0.3
                    st.image(mask2_overlay, caption=f"{img2['resolution']}x segmentation")
                
                with col3:
                    st.image(diff_viz, caption="Difference Map")
                
                # Add detailed metrics for this pair
                metrics = calculate_segmentation_metrics(mask1 > 0.5, mask2 > 0.5)
                st.markdown(f"""
                **Segmentation Similarity Metrics**:
                - IoU (Intersection over Union): {metrics['iou']:.4f}
                - Dice Coefficient: {metrics['dice']:.4f}
                - Area Difference: {metrics['area_diff_percent']:.2f}%
                """)
                
                # Add a separator
                if i < len(images_data) - 1 or j < len(images_data) - 1:
                    st.markdown("---")
    
    elif viz_mode == "Boundary comparison":
        # Focus on boundaries
        st.subheader("Boundary Comparison")
        
        # Compare the first (highest resolution) image with each other image
        reference_img = images_data[0]
        
        for i, img_data in enumerate(images_data[1:], 1):
            st.markdown(f"#### {reference_img['resolution']}x vs {img_data['resolution']}x Boundaries")
            
            # Resize masks for display
            mask1 = cv2.resize(reference_img["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.resize(img_data["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            
            # Create boundary visualizations
            boundary1 = create_boundary_visualization(mask1, kernel_size=boundary_thickness)
            boundary2 = create_boundary_visualization(mask2, kernel_size=boundary_thickness)
            
            # Combine boundaries
            combined_viz = reference_img["display"].copy()
            # Yellow for reference boundaries
            combined_viz[boundary1 > 0] = [255, 255, 0]
            # Cyan for comparison boundaries
            combined_viz[boundary2 > 0] = [0, 255, 255]
            # White for overlapping boundaries
            combined_viz[np.logical_and(boundary1 > 0, boundary2 > 0)] = [255, 255, 255]
            
            col1, col2 = st.columns(2)
            with col1:
                # Show reference image with its boundary
                ref_with_boundary = reference_img["display"].copy()
                ref_with_boundary[boundary1 > 0] = [255, 255, 0]  # Yellow boundary
                st.image(ref_with_boundary, caption=f"{reference_img['resolution']}x boundaries")
            
            with col2:
                # Show comparison image with its boundary
                comp_with_boundary = img_data["display"].copy()
                comp_with_boundary[boundary2 > 0] = [0, 255, 255]  # Cyan boundary
                st.image(comp_with_boundary, caption=f"{img_data['resolution']}x boundaries")
            
            # Show combined visualization
            st.image(combined_viz, caption="Combined boundaries (Yellow: Reference, Cyan: Comparison, White: Overlap)")
            
            # Add a separator
            if i < len(images_data) - 1:
                st.markdown("---")
    
    elif viz_mode == "Heat Map":
        # Create precision-recall heatmaps
        st.subheader("Segmentation Heat Maps")
        st.write("""
        This visualization shows precision-recall relationships between different resolutions.
        - **Green**: True Positives (areas detected in both)
        - **Red**: False Positives (areas only in the first image)
        - **Blue**: False Negatives (areas only in the second image)
        """)
        
        # Use the highest resolution as reference
        reference_img = images_data[0]
        
        for i, img_data in enumerate(images_data[1:], 1):
            st.markdown(f"#### {reference_img['resolution']}x vs {img_data['resolution']}x Heat Map")
            
            # Create precision-recall visualization
            mask1 = cv2.resize(reference_img["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.resize(img_data["mask"], (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            
            heatmap, precision, recall, f1 = create_precision_recall_heatmap(mask1, mask2)
            
            # Display in columns
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Display metrics
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
                
                # Add explanation
                st.write("""
                **Metrics explained**:
                - **Precision**: What percentage of detected panels in the reference image are correct
                - **Recall**: What percentage of actual panels were detected in the reference image
                - **F1 Score**: Harmonic mean of precision and recall
                """)
            
            with col2:
                # Display heatmap
                st.image(heatmap, caption="Precision-Recall Heatmap", use_container_width=True)
            
            with col3:
                # Show original images for reference
                st.image(reference_img["display"], caption=f"Reference: {reference_img['resolution']}x", use_container_width=True)
                st.image(img_data["display"], caption=f"Compare: {img_data['resolution']}x", use_container_width=True)
            
            # Add a separator
            if i < len(images_data) - 1:
                st.markdown("---")
    
    # Show metrics table if enabled
    if show_metrics:
        st.header("Comparison Metrics")
        metrics_df = compute_metrics_table(images_data)
        st.dataframe(metrics_df)
    
    # Show zoomed region if enabled
    if show_zoom and len(images_data) >= 2:
        st.header("Zoomed Region Comparison")
        st.write("Examine a specific region more closely:")
        
        # Allow user to select reference image for zooming
        ref_index = st.selectbox(
            "Select reference image for zoom",
            options=range(len(images_data)),
            format_func=lambda i: f"{images_data[i]['filename']} ({images_data[i]['resolution']}x)",
            index=0
        )
        
        ref_img = images_data[ref_index]
        ref_mask = ref_img["mask"]
        
        # Find a region with panels
        y_indices, x_indices = np.where(ref_mask > 0.5)
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            # Find center of a solar panel as our region of interest
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            
            # Add sliders for adjusting zoom position
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                zoom_x = st.slider(
                    "X position",
                    min_value=0,
                    max_value=ref_img["original"].shape[1] - 1,
                    value=center_x,
                    step=10
                )
            with col2:
                zoom_size = st.slider(
                    "Zoom size",
                    min_value=50,
                    max_value=min(500, min(ref_img["original"].shape[:2])),
                    value=200,
                    step=25
                )
            with col3:
                zoom_y = st.slider(
                    "Y position",
                    min_value=0,
                    max_value=ref_img["original"].shape[0] - 1,
                    value=center_y,
                    step=10
                )
            
            # Calculate zoom region
            y1 = max(0, zoom_y - zoom_size//2)
            y2 = min(ref_img["original"].shape[0], zoom_y + zoom_size//2)
            x1 = max(0, zoom_x - zoom_size//2)
            x2 = min(ref_img["original"].shape[1], zoom_x + zoom_size//2)
            
            # Display zoomed regions for all images
            cols = st.columns(len(images_data))
            
            for i, (col, img_data) in enumerate(zip(cols, images_data)):
                with col:
                    # Scale coordinates to match this image's dimensions
                    scale_y = img_data["original"].shape[0] / ref_img["original"].shape[0]
                    scale_x = img_data["original"].shape[1] / ref_img["original"].shape[1]
                    
                    img_y1 = int(y1 * scale_y)
                    img_y2 = int(y2 * scale_y)
                    img_x1 = int(x1 * scale_x)
                    img_x2 = int(x2 * scale_x)
                    
                    # Ensure bounds
                    img_y1 = max(0, img_y1)
                    img_y2 = min(img_data["original"].shape[0], img_y2)
                    img_x1 = max(0, img_x1)
                    img_x2 = min(img_data["original"].shape[1], img_x2)
                    
                    # Extract zoom region
                    zoom_img = img_data["original"][img_y1:img_y2, img_x1:img_x2]
                    zoom_mask = img_data["mask"][img_y1:img_y2, img_x1:img_x2]
                    
                    # Create visualization
                    zoom_viz = zoom_img.copy()
                    zoom_viz[zoom_mask > 0.5] = zoom_viz[zoom_mask > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3
                    
                    # Show
                    st.image(zoom_viz, caption=f"{img_data['resolution']}x zoom")
        else:
            st.warning("No segmentation found in the reference image to center zoom")

if __name__ == "__main__":
    main() 