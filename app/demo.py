import os
import torch
import streamlit as st
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
# Use the installed segmentation models rather than the local one
import segmentation_models_pytorch as smp
from utils import *

# Define helper functions
def transpose_and_convert_to_float32(img, **kwargs):
    """Helper function to transpose image and convert to float32 (avoids lambda warning)"""
    return img.transpose(2, 0, 1).astype('float32')

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

    st.write(f"Processing {total_tiles} tiles (size: {tile_size}px, overlap: {overlap}px)...")
    
    for i, y in enumerate(y_steps):
        for j, x in enumerate(x_steps):
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

# Page configuration
st.set_page_config(
    page_title='Solar Panel Segmentation Demo',
    page_icon="https://api.iconify.design/openmoji/solar-energy.svg?width=500",
    layout='wide'
)

# Header
st.write("""
# :sunny: Solar Panel Segmentation Demo
Demonstration of PV (photovoltaic) panel segmentation using deep learning.
""")

# Load images from data directory
img_dir = 'data'
img_files = list(filter(lambda x: 'label' not in x, os.listdir(img_dir)))

# Check for old_data directory and add those images too
old_data_dir = 'old_data'
old_data_path = os.path.join('app', old_data_dir)
if os.path.exists(old_data_path) and os.path.isdir(old_data_path):
    old_img_files = list(filter(lambda x: 'label' not in x and x.endswith('.png'), os.listdir(old_data_path)))
    if old_img_files:
        st.sidebar.info(f"Found {len(old_img_files)} images in old_data directory")
        # Add a prefix to differentiate the old data files
        old_img_files = [f"[OLD] {img}" for img in old_img_files]
        img_files.extend(old_img_files)

# Get model paths
model_dir = '../models'
model_files = list(filter(lambda x: x.endswith('.pth' or '.pt'), os.listdir(model_dir)))

# Interface
col_setup, col_results = st.columns([1, 3])

with col_setup:
    st.subheader("Select an image and model")
    
    # Image selection
    selected_img = st.selectbox(
        'Choose an image to analyze:',
        img_files,
        index=1
    )
    
    # Model selection
    selected_model = st.selectbox(
        'Choose a segmentation model:',
        model_files,
        index=0
    )
    
    # Settings
    st.subheader("Settings")
    
    # Add resolution control
    img_resolution = st.slider(
        "Image display resolution:",
        min_value=256,
        max_value=1024,
        value=512,
        step=128,
        help="Higher resolution shows more detail but may be slower"
    )
    
    # Add quality setting for high-res images
    quality_option = st.radio(
        "Processing quality for high-res images:",
        ["Standard", "High", "Maximum"],
        index=1,  # Default to High
        help="Higher quality uses more of the original resolution but takes longer to process"
    )
    
    # Add preprocessing strength control
    preserve_details = st.checkbox(
        "Preserve image details (for high-res images)",
        value=True,
        help="Enable for 4x and 10x images to preserve more details"
    )
    
    display_option = st.radio(
        "Display mode:",
        ["Original image", "Binary mask", "Overlay", "Bounding boxes"]
    )
    
    st.info("This demo uses pre-loaded images from the dataset. No upload is required.")

# Process the image
if selected_img and selected_model:
    # Paths
    if selected_img.startswith("[OLD] "):
        # Handle old data images
        real_img_name = selected_img[6:]  # Remove the "[OLD] " prefix
        img_path = os.path.join('app', old_data_dir, real_img_name)
        # Fix: Use os.path.splitext to handle any file extension properly
        base_path, ext = os.path.splitext(img_path)
        gt_mask_path = f"{base_path}_label.png"
        st.info(f"Using image from old dataset: {real_img_name}")
    else:
        # Regular images from data directory
        img_path = os.path.join(img_dir, selected_img)
        # Fix: Use os.path.splitext to handle any file extension properly
        base_path, ext = os.path.splitext(img_path)
        gt_mask_path = f"{base_path}_label.png"
    
    # Set model path (common for all image types)
    model_path = os.path.join(model_dir, selected_model)
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        # TIF files may need special handling
        try:
            import tifffile
            st.info(f"Attempting to load TIF file using tifffile: {img_path}")
            image = tifffile.imread(img_path)
            # Convert to BGR if needed (tifffile loads as RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Failed to load image: {img_path}. Error: {e}")
            st.stop()

    if image is None:
        st.error(f"Failed to load image: {img_path}. The file may be corrupted or too large.")
        st.stop()
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original image for display
    original_image = image.copy()
    
    # Display size for UI (doesn't affect processing)
    display_size = (img_resolution, img_resolution)
    
    # Adaptive preprocessing based on image name and settings
    is_highres_image = "4x" in selected_img or "10x" in selected_img or "x10" in selected_img
    if is_highres_image or max(original_image.shape[0], original_image.shape[1]) > 1000:
        st.info(f"Processing high-resolution image: {selected_img}")
        
        # Special handling for high-resolution images
        use_tiled_processing = st.checkbox(
            "Use tiled processing (recommended for high-res images)",
            value=True,
            help="Process image in smaller tiles for better detection of small details"
        )
        
        # Keep higher resolution for display
        image_for_display = cv2.resize(original_image, display_size, interpolation=cv2.INTER_AREA)
        
        if use_tiled_processing:
            # For tiled processing, we'll load and process the model early
            # The full-size processing will happen after model loading
            st.info("Using tiled processing for high-resolution image...")
            # We'll set a flag to use tiled processing later after model loading
            do_tiled_processing = True
            
            # Still need a standard version for regular processing pipeline
            model_input_size = (256, 256)  # Define model_input_size even when using tiled processing
            image_for_model = cv2.resize(image, model_input_size, interpolation=cv2.INTER_AREA)
        else:
            # Standard processing with detail preservation
            model_input_size = (256, 256)
            image_for_model = cv2.resize(image, model_input_size, interpolation=cv2.INTER_AREA)
            do_tiled_processing = False
    else:
        # Standard processing for regular images
        model_input_size = (256, 256)
        image_for_model = cv2.resize(image, model_input_size)
        image_for_display = cv2.resize(original_image, display_size)
        do_tiled_processing = False
    
    # From here on, use image_for_model for prediction and image_for_display for UI
    image = image_for_model
    
    # Load ground truth mask if available
    has_gt = os.path.exists(gt_mask_path)
    if has_gt:
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)[1]
        gt_mask = cv2.resize(gt_mask, (256, 256))
    
    # Process for model input
    # Apply augmentations
    test_transform = [
        A.Resize(model_input_size[0], model_input_size[1]),  # Use dynamic size
        A.PadIfNeeded(model_input_size[0], model_input_size[1])  # Use dynamic size
    ]
    augmentation = A.Compose(test_transform)
    sample = augmentation(image=image)
    proc_image = sample['image']
    
    # Load model
    with st.spinner("Loading model and running inference..."):
        try:
            # First try with weights_only=False
            try:
                model = torch.load(model_path, map_location='cpu', weights_only=False)
            except RuntimeError as e:
                if "PytorchStreamReader failed reading zip archive" in str(e):
                    st.warning("Detected issue with model file, trying alternative loading method...")
                    
                    # Try loading with different parameters based on model type
                    if "unetplusplus" in selected_model:
                        # Just show an error message for UNet++ model
                        st.error("The UNet++ model file appears to be corrupted.")
                        st.error("Please add a valid UNet++ model file to the models directory.")
                        st.stop()
                    else:
                        # For other models, just retry with different options
                        try:
                            model = torch.load(model_path, map_location='cpu', weights_only=True)
                        except Exception as retry_e:
                            st.error(f"Failed to load model: {retry_e}")
                            st.error("Please try another model.")
                            st.stop()
                else:
                    # Different error, just report it
                    st.error(f"Error loading model: {e}")
                    st.error("Please try another model.")
                    st.stop()
            
            # Basic preprocessing (without caching issues)
            device = 'cpu'
            
            # Get preprocessing parameters
            if 'resnext' in selected_model.lower():
                backbone = 'se_resnext101_32x4d'
            elif 'efficient' in selected_model.lower():
                backbone = 'efficientnet-b3'
            else:
                backbone = 'se_resnext101_32x4d'  # default
            
            preprocess_input = smp.encoders.get_preprocessing_fn(backbone)
            
            # Apply preprocessing
            _transform = [
                A.Lambda(image=preprocess_input),
                A.Lambda(image=transpose_and_convert_to_float32),
            ]
            preprocessing = A.Compose(_transform)
            sample = preprocessing(image=proc_image)
            proc_image = sample['image']
            
            # Run prediction
            image_torch = torch.from_numpy(proc_image).to(device).unsqueeze(0)
            
            # Special handling for high-res images with tiled processing
            if do_tiled_processing:
                st.info("Running tiled processing on high-resolution image...")
                # Get downsampled version for tiled processing (to keep memory usage reasonable)
                # Adjust scale factor based on image size and resolution
                max_dim = max(original_image.shape[0], original_image.shape[1])
                
                # Image type-specific settings
                if "x10" in selected_img or "10x" in selected_img:
                    # Settings for 10x images (4320x4340)
                    scale_factor = 0.6  # Use 60% of original size
                    tile_size = 768     # Use larger tiles
                    overlap = 192       # Use 25% overlap
                elif "4x" in selected_img:
                    # Settings for 4x images (3584x3584)
                    scale_factor = 0.7  # Use 70% of original size
                    tile_size = 640     # Medium-sized tiles
                    overlap = 160       # 25% overlap
                elif max_dim < 512:
                    # For old PV01/PV08 images (256x256) - no need for tiling but keep the process
                    scale_factor = 1.0  # No scaling needed
                    tile_size = 256     # Original size
                    overlap = 64        # Small overlap for 256px images
                elif max_dim < 1000:
                    # For small to medium images like the 432x434 standard image
                    scale_factor = 1.0  # No scaling needed
                    tile_size = 384     # Slightly larger than the model input
                    overlap = 96        # 25% overlap
                else:
                    # Default for large images
                    if max_dim > 4000:
                        scale_factor = 0.5  # Half size for very large
                        tile_size = 512
                    elif max_dim > 2000:
                        scale_factor = 0.7  # 70% for large
                        tile_size = 512
                    else:
                        scale_factor = 0.9  # 90% for medium-large
                        tile_size = 384
                    overlap = int(tile_size * 0.25)  # 25% overlap
                
                # Show processing details
                st.info(f"Processing with: scale={scale_factor:.1f}, tile size={tile_size}px, overlap={overlap}px, image size={original_image.shape[1]}x{original_image.shape[0]}px")
                
                # Scale the image for tiled processing
                processing_h = int(original_image.shape[0] * scale_factor)
                processing_w = int(original_image.shape[1] * scale_factor) 
                processing_img = cv2.resize(original_image, (processing_w, processing_h), interpolation=cv2.INTER_AREA)
                
                # Process the image in tiles
                pr_mask = process_image_in_tiles(model, processing_img, tile_size=tile_size, overlap=overlap)
            else:
                # Regular processing for normal images
                pr_mask = model.predict(image_torch)
                pr_mask = (pr_mask.squeeze().numpy().round())
            
            # Calculate bounding boxes
            boxes = compute_bboxes(image, pr_mask)
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error("Please try another model.")
            st.stop()
    
    # Results section
    with col_results:
        # Display columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_for_display, caption=f"Original: {selected_img}", use_container_width=True)
            
            # Show detailed resolution info for high-res images
            is_highres = "4x" in selected_img or "10x" in selected_img or max(original_image.shape) > 1000
            if is_highres:
                st.subheader("Resolution Information")
                st.write(f"**Original Resolution**: {original_image.shape[1]}x{original_image.shape[0]} pixels")
                if do_tiled_processing:
                    processing_h = int(original_image.shape[0] * scale_factor)
                    processing_w = int(original_image.shape[1] * scale_factor)
                    
                    # Calculate total tiles count - similar to calculation in process_image_in_tiles
                    h, w = processing_h, processing_w
                    # Calculate steps with overlap
                    x_steps = list(range(0, w - tile_size + 1, tile_size - overlap))
                    if len(x_steps) == 0 or x_steps[-1] + tile_size < w:
                        x_steps = list(x_steps) + [max(0, w - tile_size)]
                    
                    y_steps = list(range(0, h - tile_size + 1, tile_size - overlap))
                    if len(y_steps) == 0 or y_steps[-1] + tile_size < h:
                        y_steps = list(y_steps) + [max(0, h - tile_size)]
                    
                    total_tiles = len(x_steps) * len(y_steps)
                    
                    st.write(f"**Processing Resolution**: {processing_w}x{processing_h} pixels ({scale_factor:.2f}x scale)")
                    st.write(f"**Tile Size**: {tile_size}x{tile_size} pixels with {overlap}px overlap")
                    st.write(f"**Total Tiles Processed**: {total_tiles}")
            
            # Show metrics if ground truth is available
            if has_gt:
                st.subheader("Performance Metrics")
                
                # For high-resolution images, we need to ensure masks are comparable size
                if do_tiled_processing and is_highres:
                    # Get original shape of both masks for informational purposes
                    orig_pr_shape = pr_mask.shape
                    orig_gt_shape = gt_mask.shape
                    
                    # Resize the prediction mask to match ground truth size for metrics
                    # (We'll keep the original high-res mask for display/visualization)
                    pr_mask_for_metrics = cv2.resize(pr_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                                   interpolation=cv2.INTER_NEAREST)
                    
                    st.info(f"Resized prediction mask from {orig_pr_shape} to {pr_mask_for_metrics.shape} to match ground truth for metrics calculation.")
                    
                    # Use the resized mask for metrics
                    iou_score = compute_iou(gt_mask, pr_mask_for_metrics)
                    pixel_acc = compute_pixel_acc(gt_mask, pr_mask_for_metrics)
                    dice_coef = compute_dice_coeff(gt_mask, pr_mask_for_metrics)
                else:
                    # Standard processing - masks are already the same size
                    iou_score = compute_iou(gt_mask, pr_mask)
                    pixel_acc = compute_pixel_acc(gt_mask, pr_mask)
                    dice_coef = compute_dice_coeff(gt_mask, pr_mask)
                
                st.write(f"**IoU Score**: {iou_score}")
                st.write(f"**Pixel Accuracy**: {pixel_acc}")
                st.write(f"**Dice Coefficient (F1)**: {dice_coef}")
                
                # Add high-resolution specific metrics - boundary precision 
                if is_highres and quality_option != "Standard":
                    st.subheader("High-Resolution Metrics")
                    st.write("**Boundary Precision**: These metrics reflect how well boundaries are defined")
                    
                    # Calculate boundary F1-score (if scipy is available)
                    try:
                        from scipy import ndimage
                        
                        # For high-res images, we need to use the resized mask for comparison
                        if do_tiled_processing and is_highres:
                            boundary_pr_mask = pr_mask_for_metrics
                        else:
                            boundary_pr_mask = pr_mask
                        
                        # Get boundaries of prediction and ground truth
                        # Ensure we're working with boolean arrays for the XOR operation
                        dilated_pr = ndimage.binary_dilation(boundary_pr_mask.astype(bool))
                        pr_boundary = dilated_pr ^ boundary_pr_mask.astype(bool)
                        
                        dilated_gt = ndimage.binary_dilation(gt_mask.astype(bool))
                        gt_boundary = dilated_gt ^ gt_mask.astype(bool)
                        
                        # Calculate precision, recall for boundaries
                        boundary_precision = np.sum(pr_boundary & gt_boundary) / (np.sum(pr_boundary) + 1e-8)
                        boundary_recall = np.sum(pr_boundary & gt_boundary) / (np.sum(gt_boundary) + 1e-8)
                        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-8)
                        
                        st.write(f"**Boundary F1-score**: {boundary_f1:.4f}")
                        st.write(f"**Boundary Precision**: {boundary_precision:.4f}")
                        st.write(f"**Boundary Recall**: {boundary_recall:.4f}")
                        
                        # Show small boundary visualization
                        boundary_viz = np.zeros((256, 256, 3), dtype=np.uint8)
                        # Red = prediction boundary, Green = ground truth boundary, Yellow = overlap
                        boundary_viz[pr_boundary.astype(bool), 0] = 255  # Red channel
                        boundary_viz[gt_boundary.astype(bool), 1] = 255  # Green channel
                        st.image(cv2.resize(boundary_viz, (256, 256)), caption="Boundary comparison (Red=Prediction, Green=GT, Yellow=Match)", use_container_width=True)
                        
                    except ImportError:
                        st.info("Install scipy for additional boundary metrics")
        
        with col2:
            st.subheader("Segmentation Results")
            
            # Display according to selected mode
            if display_option == "Original image":
                st.image(image_for_display, caption="Original Image", use_container_width=True)
                
            elif display_option == "Binary mask":
                # Resize mask to display size for better visualization
                display_mask = cv2.resize(pr_mask, display_size, interpolation=cv2.INTER_NEAREST)
                st.image(display_mask, caption="Binary Mask (White = Solar Panels)", use_container_width=True)
                
            elif display_option == "Overlay":
                # Create colored overlay on display-sized image
                display_mask = cv2.resize(pr_mask, display_size, interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(image_for_display)
                colored_mask[:,:,0] = display_mask * 255  # Red channel
                overlay = cv2.addWeighted(image_for_display, 0.7, colored_mask, 0.3, 0)
                st.image(overlay, caption="Solar Panels Overlay", use_container_width=True)
                
            elif display_option == "Bounding boxes":
                # Calculate bounding boxes at model resolution
                boxes = compute_bboxes(image, pr_mask)
                
                # Scale boxes to display resolution
                scale_x = display_size[0] / 256
                scale_y = display_size[1] / 256
                scaled_boxes = []
                for box in boxes:
                    scaled_box = [
                        int(box[0] * scale_y), 
                        int(box[1] * scale_x), 
                        int(box[2] * scale_y), 
                        int(box[3] * scale_x)
                    ]
                    scaled_boxes.append(scaled_box)
                
                # Draw bounding boxes on display image
                image_bboxes = image_for_display.copy()
                for box in scaled_boxes:
                    image_bboxes = cv2.rectangle(image_bboxes, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                st.image(image_bboxes, caption="Solar Panels with Bounding Boxes", use_container_width=True)
            
            # For high-res images, add a detail comparison section with different quality settings
            if is_highres and do_tiled_processing:
                st.subheader("Resolution Comparison")
                st.write("Examine a magnified region to appreciate resolution difference:")
                
                # Create a zoomed-in view of a region of interest
                roi_col1, roi_col2 = st.columns(2)
                
                # Find a region with panels (use the mask to find a good area to zoom in)
                y_indices, x_indices = np.where(pr_mask > 0.5)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    # Find center of a solar panel as our region of interest
                    center_y = int(np.mean(y_indices))
                    center_x = int(np.mean(x_indices))
                    
                    # Calculate zoom region coordinates
                    zoom_size = min(256, pr_mask.shape[0]//4, pr_mask.shape[1]//4)
                    zoom_y1 = max(0, center_y - zoom_size//2)
                    zoom_y2 = min(pr_mask.shape[0], center_y + zoom_size//2)
                    zoom_x1 = max(0, center_x - zoom_size//2)
                    zoom_x2 = min(pr_mask.shape[1], center_x + zoom_size//2)
                    
                    # Extract zoom region from original high-resolution image (scale coordinates)
                    orig_zoom_y1 = int(zoom_y1 / scale_factor)
                    orig_zoom_y2 = int(zoom_y2 / scale_factor)
                    orig_zoom_x1 = int(zoom_x1 / scale_factor)
                    orig_zoom_x2 = int(zoom_x2 / scale_factor)
                    
                    # Ensure we don't go out of bounds
                    orig_zoom_y1 = max(0, min(orig_zoom_y1, original_image.shape[0]-1))
                    orig_zoom_y2 = max(0, min(orig_zoom_y2, original_image.shape[0]))
                    orig_zoom_x1 = max(0, min(orig_zoom_x1, original_image.shape[1]-1))
                    orig_zoom_x2 = max(0, min(orig_zoom_x2, original_image.shape[1]))
                    
                    # Extract zoomed regions
                    zoom_img = processing_img[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
                    zoom_mask = pr_mask[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
                    orig_zoom_img = original_image[orig_zoom_y1:orig_zoom_y2, orig_zoom_x1:orig_zoom_x2]
                    
                    # Create overlay for zoomed region
                    zoom_overlay = zoom_img.copy()
                    zoom_overlay[zoom_mask > 0.5, 0] = 255  # Red channel for segmentation
                    
                    with roi_col1:
                        st.write(f"**Processing Resolution ({scale_factor:.2f}x scale)**")
                        st.image(zoom_img, caption="Zoomed processing image", use_container_width=True)
                        st.image(zoom_overlay, caption="Segmentation overlay", use_container_width=True)
                    
                    with roi_col2:
                        st.write("**Original Resolution (1.0x scale)**")
                        st.image(orig_zoom_img, caption="Original resolution", use_container_width=True)
                        # Create a comparable overlay for the original image
                        resized_mask = cv2.resize(zoom_mask, (orig_zoom_img.shape[1], orig_zoom_img.shape[0]), 
                                                 interpolation=cv2.INTER_NEAREST)
                        orig_overlay = orig_zoom_img.copy()
                        orig_overlay[resized_mask > 0.5, 0] = 255  # Red channel for segmentation
                        st.image(orig_overlay, caption="Segmentation projected to original resolution", use_container_width=True)
                    
                    # Add explanation
                    st.write("""
                    This comparison shows how the segmentation appears at both the processing resolution and 
                    the original high-resolution image. Higher quality settings preserve more details but take longer to process.
                    """)
                else:
                    st.info("No solar panels detected for zoom comparison")
        
        # Area calculations
        st.subheader("Area Calculations")
        area = np.round(float(img_resolution) ** 2 * pr_mask.sum(), 4)
        perc_area = np.round(100 * pr_mask.sum() / img_resolution ** 2, 2)
        total_area = np.round(img_resolution ** 2 * float(img_resolution) ** 2, 2)
        
        st.write(f"**Total Solar Panel Area**: {area} square meters")
        st.write(f"**Coverage Percentage**: {perc_area}% of the image")
        st.write(f"**Image Area**: {total_area} square meters")
        
        # Coordinates
        st.subheader("Panel Coordinates")
        coordinates_dict = return_coordinates(boxes)
        coor_values = list(coordinates_dict.values())
        num_bboxes = len(coor_values)
        
        if num_bboxes > 0:
            st.write(f"**Number of detected panels/groups**: {num_bboxes}")
            
            # Create a dataframe for coordinates for cleaner display
            import pandas as pd
            coords_data = []
            
            for i in range(num_bboxes):
                coords_data.append({
                    "Panel #": i+1,
                    "Top Left": str(coor_values[i][0]),
                    "Top Right": str(coor_values[i][1]),
                    "Bottom Left": str(coor_values[i][2]),
                    "Bottom Right": str(coor_values[i][3]),
                })
            
            if coords_data:
                st.dataframe(pd.DataFrame(coords_data))
        else:
            st.write("No solar panels detected in this image.") 