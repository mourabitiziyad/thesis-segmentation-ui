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

# Load images and models data
img_dir = 'data'
# Filter out hidden files, system files, and non-image files
valid_image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
img_files = [f for f in os.listdir(img_dir) 
             if not f.startswith('.') and  # Exclude hidden files like .DS_Store
             not f.startswith('__') and    # Exclude system files like __pycache__
             'label' not in f and          # Exclude label files
             os.path.splitext(f)[1].lower() in valid_image_extensions]  # Only include valid image extensions

# Check for old_data directory and add those images too
old_data_dir = 'old_data'
old_data_path = os.path.join('app', old_data_dir)
if os.path.exists(old_data_path) and os.path.isdir(old_data_path):
    # Filter old data images the same way
    old_img_files = [f for f in os.listdir(old_data_path) 
                     if not f.startswith('.') and 
                     not f.startswith('__') and
                     'label' not in f and 
                     os.path.splitext(f)[1].lower() in valid_image_extensions]
    if old_img_files:
        st.sidebar.info(f"Found {len(old_img_files)} images in old_data directory")
        # Add a prefix to differentiate the old data files
        old_img_files = [f"[OLD] {img}" for img in old_img_files]
        img_files.extend(old_img_files)

# Get model paths
model_dir = '../models'
model_files = list(filter(lambda x: x.endswith('.pth') or x.endswith('.pt'), os.listdir(model_dir)))

# Main selection interface at the top
if not img_files:
    st.error("No valid image files found in the data directory. Please add images with extensions: .png, .jpg, .jpeg, .tif, .tiff, or .bmp")
    st.stop()

top_col1, top_col2 = st.columns(2)
with top_col1:
    # Image selection - show exact filenames without inference
    selected_img = st.selectbox(
        'Choose an image to analyze:',
        img_files,
        index=0 if len(img_files) > 0 else None,
        format_func=lambda x: x  # Display the exact filename
    )

with top_col2:
    # Check if any models were found
    if not model_files:
        st.error("No model files found in the models directory. Please add .pth or .pt model files.")
        st.stop()
    
    # Model selection
    selected_model = st.selectbox(
        'Choose a segmentation model:',
        model_files,
        index=0 if len(model_files) > 0 else None
    )

# Add segmentation parameters right after the header
# Check if we have high-res images that might need tiled processing
if selected_img:
    # Determine image path
    if selected_img.startswith("[OLD] "):
        # Handle old data images
        real_img_name = selected_img[6:]  # Remove the "[OLD] " prefix
        img_path = os.path.join('app', old_data_dir, real_img_name)
    else:
        img_path = os.path.join(img_dir, selected_img)
    
    # Check if file exists before trying to read it
    if os.path.exists(img_path):
        try:
            # Load image to check dimensions
            temp_img = cv2.imread(img_path)
            if temp_img is None:
                try:
                    import tifffile
                    temp_img = tifffile.imread(img_path)
                except:
                    pass
                    
            if temp_img is not None:
                # Check if high-resolution
                is_highres_image = ("4x" in selected_img or "10x" in selected_img or "x10" in selected_img or 
                                   max(temp_img.shape[0], temp_img.shape[1]) > 1000)
                
                if is_highres_image:
                    st.markdown("## Segmentation Parameters")
                    
                    # Special handling for high-resolution images
                    use_tiled_processing = st.checkbox(
                        "Use tiled processing (recommended for high-res images)",
                        value=True,
                        help="Process image in smaller tiles for better detection of small details"
                    )
                    
                    # Add controls for patch size and overlap percentage using a 3-column layout
                    if use_tiled_processing:
                        # Determine max dimension of image for sensible defaults
                        max_dim = max(temp_img.shape[0], temp_img.shape[1])
                        
                        # Calculate default patch size based on image dimensions
                        default_patch_size = min(512, max(256, max_dim // 8))
                        default_patch_size = default_patch_size + (default_patch_size % 32)  # Make divisible by 32
                        
                        # Create 3 columns for a more compact layout
                        col1, col2, col3 = st.columns(3)
                        
                        # Patch size slider in first column
                        with col1:
                            patch_size = st.slider(
                                "Patch Size (px):",
                                min_value=128,
                                max_value=1024,
                                value=default_patch_size,
                                step=32,
                                help="Size of patches to process. Larger patches need more memory but may detect larger panels better."
                            )
                        
                        # Overlap percentage slider in second column  
                        with col2:
                            overlap_percent = st.slider(
                                "Overlap (%):",
                                min_value=10,
                                max_value=50,
                                value=25,
                                step=5,
                                help="Percentage of overlap between patches. Higher overlap helps reduce boundary artifacts."
                            )
                            
                            # Calculate actual overlap in pixels
                            overlap = int(patch_size * overlap_percent / 100)
                        
                        # Scale factor slider in third column
                        with col3:
                            scale_factor = st.slider(
                                "Resolution (%):",
                                min_value=10,
                                max_value=100,
                                value=60 if max_dim > 4000 else 80,  # Default based on image size
                                step=10,
                                help="Percentage of original image resolution to use for processing."
                            ) / 100  # Convert percentage to decimal
                        
                        st.info(f"Using patches of {patch_size}x{patch_size} pixels with {overlap} pixels ({overlap_percent}%) overlap at {scale_factor*100:.0f}% of original resolution")
                        
                        # Add a visual representation of the patch settings
                        st.markdown("### Processing Overview")
                        # Move visualization outside of nested columns
                        col_viz1, col_viz2 = st.columns([1, 3])
                        with col_viz1:
                            # Create a simple visual of the patch size and overlap
                            fig_size = 200
                            patch_viz = np.zeros((fig_size, fig_size, 3), dtype=np.uint8)
                            
                            # Calculate visual parameters
                            patch_ratio = patch_size / (max_dim * scale_factor)
                            viz_patch_size = int(fig_size * patch_ratio)
                            viz_patch_size = max(40, min(viz_patch_size, 150))  # Keep size reasonable
                            
                            viz_overlap = int(viz_patch_size * overlap_percent / 100)
                            
                            # Draw a few overlapping patches
                            colors = [
                                (100, 180, 100),  # Light green
                                (100, 100, 180),  # Light blue
                                (180, 100, 180),  # Light purple
                                (180, 180, 100)   # Light yellow
                            ]
                            
                            for i in range(2):
                                for j in range(2):
                                    x = int(j * (viz_patch_size - viz_overlap) + 20)
                                    y = int(i * (viz_patch_size - viz_overlap) + 20)
                                    # Draw the patch
                                    color_idx = i * 2 + j
                                    cv2.rectangle(
                                        patch_viz, 
                                        (x, y), 
                                        (x + viz_patch_size, y + viz_patch_size), 
                                        colors[color_idx], 
                                        2
                                    )
                                    # Fill with semi-transparent color
                                    overlay = patch_viz.copy()
                                    cv2.rectangle(
                                        overlay, 
                                        (x, y), 
                                        (x + viz_patch_size, y + viz_patch_size), 
                                        colors[color_idx], 
                                        -1
                                    )
                                    # Apply transparency
                                    patch_viz = cv2.addWeighted(overlay, 0.2, patch_viz, 0.8, 0)
                                    
                                    # Add text label
                                    cv2.putText(
                                        patch_viz, 
                                        f"Patch {color_idx+1}", 
                                        (x + 5, y + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.4, 
                                        (255, 255, 255), 
                                        1
                                    )
                            
                            # Show the visualization
                            st.image(patch_viz, caption="Patch Overlap Visualization", use_container_width=True)
                        
                        with col_viz2:
                            # Add explanation text
                            st.markdown("""
                            The segmentation process will:
                            1. Scale the image to the selected resolution percentage
                            2. Divide the image into overlapping patches
                            3. Process each patch separately
                            4. Blend the overlapping areas for a seamless result
                            
                            **Processing time** depends on image size, patch size, and overlap settings.
                            Larger patches with high overlap provide better quality but take longer to process.
                            """)
                        
                        # Move expander outside of any column structure
                        st.markdown("### Parameter Effects")
                        with st.expander("How do these settings affect segmentation?"):
                            st.markdown("""
                            **Patch Size**: 
                            - **Larger patches** can detect larger solar panel arrays but require more memory
                            - **Smaller patches** work better for images with many small solar panels
                            - Recommended: 256-512px for most images, larger for high-resolution aerial imagery
                            
                            **Overlap Percentage**:
                            - **Higher overlap** (30-50%) reduces boundary artifacts but increases processing time
                            - **Lower overlap** (10-20%) is faster but may show visible seams between patches
                            - Recommended: 25% for balanced quality and speed
                            
                            **Processing Resolution**:
                            - **Higher percentages** (80-100%) preserve more detail but require more memory and time
                            - **Lower percentages** (30-60%) process faster but may miss smaller panels
                            - Recommended: 60-80% for most high-resolution images
                            """)
        except Exception as e:
            st.warning(f"Could not determine if image is high-resolution: {e}")

# Interface with left sidebar for controls and right area for results
col_setup, col_results = st.columns([1, 3])

with col_setup:
    st.subheader("Display Settings")
    
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
    
    # No longer needed since we show all visualizations
    # display_option = st.radio(
    #     "Display mode:",
    #     ["Binary mask", "Overlay", "Bounding boxes"]
    # )
    
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
    
    # Check if image file exists
    if not os.path.exists(img_path):
        st.error(f"Image file not found: {img_path}")
        st.stop()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    
    # Load image
    try:
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
                # Include additional info to help debug
                st.error(f"Please ensure the file is a valid image and not corrupted.")
                st.stop()
    except Exception as e:
        st.error(f"Error loading image {img_path}: {e}")
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
        # If we didn't already set use_tiled_processing in the top section, default to True
        if 'use_tiled_processing' not in locals():
            # Special handling for high-resolution images
            use_tiled_processing = True
        
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
                
                # Image type-specific settings - if not already set in the top parameters section
                if 'scale_factor' not in locals() or 'patch_size' not in locals() or 'overlap' not in locals():
                    if "x10" in selected_img or "10x" in selected_img:
                        # Default scaling suggestion for 10x images (4320x4340)
                        if 'scale_factor' not in locals():  # Only set if not already defined by user
                            scale_factor = 0.6  # Use 60% of original size
                    elif "4x" in selected_img:
                        # Default scaling suggestion for 4x images (3584x3584)
                        if 'scale_factor' not in locals():
                            scale_factor = 0.7  # Use 70% of original size
                    elif max_dim < 512:
                        # For old PV01/PV08 images (256x256)
                        if 'scale_factor' not in locals():
                            scale_factor = 1.0  # No scaling needed
                    elif max_dim < 1000:
                        # For small to medium images like the 432x434 standard image
                        if 'scale_factor' not in locals():
                            scale_factor = 1.0  # No scaling needed
                    else:
                        # Default for large images
                        if 'scale_factor' not in locals():
                            if max_dim > 4000:
                                scale_factor = 0.5  # Half size for very large
                            elif max_dim > 2000:
                                scale_factor = 0.7  # 70% for large
                            else:
                                scale_factor = 0.9  # 90% for medium-large
                    
                    # Set default tile size if not defined
                    if 'patch_size' not in locals():
                        patch_size = 512
                        
                    # Set default overlap if not defined
                    if 'overlap' not in locals():
                        overlap = int(patch_size * 0.25)  # 25% overlap
                
                # Use patch_size as tile_size to maintain naming consistency
                tile_size = patch_size
                
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
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error("Please try another model.")
            st.stop()
    
    # Results section
    with col_results:
        # Add a summary box of settings used at the top of results
        if do_tiled_processing:
            settings_col1, settings_col2, settings_col3 = st.columns(3)
            with settings_col1:
                st.metric("Patch Size", f"{tile_size}px")
            with settings_col2:
                st.metric("Overlap", f"{overlap_percent}%")
            with settings_col3:
                st.metric("Resolution", f"{int(scale_factor*100)}%")
            
            # Add a horizontal line to separate settings from results
            st.markdown("---")
        
        # Create a row with 3 columns for the main visualizations
        st.subheader("Analysis Results")
        input_col, overlay_col, mask_col = st.columns(3)
        
        with input_col:
            st.subheader("Input Image")
            st.image(image_for_display, caption=f"{selected_img}", use_container_width=True)
        
        with overlay_col:
            st.subheader("Segmentation Overlay")
            # Create overlay visualization
            display_mask = cv2.resize(pr_mask, display_size, interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(image_for_display)
            colored_mask[:,:,0] = display_mask * 255  # Red channel
            overlay = cv2.addWeighted(image_for_display, 0.7, colored_mask, 0.3, 0)
            
            # Include processing information in the caption
            if do_tiled_processing:
                overlay_caption = f"Overlay ({int(scale_factor*100)}% res)"
            else:
                overlay_caption = "Segmentation Overlay"
            st.image(overlay, caption=overlay_caption, use_container_width=True)
        
        with mask_col:
            st.subheader("Binary Mask")
            # Resize mask for display
            display_mask = cv2.resize(pr_mask, display_size, interpolation=cv2.INTER_NEAREST)
            
            # Include processing information in the caption
            if do_tiled_processing:
                mask_caption = f"Mask ({int(scale_factor*100)}% res)"
            else:
                mask_caption = "Binary Mask"
            st.image(display_mask, caption=mask_caption, use_container_width=True)
        
        # # Add a row showing just the binary mask under each image
        # st.subheader("Binary Mask Views")
        # bin_col1, bin_col2, bin_col3 = st.columns(3)
        
        # with bin_col1:
        #     # Create a black and white view of the binary mask (under input image)
        #     binary_mask_display = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        #     binary_mask_display[display_mask > 0.5] = [255, 255, 255]
        #     st.image(binary_mask_display, caption="Binary Mask (B&W)", use_container_width=True)
            
        # with bin_col2:
        #     # Create a red-colored binary mask view (under overlay)
        #     red_mask_display = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        #     red_mask_display[display_mask > 0.5] = [255, 0, 0]
        #     st.image(red_mask_display, caption="Binary Mask (Red)", use_container_width=True)
            
        # with bin_col3:
        #     # Create a heat map style view (under binary mask)
        #     heat_display = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        #     heat_display[display_mask > 0.5] = [0, 255, 0]
        #     heat_display[display_mask <= 0.5] = [50, 50, 50]  # Dark gray background
        #     st.image(heat_display, caption="Binary Mask (Highlight)", use_container_width=True)
        
        # Info and metrics rows
        metrics_col, info_col = st.columns(2)
        
        with info_col:
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
                    st.write(f"**Tile Size**: {tile_size}x{tile_size} pixels with {overlap}px overlap ({overlap_percent if 'overlap_percent' in locals() else int(overlap/tile_size*100)}%)")
                    st.write(f"**Total Tiles Processed**: {total_tiles}")
        
        with metrics_col:
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

        # END OF COLUMN LAYOUT - All column structures end here

        # Area calculations section - moved outside the column structure
        # st.markdown("---")
        # st.subheader("Area Calculations")
        # area = np.round(float(img_resolution) ** 2 * pr_mask.sum(), 4)
        # perc_area = np.round(100 * pr_mask.sum() / img_resolution ** 2, 2)
        # total_area = np.round(img_resolution ** 2 * float(img_resolution) ** 2, 2)
        
        # st.write(f"**Total Solar Panel Area**: {area} square meters")
        # st.write(f"**Coverage Percentage**: {perc_area}% of the image")
        # st.write(f"**Image Area**: {total_area} square meters") 