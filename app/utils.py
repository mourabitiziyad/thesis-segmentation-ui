import cv2
import numpy as np
import albumentations as A
import skimage.measure as km
import streamlit as st


@st.cache_resource(ttl=24 * 60 * 60)
def get_model(model, backbone, n_classes, activation):
    return model(backbone, classes=n_classes, activation=activation)


@st.cache_data
def get_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(256, 256),
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)


@st.cache_data
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


@st.cache_resource
def get_preprocessing(_preprocessing_fn):
    """Construct preprocessing transform

    Args:
        _preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=_preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


@st.cache_data
def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.round(np.sum(intersection) / np.sum(union), 2)
    return iou_score


@st.cache_data
def compute_pixel_acc(gt_mask, pred_mask):
    total = 256 ** 2
    errors = (np.abs((pred_mask == 0).sum() - (gt_mask == 0).sum()) + np.abs(
        (pred_mask != 0).sum() - (gt_mask != 0).sum()))
    accuracy = np.round(1 - (errors / total), 3)
    return accuracy


@st.cache_data
def compute_dice_coeff(gt_mask, pred_mask):
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))
    union = np.sum(np.logical_or(pred_mask, gt_mask)) + intersection
    dice_coef = np.round(2 * intersection / union, 3)
    return dice_coef


@st.cache_data
def compute_bboxes(image, pred_mask):
    """

    :param image:
    :param pred_mask:
    :return:
    """
    kernel = np.ones((5, 5), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    labeled = km.label(pred_mask)
    props = km.regionprops(labeled)
    bboxes = set([p.bbox for p in props])
    return bboxes


@st.cache_data
def draw_bbox(image, bboxes):
    for box in bboxes:
        image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
    return image


@st.cache_data
def return_coordinates(bboxes):
    num_boxes = len(bboxes)
    coordinate_dict = {}
    for i in range(num_boxes):
        box = list(bboxes)[i]
        top_left = (box[1], box[0])
        bottom_right = (box[3], box[2])
        top_right = (box[3], box[0])
        bottom_left = (box[1], box[2])
        coordinate_dict[i] = [top_left, top_right, bottom_left, bottom_right]
    return coordinate_dict


def get_model_info(model_name):
    info, ext = model_name.split('.')
    arch, *enc, epochs = info.split('_')

    enc = '_'.join(enc[:-1])
    
    # Handle the case where epochs starts with 'e' (e.g., 'e50')
    if epochs.startswith('e'):
        epochs = epochs[1:]  # Remove the 'e' prefix
        
    return arch, enc, int(epochs)


def calculate_segmentation_metrics(mask1, mask2):
    """Calculate comparison metrics between two segmentation masks.
    
    Args:
        mask1: First binary segmentation mask
        mask2: Second binary segmentation mask
        
    Returns:
        dict: Dictionary containing metrics (IoU, Dice coefficient, etc.)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (mask1.sum() + mask2.sum() + 1e-8)
    
    # Calculate area difference percentage
    area1 = mask1.sum()
    area2 = mask2.sum()
    area_diff_percent = abs(area1 - area2) / max(area1, area2) * 100
    
    return {
        "iou": iou,
        "dice": dice,
        "area_diff_percent": area_diff_percent
    }


def create_difference_visualization(mask1, mask2):
    """Create a visualization of differences between two segmentation masks.
    
    Args:
        mask1: First binary segmentation mask
        mask2: Second binary segmentation mask
        
    Returns:
        numpy.ndarray: RGB visualization where:
            - Green: Areas present in both masks
            - Red: Areas only in mask1
            - Blue: Areas only in mask2
    """
    h, w = mask1.shape
    diff_viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Green: Areas present in both masks
    diff_viz[np.logical_and(mask1, mask2)] = [0, 255, 0]
    # Red: Areas only in mask1
    diff_viz[np.logical_and(mask1, ~mask2)] = [255, 0, 0]
    # Blue: Areas only in mask2
    diff_viz[np.logical_and(~mask1, mask2)] = [0, 0, 255]
    
    return diff_viz


def align_images(image1, image2):
    """Align two images using ORB feature matching.
    
    Args:
        image1: Reference image
        image2: Image to be aligned
        
    Returns:
        numpy.ndarray: Aligned version of image2
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return image2  # Return original if no features found
    
    # Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take top matches
    good_matches = matches[:50]
    
    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return image2
    
    # Warp image
    aligned = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]))
    
    return aligned
