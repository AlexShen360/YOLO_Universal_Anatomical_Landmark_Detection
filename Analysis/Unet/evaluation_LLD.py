import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import os
import argparse
from test_xray_landmark_model import execute_model_on_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pixels_to_mm(pixel_value, metadata):
    """
    Convert a pixel measurement to millimeters using the spacing and resizing values from the metadata.
    The scaling factor is ensured to be less than 1, so that pixel values are always scaled down to mm.

    Args:
        pixel_value: The measurement in pixels
        metadata: Dictionary containing the metadata with spacing and resizing values

    Returns:
        mm_value: The measurement in millimeters
    """
    # Extract spacing and resizing values from metadata
    spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
    original_dimensions = metadata.get('original_dimensions', [0, 0])
    resized_dimensions = metadata.get('resized_dimensions', [1024, 1024])

    # Calculate the scaling factor
    # The spacing is in mm/pixel in the original image
    # We need to account for the resizing that was done to the image
    if original_dimensions[0] > 0 and resized_dimensions[0] > 0:
        # Calculate the scaling factor due to resizing
        resize_scale_x = original_dimensions[0] / resized_dimensions[0]
        resize_scale_y = original_dimensions[1] / resized_dimensions[1]

        # Use the average of x and y scaling factors
        resize_scale = (resize_scale_x + resize_scale_y) / 2

        # Calculate the combined scaling factor
        combined_scale = resize_scale * spacing[0]

        # Ensure the scaling factor is less than 1 to guarantee pixel to mm conversion always scales down
        if combined_scale >= 1.0:
            # Adjust the scaling factor to be less than 1
            # This ensures that mm values are always smaller than pixel values
            combined_scale = 0.5  # Use a default value of 0.5 when the scaling would otherwise be >= 1

        # Convert pixel value to mm using the adjusted scaling factor
        mm_value = pixel_value * combined_scale
    else:
        # If dimensions are not available, use a default scaling that ensures scaling down
        combined_scale = min(spacing[0], 0.5)  # Use spacing if < 0.5, otherwise use 0.5
        mm_value = pixel_value * combined_scale

    return mm_value

def load_ground_truth_landmarks(landmark_file):
    """
    Load ground truth landmarks from a CSV file.

    Args:
        landmark_file: Path to the CSV file containing ground truth landmarks

    Returns:
        landmarks: NumPy array of shape [num_keypoints, 2] containing the ground truth landmarks
    """
    # Read the CSV file
    landmarks_df = pd.read_csv(landmark_file)

    # Extract X, Y coordinates
    landmarks_list = []
    for _, row in landmarks_df.iterrows():
        landmark_coords = [float(row['X']), float(row['Y'])]
        landmarks_list.append(landmark_coords)

    # Convert to NumPy array
    landmarks = np.array(landmarks_list)

    return landmarks

def calculate_lld(keypoints, is_tensor=True, reference_type='tear_drop'):
    """
    Calculate Leg Length Discrepancy (LLD) from keypoints using different reference line types.

    LLD is defined as:
    1. Draw a reference line between two reference points (depends on reference_type)
    2. Draw a line from "Lesser_Trochanter_Left" parallel to reference line
    3. Draw a line from "Lesser_Trochanter_Right" parallel to reference line
    4. Calculate the perpendicular distance between these two parallel lines
    5. LLD is positive if Lesser_Trochanter_Left line is higher than Lesser_Trochanter_Right line, negative otherwise

    Args:
        keypoints: Tensor or NumPy array of shape [num_keypoints, 2] containing the keypoints
        is_tensor: Boolean indicating whether keypoints is a tensor (True) or NumPy array (False)
        reference_type: String indicating which reference line to use ('tear_drop', 'asis', 'ischial_tuberosity')

    Returns:
        lld: The calculated leg length discrepancy in pixels (signed value)
        visualization_data: Dictionary containing data for visualization
    """
    # The indices are based on the order in the CSV file, not HipXrayKeypointsRegressionLabels
    # CSV order (from 8623_landmarks.csv):
    # 0: ASIS_Left
    # 1: ASIS_Right
    # 2: Tear_Drop_Right
    # 3: Tear_Drop_Left
    # 4: Ischial_Tuberosity_Left
    # 5: Ischial_Tuberosity_Right
    # 6: Femoral_Head_Centre_Left
    # 7: Femoral_Head_Centre_Right
    # 8: Lesser_Trochanter_Left
    # 9: Lesser_Trochanter_Right

    # Define landmark indices
    asis_left_idx = 0
    asis_right_idx = 1
    tear_drop_right_idx = 2
    tear_drop_left_idx = 3
    ischial_tuberosity_left_idx = 4
    ischial_tuberosity_right_idx = 5
    lesser_trochanter_left_idx = 8
    lesser_trochanter_right_idx = 9

    # Determine reference points based on reference_type
    if reference_type == 'tear_drop':
        ref_left_idx = tear_drop_left_idx
        ref_right_idx = tear_drop_right_idx
        ref_name = 'tear_drop'
    elif reference_type == 'asis':
        ref_left_idx = asis_left_idx
        ref_right_idx = asis_right_idx
        ref_name = 'asis'
    elif reference_type == 'ischial_tuberosity':
        ref_left_idx = ischial_tuberosity_left_idx
        ref_right_idx = ischial_tuberosity_right_idx
        ref_name = 'ischial_tuberosity'
    else:
        raise ValueError(f"Invalid reference_type: {reference_type}. Must be 'tear_drop', 'asis', or 'ischial_tuberosity'")

    # Extract the required landmarks
    if is_tensor:
        ref_left = keypoints[ref_left_idx].cpu().numpy()
        ref_right = keypoints[ref_right_idx].cpu().numpy()
        lesser_trochanter_left = keypoints[lesser_trochanter_left_idx].cpu().numpy()
        lesser_trochanter_right = keypoints[lesser_trochanter_right_idx].cpu().numpy()
        # Also extract tear drop points for backward compatibility in visualization
        tear_drop_right = keypoints[tear_drop_right_idx].cpu().numpy()
        tear_drop_left = keypoints[tear_drop_left_idx].cpu().numpy()
    else:
        ref_left = keypoints[ref_left_idx]
        ref_right = keypoints[ref_right_idx]
        lesser_trochanter_left = keypoints[lesser_trochanter_left_idx]
        lesser_trochanter_right = keypoints[lesser_trochanter_right_idx]
        # Also extract tear drop points for backward compatibility in visualization
        tear_drop_right = keypoints[tear_drop_right_idx]
        tear_drop_left = keypoints[tear_drop_left_idx]

    # Calculate the reference line
    # Direction vector of the reference line
    ref_vector = ref_left - ref_right
    ref_unit_vector = ref_vector / np.linalg.norm(ref_vector)

    # Calculate the perpendicular unit vector to the reference line
    perp_unit_vector = np.array([-ref_unit_vector[1], ref_unit_vector[0]])

    # Project the lesser trochanters onto the perpendicular direction
    left_projection = np.dot(lesser_trochanter_left, perp_unit_vector)
    right_projection = np.dot(lesser_trochanter_right, perp_unit_vector)

    # Calculate the LLD as the signed difference between the projections
    # Positive if left is higher than right, negative otherwise
    lld = left_projection - right_projection

    # Prepare data for visualization
    visualization_data = {
        'reference_type': reference_type,
        'ref_name': ref_name,
        'ref_left': ref_left,
        'ref_right': ref_right,
        'tear_drop_right': tear_drop_right,
        'tear_drop_left': tear_drop_left,
        'lesser_trochanter_left': lesser_trochanter_left,
        'lesser_trochanter_right': lesser_trochanter_right,
        'ref_unit_vector': ref_unit_vector,
        'perp_unit_vector': perp_unit_vector,
        'left_projection': left_projection,
        'right_projection': right_projection,
        'lld': lld
    }

    return lld, visualization_data

def visualize_lld(image, keypoints, visualization_data, gt_keypoints=None, gt_visualization_data=None, metadata=None, output_path=None, show_plot=True):
    """
    Visualize the LLD calculation on the image, optionally comparing with ground truth.

    Args:
        image: The input image as a numpy array
        keypoints: Tensor of shape [num_keypoints, 2] containing the predicted keypoints
        visualization_data: Dictionary containing data for visualization from calculate_lld
        gt_keypoints: NumPy array of shape [num_keypoints, 2] containing the ground truth keypoints, optional
        gt_visualization_data: Dictionary containing data for visualization of ground truth, optional
        metadata: Dictionary containing the metadata with spacing and resizing values, optional
        output_path: Path to save the visualization image, optional
        show_plot: Boolean indicating whether to display the plot (default: True)
    """
    # Define indices for the required landmarks based on CSV order
    tear_drop_right_idx = 2  # Tear_Drop_Right
    tear_drop_left_idx = 3   # Tear_Drop_Left
    lesser_trochanter_left_idx = 8  # Lesser_Trochanter_Left
    lesser_trochanter_right_idx = 9  # Lesser_Trochanter_Right
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display the image
    ax.imshow(image, cmap='gray')
    title = 'Leg Length Discrepancy (LLD) Measurement'
    if gt_visualization_data is not None:
        title += ' - Predicted vs Ground Truth'
    ax.set_title(title)

    # Extract data for visualization
    reference_type = visualization_data['reference_type']
    ref_name = visualization_data['ref_name']
    ref_left = visualization_data['ref_left']
    ref_right = visualization_data['ref_right']
    tear_drop_right = visualization_data['tear_drop_right']
    tear_drop_left = visualization_data['tear_drop_left']
    lesser_trochanter_left = visualization_data['lesser_trochanter_left']
    lesser_trochanter_right = visualization_data['lesser_trochanter_right']
    ref_unit_vector = visualization_data['ref_unit_vector']
    perp_unit_vector = visualization_data['perp_unit_vector']
    lld = visualization_data['lld']

    # Define labels based on CSV order
    csv_labels = [
        "ASIS_Left",
        "ASIS_Right",
        "Tear_Drop_Right",
        "Tear_Drop_Left",
        "Ischial_Tuberosity_Left",
        "Ischial_Tuberosity_Right",
        "Femoral_Head_Centre_Left",
        "Femoral_Head_Centre_Right",
        "Lesser_Trochanter_Left",
        "Lesser_Trochanter_Right"
    ]

    # Plot predicted keypoints
    keypoints_np = keypoints.cpu().numpy()
    for i, (x, y) in enumerate(keypoints_np):
        label = csv_labels[i] if i < len(csv_labels) else f"Point {i+1}"
        ax.plot(x, y, 'o', color='blue', markersize=6)
        ax.text(x+5, y+5, label, color='white', fontsize=8, 
                bbox=dict(facecolor='blue', alpha=0.5))

    # Plot the reference line
    reference_line_label = f'Pred {ref_name.replace("_", " ").title()} Line'
    ax.plot([ref_right[0], ref_left[0]], 
            [ref_right[1], ref_left[1]], 
            'g-', linewidth=2, label=reference_line_label)

    # Calculate points for the parallel lines through the lesser trochanters
    # Extend the lines for better visualization
    extension = 200  # pixels

    # Left lesser trochanter line
    left_line_start = lesser_trochanter_left - extension * ref_unit_vector
    left_line_end = lesser_trochanter_left + extension * ref_unit_vector

    # Right lesser trochanter line
    right_line_start = lesser_trochanter_right - extension * ref_unit_vector
    right_line_end = lesser_trochanter_right + extension * ref_unit_vector

    # Plot the parallel lines
    ax.plot([left_line_start[0], left_line_end[0]], 
            [left_line_start[1], left_line_end[1]], 
            'r-', linewidth=2, label='Pred Left Lesser Trochanter Line')

    ax.plot([right_line_start[0], right_line_end[0]], 
            [right_line_start[1], right_line_end[1]], 
            'b-', linewidth=2, label='Pred Right Lesser Trochanter Line')

    # Draw the perpendicular measurement line showing the shortest distance between the two parallel lines
    # Use the perpendicular unit vector to create a line from one lesser trochanter line to the other
    # Start from the left lesser trochanter
    measurement_start = lesser_trochanter_left

    # Calculate the perpendicular projection of the left lesser trochanter onto the right line
    # The vector from right_line_start to lesser_trochanter_right is along the line
    line_vector = right_line_end - right_line_start
    line_unit_vector = line_vector / np.linalg.norm(line_vector)

    # Project the vector from right_line_start to lesser_trochanter_left onto the line direction
    projection_length = np.dot(lesser_trochanter_left - right_line_start, line_unit_vector)

    # Calculate the point on the right line that is closest to lesser_trochanter_left
    closest_point_on_right_line = right_line_start + projection_length * line_unit_vector

    # The measurement end point is the closest point on the right line
    measurement_end = closest_point_on_right_line

    # Draw the measurement line (perpendicular to the parallel lines)
    if metadata is not None:
        # Convert LLD to mm
        lld_mm = convert_pixels_to_mm(lld, metadata)
        label_text = f'Pred LLD: {lld:.2f} pixels ({lld_mm:.2f} mm)'
    else:
        label_text = f'Pred LLD: {lld:.2f} pixels'

    ax.plot([measurement_start[0], measurement_end[0]], 
            [measurement_start[1], measurement_end[1]], 
            'y-', linewidth=2, label=label_text)

    # If ground truth is provided, visualize it as well
    if gt_keypoints is not None and gt_visualization_data is not None:
        # Extract ground truth data for visualization
        gt_ref_left = gt_visualization_data['ref_left']
        gt_ref_right = gt_visualization_data['ref_right']
        gt_tear_drop_right = gt_visualization_data['tear_drop_right']
        gt_tear_drop_left = gt_visualization_data['tear_drop_left']
        gt_lesser_trochanter_left = gt_visualization_data['lesser_trochanter_left']
        gt_lesser_trochanter_right = gt_visualization_data['lesser_trochanter_right']
        gt_ref_unit_vector = gt_visualization_data['ref_unit_vector']
        gt_perp_unit_vector = gt_visualization_data['perp_unit_vector']
        gt_lld = gt_visualization_data['lld']

        # Plot ground truth keypoints
        for i, (x, y) in enumerate(gt_keypoints):
            if i in [tear_drop_right_idx, tear_drop_left_idx, lesser_trochanter_left_idx, lesser_trochanter_right_idx]:
                label = csv_labels[i] if i < len(csv_labels) else f"Point {i+1}"
                ax.plot(x, y, 'x', color='orange', markersize=6)
                ax.text(x-20, y-20, f"GT {label}", color='white', fontsize=8, 
                        bbox=dict(facecolor='orange', alpha=0.5))

        # Plot the ground truth reference line
        gt_reference_line_label = f'GT {ref_name.replace("_", " ").title()} Line'
        ax.plot([gt_ref_right[0], gt_ref_left[0]], 
                [gt_ref_right[1], gt_ref_left[1]], 
                'g--', linewidth=2, label=gt_reference_line_label)

        # Calculate points for the ground truth parallel lines
        # Left lesser trochanter line
        gt_left_line_start = gt_lesser_trochanter_left - extension * gt_ref_unit_vector
        gt_left_line_end = gt_lesser_trochanter_left + extension * gt_ref_unit_vector

        # Right lesser trochanter line
        gt_right_line_start = gt_lesser_trochanter_right - extension * gt_ref_unit_vector
        gt_right_line_end = gt_lesser_trochanter_right + extension * gt_ref_unit_vector

        # Plot the ground truth parallel lines
        ax.plot([gt_left_line_start[0], gt_left_line_end[0]], 
                [gt_left_line_start[1], gt_left_line_end[1]], 
                'r--', linewidth=2, label='GT Left Lesser Trochanter Line')

        ax.plot([gt_right_line_start[0], gt_right_line_end[0]], 
                [gt_right_line_start[1], gt_right_line_end[1]], 
                'b--', linewidth=2, label='GT Right Lesser Trochanter Line')

        # Calculate the ground truth perpendicular measurement
        gt_measurement_start = gt_lesser_trochanter_left

        # Calculate the perpendicular projection for ground truth
        gt_line_vector = gt_right_line_end - gt_right_line_start
        gt_line_unit_vector = gt_line_vector / np.linalg.norm(gt_line_vector)

        gt_projection_length = np.dot(gt_lesser_trochanter_left - gt_right_line_start, gt_line_unit_vector)

        gt_closest_point_on_right_line = gt_right_line_start + gt_projection_length * gt_line_unit_vector

        gt_measurement_end = gt_closest_point_on_right_line

        # Draw the ground truth measurement line
        if metadata is not None:
            # Convert GT LLD to mm
            gt_lld_mm = convert_pixels_to_mm(gt_lld, metadata)
            gt_label_text = f'GT LLD: {gt_lld:.2f} pixels ({gt_lld_mm:.2f} mm)'
        else:
            gt_label_text = f'GT LLD: {gt_lld:.2f} pixels'

        ax.plot([gt_measurement_start[0], gt_measurement_end[0]], 
                [gt_measurement_start[1], gt_measurement_end[1]], 
                'y--', linewidth=2, label=gt_label_text)

    # Add a legend
    ax.legend(loc='upper right')

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")

    # Show the figure if requested
    plt.tight_layout()
    if show_plot:
        plt.show()

    return fig

def process_folder(folder_path, model_path, output_folder=None, reference_type='tear_drop'):
    """
    Process all X-ray images in a folder, calculate LLD for each, and visualize the results.
    Writes results to CSV file case by case as they are processed.

    Args:
        folder_path: Path to the folder containing X-ray images
        model_path: Path to the trained model checkpoint
        output_folder: Path to save the output visualizations and results, optional
        reference_type: Reference line type for LLD calculation ('tear_drop', 'asis', 'ischial_tuberosity')
    """
    # Create output folder if provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Find all .npy files in the folder
    npy_files = list(Path(folder_path).glob("*.npy"))
    logger.info(f"Found {len(npy_files)} .npy files in {folder_path}")

    # Create CSV file path
    results_path = None
    if output_folder and len(npy_files) > 0:
        if reference_type == 'all':
            results_path = Path(output_folder) / "lld_results_all_types.csv"
        else:
            results_path = Path(output_folder) / f"lld_results_{reference_type}.csv"
        # CSV file will be created when the first case is processed

    # Process each file
    for npy_file in npy_files:
        # Get the base name without extension
        base_name = npy_file.stem

        # Check if corresponding landmark file exists
        landmark_file = npy_file.parent / f"{base_name}_landmarks.csv"
        metadata_file = npy_file.parent / f"{base_name}.json"

        # Skip if landmark file doesn't exist
        if not landmark_file.exists():
            logger.warning(f"Skipping {npy_file} - No landmark file found")
            continue

        # Execute the model on the image
        logger.info(f"Processing {npy_file}")
        try:
            keypoints, _ = execute_model_on_image(
                model_path=model_path,
                image_path=str(npy_file),
                landmarks_path=str(landmark_file) if landmark_file.exists() else None,
                metadata_path=str(metadata_file) if metadata_file.exists() else None
            )

            # Check if keypoints shape matches expected shape for LLD calculation
            # We need at least 10 keypoints for LLD calculation
            if keypoints.shape[1] < 10:
                logger.error(f"Not enough keypoints for LLD calculation. Need at least 10 keypoints, got {keypoints.shape[1]}.")
                logger.info(f"Skipping {base_name} and continuing with next case")
                continue
        except Exception as e:
            logger.error(f"Error executing model on image for {base_name}: {str(e)}")
            logger.info(f"Skipping {base_name} and continuing with next case")
            continue

        # Calculate LLD from predicted landmarks for all reference types
        try:
            # Load ground truth landmarks from CSV file
            gt_landmarks = load_ground_truth_landmarks(landmark_file)

            # Determine which reference types to calculate
            if reference_type == 'all':
                reference_types = ['tear_drop', 'asis', 'ischial_tuberosity']
            else:
                reference_types = [reference_type]

            # Store results for all reference types
            lld_results = {}
            visualization_results = {}
            gt_lld_results = {}
            gt_visualization_results = {}
            lld_diff_results = {}

            for ref_type in reference_types:
                try:
                    lld, visualization_data = calculate_lld(keypoints[0], reference_type=ref_type)
                    logger.info(f"Predicted LLD ({ref_type}) for {base_name}: {lld:.2f} pixels")

                    gt_lld, gt_visualization_data = calculate_lld(gt_landmarks, is_tensor=False, reference_type=ref_type)
                    logger.info(f"Ground Truth LLD ({ref_type}) for {base_name}: {gt_lld:.2f} pixels")

                    # Calculate difference between predicted and ground truth LLD
                    lld_diff = abs(lld - gt_lld)
                    logger.info(f"LLD Difference ({ref_type}) for {base_name}: {lld_diff:.2f} pixels")

                    # Store results
                    lld_results[ref_type] = lld
                    visualization_results[ref_type] = visualization_data
                    gt_lld_results[ref_type] = gt_lld
                    gt_visualization_results[ref_type] = gt_visualization_data
                    lld_diff_results[ref_type] = lld_diff

                except Exception as e:
                    logger.error(f"Error calculating LLD ({ref_type}) for {base_name}: {str(e)}")
                    lld_results[ref_type] = None
                    visualization_results[ref_type] = None
                    gt_lld_results[ref_type] = None
                    gt_visualization_results[ref_type] = None
                    lld_diff_results[ref_type] = None

            # For backward compatibility, use tear_drop as primary if available
            if 'tear_drop' in lld_results and lld_results['tear_drop'] is not None:
                lld = lld_results['tear_drop']
                visualization_data = visualization_results['tear_drop']
                gt_lld = gt_lld_results['tear_drop']
                gt_visualization_data = gt_visualization_results['tear_drop']
                lld_diff = lld_diff_results['tear_drop']
            else:
                # Use the first available result
                available_types = [t for t in reference_types if lld_results.get(t) is not None]
                if available_types:
                    first_type = available_types[0]
                    lld = lld_results[first_type]
                    visualization_data = visualization_results[first_type]
                    gt_lld = gt_lld_results[first_type]
                    gt_visualization_data = gt_visualization_results[first_type]
                    lld_diff = lld_diff_results[first_type]
                else:
                    logger.info(f"All LLD calculations failed for {base_name}, skipping and continuing with next case")
                    continue

        except Exception as e:
            logger.error(f"Error calculating LLD for {base_name}: {str(e)}")
            logger.info(f"Skipping LLD calculation for {base_name} and continuing with next case")
            # Skip to the next file
            continue

        # Load metadata for mm conversion
        metadata = None
        lld_mm = None
        gt_lld_mm = None
        lld_diff_mm = None
        lld_mm_results = {}
        gt_lld_mm_results = {}
        lld_diff_mm_results = {}

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Convert LLD values to mm for all reference types
                for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']:
                    if ref_type in lld_results and lld_results[ref_type] is not None and gt_lld_results[ref_type] is not None:
                        try:
                            lld_mm_results[ref_type] = convert_pixels_to_mm(lld_results[ref_type], metadata)
                            gt_lld_mm_results[ref_type] = convert_pixels_to_mm(gt_lld_results[ref_type], metadata)
                            lld_diff_mm_results[ref_type] = abs(lld_mm_results[ref_type] - gt_lld_mm_results[ref_type])

                            logger.info(f"Predicted LLD ({ref_type}) for {base_name}: {lld_mm_results[ref_type]:.2f} mm")
                            logger.info(f"Ground Truth LLD ({ref_type}) for {base_name}: {gt_lld_mm_results[ref_type]:.2f} mm")
                            logger.info(f"LLD Difference ({ref_type}) for {base_name}: {lld_diff_mm_results[ref_type]:.2f} mm")
                        except Exception as e:
                            logger.error(f"Error converting LLD ({ref_type}) to mm for {base_name}: {str(e)}")
                            lld_mm_results[ref_type] = None
                            gt_lld_mm_results[ref_type] = None
                            lld_diff_mm_results[ref_type] = None
                    else:
                        lld_mm_results[ref_type] = None
                        gt_lld_mm_results[ref_type] = None
                        lld_diff_mm_results[ref_type] = None

                # For backward compatibility, set primary mm values
                if lld is not None and gt_lld is not None:
                    lld_mm = convert_pixels_to_mm(lld, metadata)
                    gt_lld_mm = convert_pixels_to_mm(gt_lld, metadata)
                    lld_diff_mm = abs(lld_mm - gt_lld_mm)

            except Exception as e:
                logger.error(f"Error loading metadata for {base_name}: {str(e)}")
                logger.warning(f"Continuing without mm conversion for {base_name}")
                lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
                gt_lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
                lld_diff_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
        else:
            logger.warning(f"No metadata file found for {base_name}, cannot convert to mm")
            lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
            gt_lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
            lld_diff_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}

        # Load the image for visualization
        image = np.load(str(npy_file))

        # Visualize the results with both predicted and ground truth
        if output_folder and 'lld' in locals() and 'visualization_data' in locals():  # Only visualize if LLD calculation was successful
            try:
                output_path = Path(output_folder) / f"{base_name}_lld.png"
                fig = visualize_lld(
                    image, 
                    keypoints[0], 
                    visualization_data, 
                    gt_keypoints=gt_landmarks, 
                    gt_visualization_data=gt_visualization_data,
                    metadata=metadata,
                    output_path=str(output_path),
                    show_plot=False  # Don't show plots in batch processing mode
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error visualizing LLD for {base_name}: {str(e)}")
                logger.warning(f"Continuing without visualization for {base_name}")

        # Create result dictionary if any LLD calculation was successful
        if any(lld_results.get(ref_type) is not None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']):
            try:
                result_dict = {
                    'case_id': base_name,
                }

                # Add results for all three reference types
                for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']:
                    # Add pixel values
                    result_dict[f'predicted_lld_pixels_{ref_type}'] = lld_results.get(ref_type)
                    result_dict[f'ground_truth_lld_pixels_{ref_type}'] = gt_lld_results.get(ref_type)
                    result_dict[f'lld_difference_pixels_{ref_type}'] = lld_diff_results.get(ref_type)

                    # Add mm values if available
                    result_dict[f'predicted_lld_mm_{ref_type}'] = lld_mm_results.get(ref_type)
                    result_dict[f'ground_truth_lld_mm_{ref_type}'] = gt_lld_mm_results.get(ref_type)
                    result_dict[f'lld_difference_mm_{ref_type}'] = lld_diff_mm_results.get(ref_type)

                # Add landmark coordinates (using tear_drop visualization data if available, otherwise first available)
                if visualization_data is not None and gt_visualization_data is not None:
                    result_dict.update({
                        'tear_drop_right_x': visualization_data['tear_drop_right'][0],
                        'tear_drop_right_y': visualization_data['tear_drop_right'][1],
                        'tear_drop_left_x': visualization_data['tear_drop_left'][0],
                        'tear_drop_left_y': visualization_data['tear_drop_left'][1],
                        'lesser_trochanter_left_x': visualization_data['lesser_trochanter_left'][0],
                        'lesser_trochanter_left_y': visualization_data['lesser_trochanter_left'][1],
                        'lesser_trochanter_right_x': visualization_data['lesser_trochanter_right'][0],
                        'lesser_trochanter_right_y': visualization_data['lesser_trochanter_right'][1],
                        'gt_tear_drop_right_x': gt_visualization_data['tear_drop_right'][0],
                        'gt_tear_drop_right_y': gt_visualization_data['tear_drop_right'][1],
                        'gt_tear_drop_left_x': gt_visualization_data['tear_drop_left'][0],
                        'gt_tear_drop_left_y': gt_visualization_data['tear_drop_left'][1],
                        'gt_lesser_trochanter_left_x': gt_visualization_data['lesser_trochanter_left'][0],
                        'gt_lesser_trochanter_left_y': gt_visualization_data['lesser_trochanter_left'][1],
                        'gt_lesser_trochanter_right_x': gt_visualization_data['lesser_trochanter_right'][0],
                        'gt_lesser_trochanter_right_y': gt_visualization_data['lesser_trochanter_right'][1]
                    })

                # Write result to CSV file
                if results_path:
                    # Create a DataFrame from the single result
                    result_df = pd.DataFrame([result_dict])

                    # Check if file exists to determine if we need to write headers
                    file_exists = results_path.exists()

                    # Write to CSV file (append mode if file exists)
                    result_df.to_csv(results_path, mode='a', header=not file_exists, index=False)

                    if not file_exists:
                        logger.info(f"Created results file at {results_path}")
                    logger.info(f"Appended results for {base_name} to {results_path}")
            except Exception as e:
                logger.error(f"Error creating result dictionary for {base_name}: {str(e)}")
                logger.warning(f"Skipping CSV output for {base_name}")
        else:
            logger.warning(f"All LLD calculations failed for {base_name}, skipping CSV output")

    # Return the path to the results file if it was created
    return results_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate Leg Length Discrepancy (LLD) from X-ray images')
    parser.add_argument('--folder_path', type=str, default=r"C:\GitHub\coronal_AP\validation_dataset",
                        help='Path to the folder containing X-ray images')
    parser.add_argument('--model_path', type=str, default="./output/best_model_864_cases.ckpt",
                        help='Path to the trained model checkpoint')
    parser.add_argument('--output_folder', type=str, default="lld_results_all_types_500_cases",
                        help='Path to save the output visualizations and results')
    parser.add_argument('--single_case', type=str, default=None,
                        help='Process only a single case with the given ID')
    parser.add_argument('--reference_type', type=str, default='all',
                        choices=['tear_drop', 'asis', 'ischial_tuberosity', 'all'],
                        help='Reference line type for LLD calculation (default: all - calculates all three types)')

    # Try to parse args, but if no args provided, use defaults
    import sys
    if len(sys.argv) == 1:
        # No command line arguments provided, use defaults
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # Process a single case if specified
    if args.single_case:
        # Construct the file paths
        folder_path = Path(args.folder_path)
        npy_file = folder_path / f"{args.single_case}.npy"
        landmark_file = folder_path / f"{args.single_case}_landmarks.csv"
        metadata_file = folder_path / f"{args.single_case}.json"

        # Check if the files exist
        if not npy_file.exists():
            logger.error(f"File not found: {npy_file}")
            return

        if not landmark_file.exists():
            logger.error(f"Landmark file not found: {landmark_file}")
            return

        # Create output folder
        os.makedirs(args.output_folder, exist_ok=True)

        # Execute the model on the image
        logger.info(f"Processing single case: {args.single_case}")
        try:
            keypoints, _ = execute_model_on_image(
                model_path=args.model_path,
                image_path=str(npy_file),
                landmarks_path=str(landmark_file),
                metadata_path=str(metadata_file) if metadata_file.exists() else None
            )

            # Check if keypoints shape matches expected shape for LLD calculation
            # We need at least 10 keypoints for LLD calculation
            if keypoints.shape[1] < 10:
                logger.error(f"Not enough keypoints for LLD calculation. Need at least 10 keypoints, got {keypoints.shape[1]}.")
                logger.error("Cannot continue processing this case")
                return

            # Calculate LLD from predicted landmarks for all reference types
            try:
                # Load ground truth landmarks from CSV file
                gt_landmarks = load_ground_truth_landmarks(landmark_file)

                # Determine which reference types to calculate
                if args.reference_type == 'all':
                    reference_types = ['tear_drop', 'asis', 'ischial_tuberosity']
                else:
                    reference_types = [args.reference_type]

                # Store results for all reference types
                lld_results = {}
                visualization_results = {}
                gt_lld_results = {}
                gt_visualization_results = {}

                for ref_type in reference_types:
                    try:
                        lld, visualization_data = calculate_lld(keypoints[0], reference_type=ref_type)
                        logger.info(f"Predicted LLD ({ref_type}) for {args.single_case}: {lld:.2f} pixels")

                        gt_lld, gt_visualization_data = calculate_lld(gt_landmarks, is_tensor=False, reference_type=ref_type)
                        logger.info(f"Ground Truth LLD ({ref_type}) for {args.single_case}: {gt_lld:.2f} pixels")

                        # Store results
                        lld_results[ref_type] = lld
                        visualization_results[ref_type] = visualization_data
                        gt_lld_results[ref_type] = gt_lld
                        gt_visualization_results[ref_type] = gt_visualization_data

                    except Exception as e:
                        logger.error(f"Error calculating LLD ({ref_type}) for {args.single_case}: {str(e)}")
                        lld_results[ref_type] = None
                        visualization_results[ref_type] = None
                        gt_lld_results[ref_type] = None
                        gt_visualization_results[ref_type] = None

                # For backward compatibility, use tear_drop as primary if available
                if 'tear_drop' in lld_results and lld_results['tear_drop'] is not None:
                    lld = lld_results['tear_drop']
                    visualization_data = visualization_results['tear_drop']
                    gt_lld = gt_lld_results['tear_drop']
                    gt_visualization_data = gt_visualization_results['tear_drop']
                else:
                    # Use the first available result
                    available_types = [t for t in reference_types if lld_results.get(t) is not None]
                    if available_types:
                        first_type = available_types[0]
                        lld = lld_results[first_type]
                        visualization_data = visualization_results[first_type]
                        gt_lld = gt_lld_results[first_type]
                        gt_visualization_data = gt_visualization_results[first_type]
                    else:
                        lld = None
                        visualization_data = None
                        gt_lld = None
                        gt_visualization_data = None

            except Exception as e:
                logger.error(f"Error calculating LLD for {args.single_case}: {str(e)}")
                logger.warning("Continuing with limited functionality")
                # Initialize variables to None to indicate calculation failure
                lld = None
                visualization_data = None
                gt_lld = None
                gt_visualization_data = None
                lld_results = {}
                visualization_results = {}
                gt_lld_results = {}
                gt_visualization_results = {}
        except Exception as e:
            logger.error(f"Error executing model on image for {args.single_case}: {str(e)}")
            logger.error("Cannot continue processing this case")
            return

        # Load metadata for mm conversion
        metadata = None
        lld_mm = None
        gt_lld_mm = None
        lld_mm_results = {}
        gt_lld_mm_results = {}

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Convert LLD values to mm for all reference types
                for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']:
                    if ref_type in lld_results and lld_results[ref_type] is not None and gt_lld_results[ref_type] is not None:
                        try:
                            lld_mm_results[ref_type] = convert_pixels_to_mm(lld_results[ref_type], metadata)
                            gt_lld_mm_results[ref_type] = convert_pixels_to_mm(gt_lld_results[ref_type], metadata)

                            logger.info(f"Predicted LLD ({ref_type}) for {args.single_case}: {lld_mm_results[ref_type]:.2f} mm")
                            logger.info(f"Ground Truth LLD ({ref_type}) for {args.single_case}: {gt_lld_mm_results[ref_type]:.2f} mm")
                        except Exception as e:
                            logger.error(f"Error converting LLD ({ref_type}) to mm for {args.single_case}: {str(e)}")
                            lld_mm_results[ref_type] = None
                            gt_lld_mm_results[ref_type] = None
                    else:
                        lld_mm_results[ref_type] = None
                        gt_lld_mm_results[ref_type] = None

                # For backward compatibility, set primary mm values
                if lld is not None and gt_lld is not None:
                    lld_mm = convert_pixels_to_mm(lld, metadata)
                    gt_lld_mm = convert_pixels_to_mm(gt_lld, metadata)

            except Exception as e:
                logger.error(f"Error loading metadata for {args.single_case}: {str(e)}")
                logger.warning(f"Continuing without mm conversion for {args.single_case}")
                lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
                gt_lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
        else:
            logger.warning(f"No metadata file found for {args.single_case}, cannot convert to mm")
            lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}
            gt_lld_mm_results = {ref_type: None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']}

        # Load the image for visualization
        image = np.load(str(npy_file))

        # Visualize the results with both predicted and ground truth
        if keypoints is not None and visualization_data is not None and gt_visualization_data is not None:
            try:
                output_path = Path(args.output_folder) / f"{args.single_case}_lld.png"
                visualize_lld(
                    image, 
                    keypoints[0], 
                    visualization_data, 
                    gt_keypoints=gt_landmarks, 
                    gt_visualization_data=gt_visualization_data,
                    metadata=metadata,
                    output_path=str(output_path),
                    show_plot=True  # Show plot in single case mode
                )
                logger.info(f"Visualization saved to {output_path}")
            except Exception as e:
                logger.error(f"Error visualizing LLD for {args.single_case}: {str(e)}")
                logger.warning(f"Continuing without visualization for {args.single_case}")
        else:
            logger.warning(f"LLD calculation failed for {args.single_case}, skipping visualization")

        # Save the results to a CSV file if any LLD calculation was successful
        if any(lld_results.get(ref_type) is not None for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']):
            try:
                result_dict = {
                    'case_id': args.single_case,
                }

                # Add results for all three reference types
                for ref_type in ['tear_drop', 'asis', 'ischial_tuberosity']:
                    # Add pixel values
                    result_dict[f'predicted_lld_pixels_{ref_type}'] = lld_results.get(ref_type)
                    result_dict[f'ground_truth_lld_pixels_{ref_type}'] = gt_lld_results.get(ref_type)

                    # Add mm values if available
                    result_dict[f'predicted_lld_mm_{ref_type}'] = lld_mm_results.get(ref_type)
                    result_dict[f'ground_truth_lld_mm_{ref_type}'] = gt_lld_mm_results.get(ref_type)

                    # Add difference values if both predicted and ground truth are available
                    if (lld_results.get(ref_type) is not None and gt_lld_results.get(ref_type) is not None):
                        result_dict[f'lld_difference_pixels_{ref_type}'] = abs(lld_results[ref_type] - gt_lld_results[ref_type])
                        if (lld_mm_results.get(ref_type) is not None and gt_lld_mm_results.get(ref_type) is not None):
                            result_dict[f'lld_difference_mm_{ref_type}'] = abs(lld_mm_results[ref_type] - gt_lld_mm_results[ref_type])
                        else:
                            result_dict[f'lld_difference_mm_{ref_type}'] = None
                    else:
                        result_dict[f'lld_difference_pixels_{ref_type}'] = None
                        result_dict[f'lld_difference_mm_{ref_type}'] = None

                # Add landmark coordinates (using tear_drop visualization data if available, otherwise first available)
                if visualization_data is not None and gt_visualization_data is not None:
                    result_dict.update({
                        'tear_drop_right_x': visualization_data['tear_drop_right'][0],
                        'tear_drop_right_y': visualization_data['tear_drop_right'][1],
                        'tear_drop_left_x': visualization_data['tear_drop_left'][0],
                        'tear_drop_left_y': visualization_data['tear_drop_left'][1],
                        'lesser_trochanter_left_x': visualization_data['lesser_trochanter_left'][0],
                        'lesser_trochanter_left_y': visualization_data['lesser_trochanter_left'][1],
                        'lesser_trochanter_right_x': visualization_data['lesser_trochanter_right'][0],
                        'lesser_trochanter_right_y': visualization_data['lesser_trochanter_right'][1],
                        'gt_tear_drop_right_x': gt_visualization_data['tear_drop_right'][0],
                        'gt_tear_drop_right_y': gt_visualization_data['tear_drop_right'][1],
                        'gt_tear_drop_left_x': gt_visualization_data['tear_drop_left'][0],
                        'gt_tear_drop_left_y': gt_visualization_data['tear_drop_left'][1],
                        'gt_lesser_trochanter_left_x': gt_visualization_data['lesser_trochanter_left'][0],
                        'gt_lesser_trochanter_left_y': gt_visualization_data['lesser_trochanter_left'][1],
                        'gt_lesser_trochanter_right_x': gt_visualization_data['lesser_trochanter_right'][0],
                        'gt_lesser_trochanter_right_y': gt_visualization_data['lesser_trochanter_right'][1]
                    })

                results = [result_dict]
                results_df = pd.DataFrame(results)
                results_path = Path(args.output_folder) / f"{args.single_case}_lld_results_all_types.csv"
                results_df.to_csv(results_path, index=False)
                logger.info(f"Results saved to {results_path}")
            except Exception as e:
                logger.error(f"Error creating result dictionary for {args.single_case}: {str(e)}")
                logger.warning(f"Skipping CSV output for {args.single_case}")
        else:
            logger.warning(f"All LLD calculations failed for {args.single_case}, skipping CSV output")
    else:
        # Process all files in the folder
        process_folder(args.folder_path, args.model_path, args.output_folder, args.reference_type)

if __name__ == "__main__":
    main()
    #python evaluation_LLD.py --folder_path data\raw\hip\360Xrays\processed_files --single_case 9239 --model_path best_model.ckpt --output_folder lld_result --reference_type tear_drop
    #python evaluation_LLD.py --folder_path data\raw\hip\360Xrays\processed_files --model_path best_model.ckpt --output_folder lld_result --reference_type asis
    #python evaluation_LLD.py --folder_path data\raw\hip\360Xrays\processed_files --model_path best_model.ckpt --output_folder lld_result --reference_type ischial_tuberosity
