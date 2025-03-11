import os
import shutil
import cv2
import glob
import numpy as np
import json
from skimage.registration import phase_cross_correlation
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from collections import defaultdict

def get_row_and_index(filename):
    """ Extracts row number and image index from filename. """
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None  # Invalid format
    row = parts[1]
    index = int(parts[2].split(".")[0])  # Extract number before file extension
    return row, index

def is_black_image(img_path, threshold):
    """Checks if an image has more than the given percentage of black pixels."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {img_path}, skipping.")
        return False

    total_pixels = img.size
    black_pixels = np.sum(img == 0)
    black_ratio = black_pixels / total_pixels

    return black_ratio > threshold

def delete_black_images(folder_path, threshold=0.1):
    """
    Deletes black images from the start and end of rows but keeps black images in the middle.
    Images deleted in row 1 will also be deleted in other rows.
    
    :param folder_path: Path to the folder containing images.
    :param threshold: Percentage of black pixels to consider an image for deletion (default: 10%).
    """
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return

    # List all image files
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Organize images by row
    rows = {}
    for img_file in img_files:
        row, index = get_row_and_index(img_file)
        if row is not None:
            if row not in rows:
                rows[row] = []
            rows[row].append((index, img_file))

    # Ensure all rows have sorted images
    for row in rows:
        rows[row].sort()

    # Find images to delete from row 1
    delete_indices = set()
    if "01" in rows:
        row_images = rows["01"]
        num_images = len(row_images)

        # Identify black images at start
        for i in range(num_images):
            img_file = row_images[i][1]
            if is_black_image(os.path.join(folder_path, img_file), threshold):
                delete_indices.add(row_images[i][0])
            else:
                break  # Stop when a non-black image is found

        # Identify black images at end
        for i in range(num_images - 1, -1, -1):
            img_file = row_images[i][1]
            if is_black_image(os.path.join(folder_path, img_file), threshold):
                delete_indices.add(row_images[i][0])
            else:
                break  # Stop when a non-black image is found

    # Delete corresponding images in all rows
    for row, images in rows.items():
        for index, img_file in images:
            if index in delete_indices:
                img_path = os.path.join(folder_path, img_file)
                print(f"Deleting {img_file} from row {row}")
                os.remove(img_path)

    print("Processing complete.")

def rename_images(source_folder):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: The directory {source_folder} does not exist.")
        return

    # Process all image files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add other extensions if needed
            # Split the filename into parts by the underscore "_"
            parts = filename.split("_")
            
            # Ensure there are enough parts in the filename
            if len(parts) >= 10:  # We need at least 10 parts to access parts[6] and parts[7]
                # Extract the desired parts (parts[6] and parts[7] should give "01" and "17")
                part1 = parts[6]  # This should give "01"
                part2 = parts[7]  # This should give "17"
                
                # Print the extracted parts for verification
                print(f"Extracted parts: '{part1}', '{part2}'")
                
                # Create the new name by using the extracted parts (if required)
                new_name = f"15_{part1}_{part2}.png"  # You can format it differently if needed
                
                # Get the full path for the old and new file names
                source_path = os.path.join(source_folder, filename)
                new_path = os.path.join(source_folder, new_name)
                
                # Check if the new name already exists and append a counter if needed
                counter = 1
                while os.path.exists(new_path):
                    new_name = f"11_{part1}_{part2}_{counter}.png"  # Adjust the extension if needed
                    new_path = os.path.join(source_folder, new_name)
                    counter += 1
                
                # Rename the file
                os.rename(source_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")

    print("Renaming complete!")

def arrange_images(input_folder,output_folder):
   
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
       os.makedirs(output_folder, exist_ok=True)

    # Mapping of numbers for reversing (only the second section)
    rename_map = {
        "01": "12", "02": "11", "03": "10", "04": "09", "05": "08", "06": "07",
        "07": "06", "08": "05", "09": "04", "10": "03", "11": "02", "12": "01"
    }

    # Process files
    for filename in os.listdir(input_folder):
        # Check if the filename has the correct format
        parts = filename.split("_")
        if len(parts) >= 3 and parts[1] in rename_map:
            # Extract the first, second, and remaining parts
            first_part = parts[0]
            second_part = parts[1]
            remaining_parts = "_".join(parts[2:])

            # Rename the second part using the mapping
            new_second_part = rename_map[second_part]
            new_filename = f"{first_part}_{new_second_part}_{remaining_parts}"

            # Copy the file to the output folder with the new name
            shutil.copy(
                os.path.join(input_folder, filename),
                os.path.join(output_folder, new_filename)
            )

    print(f"Renaming and copying completed. Files saved to {output_folder}.")

def rotate_images_left(input_folder):
    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The directory {input_folder} does not exist.")
        return

    # Process all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add other extensions if needed
            # Get the full path for the image
            image_path = os.path.join(input_folder, filename)

            # Open the image
            with Image.open(image_path) as img:
                # Rotate the image 90 degrees counterclockwise (left)
                rotated_img = img.rotate(90, expand=True)

                # Save the rotated image back to the same path
                rotated_img.save(image_path)
                print(f"Rotated and saved: {filename}")

    print("Rotation complete!")

def split_images(input_folder, output_top_folder, output_bottom_folder):
    # Create output folders if they don't exist
    if not os.path.exists(output_top_folder):
        os.makedirs(output_top_folder, exist_ok=True)
    if not os.path.exists(output_bottom_folder):
        os.makedirs(output_bottom_folder, exist_ok=True)

    # Get all image files in the folder (e.g., .png, .jpg, .jpeg)
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.jpeg"))

    # Process each image in the folder
    for image_path in image_files:
        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Could not load the image from path: {image_path}")
            continue

        # Get the original image dimensions
        height, width = image.shape[:2]

        # Split the image into two parts by height
        top_part = image[:height // 2, :]  # Top half of the image
        bottom_part = image[height // 2:, :]  # Bottom half of the image

        # Get the base filename (without extension)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the top and bottom parts with new names
        top_output_path = os.path.join(output_top_folder, f"{base_filename}_1.png")  # Add _1 to the filename
        bottom_output_path = os.path.join(output_bottom_folder, f"{base_filename}_2.png")  # Add _2 to the filename

        cv2.imwrite(top_output_path, top_part)  # Save top part in 'Top' folder
        cv2.imwrite(bottom_output_path, bottom_part)  # Save bottom part in 'Bottom' folder

        # Display saved images paths
        print(f"Top part saved at: {top_output_path}")
        print(f"Bottom part saved at: {bottom_output_path}")

    print("Processing complete.")

def apply_prespective_top(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # Get all image files in the folder (e.g., .png, .jpg, .jpeg)
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg"))

    # Process each image in the folder
    for image_path in image_files:
        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Could not load the image from path: {image_path}")
            continue

        # Get the original image dimensions
        height, width = image.shape[:2]

        # Define four points on the original image (source points)
        # These are the corners of the original image (full image)
        src_points = np.float32([
            [0, 0],        # Top-left
            [width, 0],    # Top-right
            [0, height],   # Bottom-left
            [width, height] # Bottom-right
        ])

        # Top 
        dst_points = np.float32([
        [-15, 0],              # Top-left shifted right by 15 pixels
        [width + 15, 0],      # Top-right shifted left by 15 pixels
        [0, height],          # Bottom-left remains unchanged
        [width, height]       # Bottom-right remains unchanged
    ])

        # Get the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Perform the perspective transformation with stretching
        warped_image = cv2.warpPerspective(image, matrix, (width, height))

        # Get the base filename (without extension)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the transformed image in the 'Transformed' folder, keeping the original name
        output_path = os.path.join(output_folder, f"{base_filename}.jpg")  # Keep the original filename

        cv2.imwrite(output_path, warped_image)

        # Display the transformed image
        print(f"Transformed image saved at {output_path}")

    print("apply_prespective_top complete.")

def apply_prespective_down(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the folder (e.g., .png, .jpg, .jpeg)
    image_files = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg"))

    # Process each image in the folder
    for image_path in image_files:
        # Load the image
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Could not load the image from path: {image_path}")
            continue

        # Get the original image dimensions
        height, width = image.shape[:2]

        # Define four points on the original image (source points)
        # These are the corners of the original image (full image)
        src_points = np.float32([
            [0, 0],        # Top-left
            [width, 0],    # Top-right
            [0, height],   # Bottom-left
            [width, height] # Bottom-right
        ])

        # # Define the destination points for the transformed image
        #bottom
        # Move the bottom-left and bottom-right corners inward by 15 pixels
        dst_points = np.float32([
            [0, 0],               # Top-left moved inward (to the right by 15 pixels)
            [width , 0],       # Top-right moved inward (to the left by 15 pixels)
            [-15, height],          # Bottom-left moved outward (to the right by 15 pixels)
            [width +15, height]   # Bottom-right moved outward (to the left by 15 pixels)
        ])

        # Get the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Perform the perspective transformation with stretching
        warped_image = cv2.warpPerspective(image, matrix, (width, height))

        # Get the base filename (without extension)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the transformed image in the 'Transformed' folder, keeping the original name
        output_path = os.path.join(output_folder, f"{base_filename}.jpg")  # Keep the original filename

        cv2.imwrite(output_path, warped_image)

        # Display the transformed image
        print(f"Transformed image saved at {output_path}")

    print("apply_prespective_down complete.")

def Images_merging_after_prespective(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Function to merge images
    def merge_images(image_1_path, image_2_path):
        # Open the images
        img_1 = Image.open(image_1_path)
        img_2 = Image.open(image_2_path)

        # Get the size of the images
        width_1, height_1 = img_1.size
        width_2, height_2 = img_2.size

        # Ensure both images have the same width
        if width_1 != width_2:
            raise ValueError("Images must have the same width to be merged.")

        # Create a new blank image with combined height
        merged_img = Image.new('RGB', (width_1, height_1 + height_2))

        # Paste the images one on top of the other
        merged_img.paste(img_1, (0, 0))  # Top part
        merged_img.paste(img_2, (0, height_1))  # Bottom part

        return merged_img

    # Loop through the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('_1.jpg'):  # Adjust the extension if necessary
            base_name = filename.rsplit('_', 1)[0]  # Get the base name (e.g., 10_01_16)
            image_1_path = os.path.join(input_folder, filename)
            image_2_path = os.path.join(input_folder, base_name + '_2.jpg')  # Look for the corresponding _2 image

            # Check if both _1 and _2 images exist
            if os.path.exists(image_2_path):
                # Merge the images
                merged_img = merge_images(image_1_path, image_2_path)

                # Save the merged image in the output folder with the base name
                merged_img.save(os.path.join(output_folder, base_name + '.jpg'))

                print(f"Merged and saved: {base_name}.jpg")
            else:
                print(f"Skipping {base_name}, no corresponding _2 image found.")

    print("Images_merging_after_prespective completed.")

def calculate_overlap_x(image1, image2):
    """
    Calculate the overlap in pixels between two images along the X-axis.
    """
    image1 = np.rot90(image1)
    image2 = np.rot90(image2)
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = color.rgb2gray(image1)
    if image2.ndim == 3:
        image2 = color.rgb2gray(image2)

    # Compute the translation vector
    shift, _, _ = phase_cross_correlation(image1, image2)

    # Extract X-axis shift (dx)
    dx, _ = shift

    # Calculate X-axis overlap
    height = image1.shape[0]
    width = image1.shape[1]
    if dx >= 0:
        overlap_x_pixels = width - int(dx)
    else:
        overlap_x_pixels = width + int(dx)

    return (height-abs(dx)), overlap_x_pixels

    """
    Process images in the input folder, extracting 150 pixels from the top and bottom of each image as required,
    and save the cropped images to the output folder, depending on the row number in filenames.

    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the cropped images will be saved.
    """
  
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get sorted list of image filenames in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Group images based on row or camera number in filenames
    rows = {}
    for filename in image_files:
        # Assuming row number is embedded in the filename like xx_01_xx, xx_02_xx, etc.
        parts = filename.split('_')
        row_number = parts[1]  # Use the second part as the row number
        if row_number not in rows:
            rows[row_number] = []
        rows[row_number].append(filename)

    # Process each row
    for row_number, filenames in rows.items():
        print(f"Processing row {row_number}...")

        for i, filename in enumerate(filenames):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping file (could not read as an image): {filename}")
                continue

            height, width = image.shape[:2]

            if row_number == '01':  # First row: crop only from the bottom
                cropped_bottom = image[height -crop_num:, :]  # Extract the bottom 150 pixels
                output_filename_bottom = f"{os.path.splitext(filename)[0]}_bottom.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_bottom), cropped_bottom)
                print(f"Processed {filename} (First row, bottom crop_num pixels saved as {output_filename_bottom})")
            elif row_number == '12':  # Last row: crop only from the top
                cropped_top = image[:crop_num, :]  # Extract the top crop_num pixels
                output_filename_top = f"{os.path.splitext(filename)[0]}_top.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_top), cropped_top)
                print(f"Processed {filename} (Last row, top 150 pixels saved as {output_filename_top})")
            else:
                # Middle rows (2 to 11): crop from both top and bottom
                cropped_top = image[:crop_num, :]  # Extract the top crop_num pixels
                cropped_bottom = image[height - crop_num:, :]  # Extract the bottom 150 pixels

                # Save the top crop_num pixels
                output_filename_top = f"{os.path.splitext(filename)[0]}_top.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_top), cropped_top)
                print(f"Processed {filename} (Top 150 pixels saved as {output_filename_top})")

                # Save the bottom crop_num pixels
                output_filename_bottom = f"{os.path.splitext(filename)[0]}_bottom.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_bottom), cropped_bottom)
                print(f"Processed {filename} (Bottom crop_num pixels saved as {output_filename_bottom})")

    print("Processing complete. All results are saved.")

def calculate_overlap_y(image1, image2):
    """
    Calculate the overlap in pixels between two images along the X-axis.
    """
    image1 = np.rot90(image1)
    image2 = np.rot90(image2)
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = color.rgb2gray(image1)
    if image2.ndim == 3:
        image2 = color.rgb2gray(image2)

    # Compute the translation vector
    shift, _, _ = phase_cross_correlation(image1, image2)

    _, dy = shift

    # Calculate Y-axis overlap
    height = image1.shape[0]

    if dy >= 0:
        overlap_y_pixels = height - int(dy)
    else:
        overlap_y_pixels = height +  int(dy)

    width = image1.shape[1]
    return (width-abs(dy)), overlap_y_pixels

def crop_images_x_axis(input_folder, output_folder):
    """
    Process images in the input folder, extracting 150 pixels from the left and right of each image as required,
    and save the cropped images to the output folder, depending on row or camera pattern in filenames.

    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the cropped images will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get sorted list of image filenames in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Group images based on row or camera number in filenames
    rows = {}
    for filename in image_files:
        # Assuming row number is embedded in the filename like xx_01_xx, xx_02_xx, etc.
        parts = filename.split('_')
        row_number = parts[1]  # Use the second part as the row number
        if row_number not in rows:
            rows[row_number] = []
        rows[row_number].append(filename)

    # Process each row
    for row_number, filenames in rows.items():
        print(f"Processing row {row_number}...")

        for i, filename in enumerate(filenames):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping file (could not read as an image): {filename}")
                continue

            height, width = image.shape[:2]

            if i == 0:
                # First image of the row: extract 150 pixels from the right
                cropped_right = image[:, width - 150:]  # Extract the right 150 pixels
                output_filename_right = f"{os.path.splitext(filename)[0]}_2.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_right), cropped_right)
                print(f"Processed {filename} (First image, right 150 pixels saved as {output_filename_right})")
            elif i == len(filenames) - 1:
                # Last image of the row: extract 150 pixels from the left
                cropped_left = image[:, :150]  # Extract the left 150 pixels
                output_filename_left = f"{os.path.splitext(filename)[0]}_1.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_left), cropped_left)
                print(f"Processed {filename} (Last image, left 150 pixels saved as {output_filename_left})")
            else:
                # Middle images: process as required (no specific cropping defined)
                cropped_right = image[:, width - 150:]  # Extract the right 150 pixels
                cropped_left = image[:, :150]  # Extract the left 150 pixels

                # Save the right 150 pixels
                output_filename_right = f"{os.path.splitext(filename)[0]}_2.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_right), cropped_right)
                print(f"Processed {filename} (Right 150 pixels saved as {output_filename_right})")

                # Save the left 150 pixels
                output_filename_left = f"{os.path.splitext(filename)[0]}_1.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_left), cropped_left)
                print(f"Processed {filename} (Left 150 pixels saved as {output_filename_left})")

    print("Processing complete. All results are saved.")

def crop_images_y_axis(input_folder, output_folder,  crop_num):

    """
    Process images in the input folder, extracting 150 pixels from the top and bottom of each image as required,
    and save the cropped images to the output folder, depending on the row number in filenames.

    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the cropped images will be saved.
    """
  
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get sorted list of image filenames in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Group images based on row or camera number in filenames
    rows = {}
    for filename in image_files:
        # Assuming row number is embedded in the filename like xx_01_xx, xx_02_xx, etc.
        parts = filename.split('_')
        row_number = parts[1]  # Use the second part as the row number
        if row_number not in rows:
            rows[row_number] = []
        rows[row_number].append(filename)

    # Process each row
    for row_number, filenames in rows.items():
        print(f"Processing row {row_number}...")

        for i, filename in enumerate(filenames):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping file (could not read as an image): {filename}")
                continue

            height, width = image.shape[:2]

            if row_number == '01':  # First row: crop only from the bottom
                cropped_bottom = image[height -crop_num:, :]  # Extract the bottom 150 pixels
                output_filename_bottom = f"{os.path.splitext(filename)[0]}_bottom.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_bottom), cropped_bottom)
                print(f"Processed {filename} (First row, bottom crop_num pixels saved as {output_filename_bottom})")
            elif row_number == '12':  # Last row: crop only from the top
                cropped_top = image[:crop_num, :]  # Extract the top crop_num pixels
                output_filename_top = f"{os.path.splitext(filename)[0]}_top.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_top), cropped_top)
                print(f"Processed {filename} (Last row, top 150 pixels saved as {output_filename_top})")
            else:
                # Middle rows (2 to 11): crop from both top and bottom
                cropped_top = image[:crop_num, :]  # Extract the top crop_num pixels
                cropped_bottom = image[height - crop_num:, :]  # Extract the bottom 150 pixels

                # Save the top crop_num pixels
                output_filename_top = f"{os.path.splitext(filename)[0]}_top.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_top), cropped_top)
                print(f"Processed {filename} (Top 150 pixels saved as {output_filename_top})")

                # Save the bottom crop_num pixels
                output_filename_bottom = f"{os.path.splitext(filename)[0]}_bottom.png"
                cv2.imwrite(os.path.join(output_folder, output_filename_bottom), cropped_bottom)
                print(f"Processed {filename} (Bottom crop_num pixels saved as {output_filename_bottom})")

    print("Processing complete. All results are saved.")

def improve_contrast_images(input_folder, output_folder, max_images=1500):
    """
    Process all images in the input folder using CLAHE and save the results in the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        max_images (int): Maximum number of images to process (default: 1400).
    """
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder, exist_ok=True)
    # Get all image files from the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images.")

    # Limit the number of images to process
    image_files = image_files[:max_images]
    print(f"Processing up to {len(image_files)} images.")

    # CLAHE parameters
    clip_limit = 2.1  # Contrast limit
    tile_grid_size = (8, 8)  # Size of grid for adaptive histogram equalization

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    for image_filename in image_files:
        # Get the absolute image path
        full_image_path = os.path.join(input_folder, image_filename)
        print(f"Processing: {full_image_path}")

        # Load the current image in grayscale
        image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not load image: {image_filename}")
            continue

        # Apply Adaptive Histogram Equalization (AHE)
        equalized_image = clahe.apply(image)

        # Convert the processed image back to RGB
        equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

        # Save the processed image to the specified output path
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, equalized_image_rgb)
        print(f"Saved: {output_path}")

def filter_dynamic_range_by_column(matrix, bins):
    """
    Filter each column of the matrix to retain only values within the most populated range,
    then calculate the average of the filtered values for each column.

    Parameters:
        matrix (2D array-like): The input 2D matrix.
        bins (int): Number of bins to use for the histogram to find the range.

    Returns:
        list: A list containing the average of the filtered values for each column.
    """
    # Convert the input matrix to a NumPy array for easier manipulation
    matrix = np.array(matrix)
    num_cols = matrix.shape[1]
    average_data = []

    for col in range(num_cols):
        # Get the column data
        column_data = matrix[:, col]

        # Create a histogram for the column
        hist, bin_edges = np.histogram(column_data, bins=bins)

        # Find the bin with the maximum count
        max_bin_index = np.argmax(hist)

        # Determine the lower and upper bounds of the range
        lower_bound = bin_edges[max_bin_index]
        upper_bound = bin_edges[max_bin_index + 1]

        # Filter the column to include only numbers within the detected range
        filtered_column = [x for x in column_data if lower_bound <= x <= upper_bound]

        # Calculate the average of the filtered column (if there are any filtered values)
        if filtered_column:
            average = sum(filtered_column) / len(filtered_column)
            average_data.append(round(average))
        else:
            # If no values fall into the most populated range, append 0 or some default
            average_data.append(0)

    return average_data

def filter_dynamic_range_by_row(matrix, bins):
    """
    Filter each row of the matrix to retain only values within the most populated range.
    
    Parameters:
        matrix (2D array-like): The input 2D matrix.
        bins (int): Number of bins to use for the histogram to find the range.

    Returns:
        list: A filtered matrix where each row only contains values within the most populated range.
    """
    filtered_rows = []
    average_data=[]

    for row in matrix:
        # Create a histogram for the row
        hist, bin_edges = np.histogram(row, bins=bins)

        # Find the bin with the maximum count
        max_bin_index = np.argmax(hist)

        # Determine the lower and upper bounds of the range
        lower_bound = bin_edges[max_bin_index]
        upper_bound = bin_edges[max_bin_index + 1]

        # Filter the row to include only numbers within the detected range
        filtered_row = [x for x in row if lower_bound <= x <= upper_bound]
        filtered_rows.append(filtered_row)
       

    for row in  filtered_rows:
        sum=0
        i=0
        for num in row :
            sum=sum+num
            i=i+1
        average =sum/i
        average_data.append(round(average))    


    return average_data

def process_images_x_axis(input_folder, output_json_path):
    """
    Loop through images in the input folder, calculate overlap for valid pairs, and save results to a JSON file.
    """
    # Get all image files from the folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Group images by logs based on the filename convention
    grouped_images = {}
    for image_file in image_files:
        # Extract log number (e.g., "15_01_15.2" -> "15_01")
        log_number = "_".join(image_file.split('_')[:2])
        # print(log_number)
        # print(image_file)
        if log_number not in grouped_images:
            grouped_images[log_number] = []
        grouped_images[log_number].append(image_file)

    # Calculate overlaps

    results = {}
    dx_data=[]
    array_dx_data=[]
    for log_number, images in grouped_images.items():
        print(f"Processing {log_number}...")  # Debugging line
        results[log_number] = []
        pair_counter = 1 
        dx_data.clear() 
        # Process pairs within the log
        for i in range(0, len(images) - 1, 2):
            # Only calculate overlap for valid pairs (e.g., 15_01_15.2 with 15_01_16.1)

            image_base_1 = images[i].split('.')[0]
            image_end_1 = image_base_1.split('_')[-1]  # Gets the last part (e.g., '1' or '2')
            num_1 = int(image_end_1)

            image_base_2 = images[i+1].split('.')[0]
            image_end_2 = image_base_2.split('_')[-1]  # Gets the last part (e.g., '1' or '2')
            num_2 = int(image_end_2)

            if num_1 == 2 and num_2 == 1:
                # print(f"Valid pair: {images[i]} with {images[i + 1]}")  # Debugging line

                image1_path = os.path.join(input_folder, images[i])
                image2_path = os.path.join(input_folder, images[i + 1])

                # Load images and convert them to NumPy arrays
                image1 = np.array(Image.open(image1_path).convert('RGB'))
                image2 = np.array(Image.open(image2_path).convert('RGB'))

                # Calculate overlap
                dx, overlap_x_pixels = calculate_overlap_x(image1, image2)

                dx_data.append(dx)
                pair_counter += 1


        # Record the results
        results[log_number].append({
            "log_number": log_number,
            "shift_x": dx_data.copy()
            })

        array_dx_data.append(dx_data.copy())
                
   
    filtered_data = filter_dynamic_range_by_column(array_dx_data,10)

    # Display the result
    for i, row in enumerate(filtered_data):
        print(f"Row {i+1} filtered values: {row}")


    # Save results to JSON
    with open(output_json_path, 'w') as json_file:
         json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_json_path}")

    return filtered_data

def process_images_y_axis(input_folder, output_json_path):
    """
    Process images by creating separate arrays for top and bottom parts of each row,
    then calculate overlap between bottom images of one row and top images of the next row.

    Args:
        input_folder (str): Path to the folder containing images.
        output_json_path (str): Path to save the results in JSON format.
    """
    # Initialize dictionaries for top and bottom parts
    top_parts = {}
    bottom_parts = {}

    # Get all image files
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Group images into top and bottom parts
    for image_file in image_files:
        # Extract row number from the filename (e.g., "xx_01_001_top.png")
        parts = image_file.split('_')
        row_number = parts[1]  # Assuming the second part is the row number
        row_key = f"row_{row_number}"
        # Check if the image is top or bottom
        if "_top" in image_file:
            if row_key not in top_parts:
                top_parts[row_key] = []
            top_parts[row_key].append(image_file)
        elif "_bottom" in image_file:
            if row_key not in bottom_parts:
                bottom_parts[row_key] = []
            bottom_parts[row_key].append(image_file)

    # Sort images within each row to ensure proper indexing
    for row in top_parts:
        top_parts[row].sort()
    for row in bottom_parts:
        bottom_parts[row].sort()

    # Calculate overlaps between consecutive rows
    results = []
    # Initialize the list to store data
    dy_values = []
    array_dy_values = []
    # Combine keys from both dictionaries to ensure all rows are considered
    all_row_keys = set(top_parts.keys()).union(set(bottom_parts.keys()))
    sorted_rows = sorted(all_row_keys)  # Sort rows by row number
    for i in range(len(sorted_rows) - 1):
        current_row = sorted_rows[i]
        next_row = sorted_rows[i + 1]
        # Get bottom parts of the current row and top parts of the next row
        bottom_images = bottom_parts.get(current_row, [])
        top_images = top_parts.get(next_row, [])
        dy_values.clear() 
        # Pair images with matching indices and calculate overlap
        for j in range(min(len(bottom_images), len(top_images))):
            bottom_image_path = os.path.join(input_folder, bottom_images[j])
            top_image_path = os.path.join(input_folder, top_images[j])

            # Load images
            bottom_image = np.array(Image.open(bottom_image_path).convert('RGB'))
            top_image = np.array(Image.open(top_image_path).convert('RGB'))

            # Calculate overlap
            dy, overlap_y_pixels = calculate_overlap_y(bottom_image, top_image)
             # Append values to the lists
            dy_values.append(dy)
            # print(f"shift{j}_and_{j}",dy)
          

        results.append({
                "row_A": f"row_{i+1}",
                "row_B": f"row_{i+2}",
                "shift_Y": dy_values.copy()
            })
        array_dy_values.append(dy_values.copy())


    filtered_data = filter_dynamic_range_by_row(array_dy_values, 10)

    # Display the result
    for i, row in enumerate(filtered_data):
        print(f"Row {i+1} filtered values: {row}")

    # Save results to JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_json_path}")

    return filtered_data

def load_images_by_row(input_folder):
    """
    Dynamically load images and organize them into rows based on their filenames.
    Assumes filenames follow the format: 'xx_rownumber_imageindex.jpg'.
    """
    images_by_row = defaultdict(list)
    
    # Iterate through files in the input folder
    for file in sorted(os.listdir(input_folder)):
        if file.endswith((".jpg", ".png", ".jpeg")):
            # Extract row number from the filename
            parts = file.split('_')
            if len(parts) >= 3 and parts[1].isdigit():  # Ensure it's in the correct format
                row_number = int(parts[1])
                img_path = os.path.join(input_folder, file)
                images_by_row[row_number].append(img_path)
    
    return images_by_row

def merge_images_horizontally(img1, img2, dx):
    """
    Crop and merge two images horizontally based on the dx value.
    """
    crop_val = dx // 2
    img1_cropped = img1[:, :-crop_val]  # Crop right of img1
    img2_cropped = img2[:, crop_val:]  # Crop left of img2
    merged = np.hstack((img1_cropped, img2_cropped))  # Horizontally merge
    return merged

def process_all_rows(input_folder, output_folder, array_data):
    """
    Process all rows of images in the input folder, merge images in each row,
    and save the results to the output folder.
    """
    if not os.path.exists(output_folder):
         os.makedirs(output_folder, exist_ok=True)
    # Load images grouped by row
    images_by_row = load_images_by_row(input_folder)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through rows
    for row_num, image_paths in sorted(images_by_row.items()):
        if len(image_paths) < 2:
            print(f"Row {row_num} has less than 2 images, skipping...")
            continue
        
        # Load the first image
        merged_image = cv2.imread(image_paths[0])
        
        # Merge all images in the row
        for i in range(1, len(image_paths)):
            img2 = cv2.imread(image_paths[i])
            dx = array_data[i - 1]  # Get corresponding dx value
            merged_image = merge_images_horizontally(merged_image, img2, dx)
        
        # Save the merged row image
        output_path = os.path.join(output_folder, f"row_{row_num:02d}_merged.jpg")
        cv2.imwrite(output_path, merged_image)
        print(f"Row {row_num} merged and saved to {output_path}")

def merge_rows_vertically(input_folder, output_path, data):
    """
    Merge pre-merged row images vertically into one final image.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # Load merged row images
    merged_rows = []
    for file in sorted(os.listdir(input_folder)):
        if file.startswith("row_") and file.endswith((".jpg", ".png", ".jpeg")):
            merged_rows.append(os.path.join(input_folder, file))
    
    # Load the first merged row image
    final_image = cv2.imread(merged_rows[0])
    
    # Merge all rows
    for i in range(1, len(merged_rows)):
        row_img = cv2.imread(merged_rows[i])
        dy = data[i - 1]  # Get corresponding dy value
        crop_val = dy // 2
        
        # Crop final_image from the bottom and row_img from the top
        final_image_cropped = final_image[:-crop_val, :]
        row_img_cropped = row_img[crop_val:, :]
        
        # Merge rows vertically
        final_image = np.vstack((final_image_cropped, row_img_cropped))
    
    # Save the final merged image
    cv2.imwrite(output_path+"\final_merged_image.jpg", final_image)
    cv2.imwrite(os.path.join(output_path, "final_merged_image.jpg"), final_image)
    print(f"Final merged image saved to {output_path}")

def merge_cropped_images_quarters_from_folder(folder_path, crop_values, out_path):
    """
    Merge every three images vertically for four quarters, using the crop_values array to determine
    the cropping amount for each image pair in the sequence. The images are fetched from the specified folder
    and saved with dynamic filenames like row_01_merged, row_02_merged, etc.

    :param folder_path: Path to the folder containing the images.
    :param crop_values: Array of 8 crop values used for cropping the images.
    :param out_path: Output directory to save the merged images.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # Get all image files in the folder and sort them (optional, but useful if the files are not in order)
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Ensure we have the correct number of images (12 images expected)
    if len(img_files) != 12:
        raise ValueError("There should be exactly 12 image files in the folder.")

    # Process each quarter (group of 3 images)
    i=1
    for quarter in range(4):
        # Get the indices for the three images in this quarter
        start_index = quarter * 3
        end_index = start_index + 3
        imgs_for_quarter = img_files[start_index:end_index]

        # Get the crop values for the two pairs in this quarter
        crop_bottom1 = crop_values[quarter * 2]
        crop_top2 = crop_values[quarter * 2 + 1]

        # Load the images for the quarter
        img1 = cv2.imread(os.path.join(folder_path, imgs_for_quarter[0]))
        img2 = cv2.imread(os.path.join(folder_path, imgs_for_quarter[1]))
        img3 = cv2.imread(os.path.join(folder_path, imgs_for_quarter[2]))

        # Validate that images are loaded
        if img1 is None or img2 is None or img3 is None:
            raise ValueError(f"One or more images in quarter {quarter + 1} could not be loaded. Check the file paths.")

        # Crop the first image (remove pixels from the bottom)
        height1, width1, _ = img1.shape
        cropped_img1 = img1[:height1 - crop_bottom1, :, :]

        # Crop the second image (remove pixels from the top)
        height2, width2, _ = img2.shape
        cropped_img2 = img2[crop_top2:, :, :]

        # Resize the images to have the same width if necessary
        # Initialize new_width with the width of the first image
        new_width = width1
        if width1 != width2:
            new_width = min(width1, width2)
        
        # Resize the first and second images to match new_width
        cropped_img1 = cv2.resize(cropped_img1, (new_width, cropped_img1.shape[0]))
        cropped_img2 = cv2.resize(cropped_img2, (new_width, cropped_img2.shape[0]))

        # Merge the first two images
        merged_12 = np.vstack((cropped_img1, cropped_img2))

        # Crop the third image (remove pixels from the top based on the second crop value)
        height3, width3, _ = img3.shape
        cropped_img3 = img3[crop_top2:, :, :]

        # Resize the third image to match new_width
        cropped_img3 = cv2.resize(cropped_img3, (new_width, cropped_img3.shape[0]))

        # Final merge with the third image
        final_merged_image = np.vstack((merged_12, cropped_img3))

        # Convert the merged image to RGB
        final_merged_image_rgb = cv2.cvtColor(final_merged_image, cv2.COLOR_BGR2RGB)

        # Generate dynamic filename for the output image
        output_file = os.path.join(out_path, f"quarter_{i}.jpg")
        i+=1
        cv2.imwrite(output_file, final_merged_image_rgb)

    print("Image merging completed.")

def spilt_quarters_to_3_parts(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define processing parameters
    height_divisions = 1  # Number of vertical divisions
    width_divisions = 3  # Number of horizontal divisions
    target_width = 8000  # Target width of each resized segment
    target_height = 1020  # Target height of each resized segment

    def add_segment_label(segment, label_text, transparency=0.5):
        overlay = segment.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 5
        label_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black rectangle behind the text

        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = 20
        text_y = text_size[1] + 20

        cv2.rectangle(
            overlay, (10, 10), (10 + text_size[0] + 20, 10 + text_size[1] + 20), bg_color, -1
        )
        cv2.putText(
            overlay, label_text, (text_x, text_y), font, font_scale, label_color, font_thickness
        )
        cv2.addWeighted(overlay, transparency, segment, 1 - transparency, 0, segment)
        return segment

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not (file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
            print(f"Skipping non-image file: {file_name}")
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not load image: {file_name}")
            continue

        image_height, image_width, _ = image.shape
        segment_height = image_height // height_divisions
        segment_width = image_width // width_divisions
        base_name = os.path.splitext(file_name)[0]

        for i in range(height_divisions):
            for j in range(width_divisions):
                y_start, y_end = i * segment_height, (i + 1) * segment_height
                x_start, x_end = j * segment_width, (j + 1) * segment_width
                segment = image[y_start:y_end, x_start:x_end]
                resized_segment = cv2.resize(segment, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                label_text = f"{base_name}_Segment {i + 1},{j + 1}"
                labeled_segment = add_segment_label(resized_segment, label_text)

                output_filename = f"{base_name}segment{i + 1}_{j + 1}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, labeled_segment)
                print(f"Saved: {output_path}")

    print("All images have been processed and their segments saved.")

def adaptive_histogram_equalization(input_folder, max_images=1500):
    """
    Processes images in the input folder using CLAHE and saves them back to the same folder.
    
    :param input_folder: Path to the input folder containing images.
    :param max_images: Maximum number of images to process.
    """
    # Get all image files from the folder (adjust extensions if needed)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images.")
    
    # Limit to max_images
    image_files = image_files[:max_images]
    print(f"Processing up to {len(image_files)} images.")
    
    # CLAHE parameters
    clip_limit = 1  # Contrast limit
    tile_grid_size = (100, 100)  # Size of grid for adaptive histogram equalization
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    for image_path in image_files:
        full_image_path = os.path.join(input_folder, image_path)
        print(f"Processing: {full_image_path}")
        
        # Load the current image in grayscale
        image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Apply Adaptive Histogram Equalization (AHE)
        equalized_image = clahe.apply(image)
        
        # Convert the processed image back to RGB
        equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)
        
        # Save the processed image to the same path
        cv2.imwrite(full_image_path, equalized_image_rgb)
        print(f"Saved: {full_image_path}")


# Example usage in main
if __name__ == "__main__":
        source_folder = r"D:\Codes\full_test\6-2-2025\image_test"
        arranged_images_path = os.path.join(os.path.dirname(source_folder), "arranged_images")
        output_top_folder = os.path.join(os.path.dirname(source_folder), "Top")
        output_bottom_folder =os.path.join(os.path.dirname(source_folder), "Bottom")
        Transformed_path = os.path.join(os.path.dirname(source_folder), "Transformed")
        output_images = os.path.join(os.path.dirname(source_folder), "output_images")  
        output_path_x =os.path.join(os.path.dirname(source_folder), "x_analysis")
        output_json_path_x = os.path.join(output_path_x, "overlap_results.json")
        output_folder = os.path.join(os.path.dirname(source_folder), "final_rows") 
        output_path_y =os.path.join(os.path.dirname(source_folder), "y_analysis")
        output_json_path_y = os.path.join(output_path_y, "overlap_results.json")
        final_image = os.path.join(os.path.dirname(source_folder), "final_merged_image")
        quarters_path = os.path.join(os.path.dirname(source_folder), " quarters_images")
        spilt_quarters_path = os.path.join(quarters_path, "spilt_quarters")
        input_path = output_images


        rename_images(source_folder)
        rotate_images_left(source_folder)
        delete_black_images(source_folder,0.1)       
        
        # arrange_images(source_folder,arranged_images_path)
        # split_images(arranged_images_path, output_top_folder, output_bottom_folder)
        # apply_prespective_top(output_top_folder, Transformed_path)
        # apply_prespective_down(output_bottom_folder, Transformed_path)
        # Images_merging_after_prespective(Transformed_path,output_images)
        
        # # # #analysis for row (x) and merging the images to rows 
        # crop_images_x_axis(input_path, output_path_x)
        # improve_contrast_images(output_path_x, output_path_x, max_images=1500)
        # dx_data= process_images_x_axis(output_path_x, output_json_path_x)
    
    
        # # # merging images to rows
        # process_all_rows(input_path, output_folder, dx_data)
        
        # ## analysis for col (y) and merging the row the one image  
        # crop_images_y_axis(input_path, output_path_y,300)
        # improve_contrast_images(output_path_y,  output_path_y, max_images=1500)
        # dy_data=process_images_y_axis(output_path_y, output_json_path_y)
    
        # # merging to one image
        # merge_rows_vertically(output_folder, final_image, dy_data)

        # crop_values = [0, 150,         
        #                20, 150,
        #                30, 145, 
        #                0, 210]  # Example crop values, one for each pair of images
    
        # merge_cropped_images_quarters_from_folder(output_folder, crop_values, quarters_path)
        # spilt_quarters_to_3_parts(quarters_path,spilt_quarters_path)
        # adaptive_histogram_equalization(spilt_quarters_path,1500)
        # print("All processes completed successfully.")

        
        























# def delete_black_images_old(folder_path, threshold):
#     """
#     Deletes images that have more than a given percentage of black pixels (default: 10%).
#     If black pixels exceed 10% on the left or right side, the image is also deleted.

#     :param folder_path: Path to the folder containing images.
#     :param threshold: Percentage of black pixels to consider an image for deletion (default: 10%).
#     """
#     if not os.path.exists(folder_path):
#         print("Folder does not exist.")
#         return

#     # List all image files
#     img_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

#     for img_file in img_files:
#         img_path = os.path.join(folder_path, img_file)
#         img = cv2.imread(img_path)

#         if img is None:
#             print(f"Could not read {img_file}, skipping.")
#             continue

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Get total pixels in image
#         total_pixels = gray.shape[0] * gray.shape[1]

#         # Count black pixels (pixel value = 0)
#         black_pixels = np.sum(gray == 0)

#         # Calculate black pixel percentage
#         black_ratio = black_pixels / total_pixels

#         # If total black pixels exceed threshold, delete the image
#         if black_ratio > threshold:
#             print(f"Deleting {img_file} - {black_ratio:.2%} black")
#             os.remove(img_path)
#             continue  # No need to check left/right if already deleted

#         # Check left and right side separately (split the image vertically)
#         width = gray.shape[1]
#         left_side = gray[:, :width // 2]
#         right_side = gray[:, width // 2:]

#         left_black_ratio = np.sum(left_side == 0) / (total_pixels / 2)
#         right_black_ratio = np.sum(right_side == 0) / (total_pixels / 2)

#         # If either side has too much black, delete the image
#         if left_black_ratio > threshold or right_black_ratio > threshold:
#             print(f"Deleting {img_file} - Black concentration on left ({left_black_ratio:.2%}) or right ({right_black_ratio:.2%})")
#             os.remove(img_path)

#     print("Processing complete.")
