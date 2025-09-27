# -AutoVision-Stitcher-Automated-Image-Processing-Alignment  


Main Functionality
The repo is designed for automated image processing and alignment, providing a complete pipeline for preparing, transforming, analyzing, and merging images. Typical use cases include computer vision, scientific imaging, and any application requiring precise image stitching and enhancement.

Key Modules & Functions
Preprocessing

rename_images(source_folder): Renames image files based on patterns in their filenames.
rotate_images_left(input_folder): Rotates all images in a folder 90 degrees counterclockwise.
delete_black_images(folder_path, threshold): Removes images with excessive black pixels.
Image Arrangement & Splitting

arrange_images(input_folder, output_folder): Reorders images using a custom mapping.
split_images(input_folder, output_top_folder, output_bottom_folder): Splits each image into top and bottom halves.
spilt_quarters_to_3_parts(input_dir, output_dir): Divides images into three segments horizontally, resizing and labeling each part.
Geometric Transformations

apply_prespective_top(input_folder, output_folder): Applies perspective transformation to the top half of images.
apply_prespective_down(input_folder, output_folder): Similar to above, but for the bottom half.
Images_merging_after_prespective(input_folder, output_folder): Merges images after perspective correction.
Contrast & Enhancement

improve_contrast_images(input_folder, output_folder, max_images): Uses CLAHE for local contrast enhancement.
adaptive_histogram_equalization(input_folder, max_images): Applies adaptive histogram equalization to a batch of images.
Alignment & Stitching

calculate_overlap_x(image1, image2), calculate_overlap_y(image1, image2): Uses phase cross-correlation to determine optimal overlap for stitching.
merge_images_horizontally(img1, img2, dx), merge_rows_vertically(input_folder, output_path, data): Stitch images together horizontally or vertically based on calculated overlaps.
process_images_x_axis(input_folder, output_json_path), process_images_y_axis(input_folder, output_json_path): Analyze image pairs for stitching and save results.
Cropping

crop_images_x_axis(input_folder, output_folder): Crops images along the X axis based on filename patterns.
crop_images_y_axis(input_folder, output_folder, crop_num): Crops images along the Y axis.
Example Pipeline (as seen in main block)
Rename and rotate images.
Optionally delete black images.
Arrange and split images.
Apply perspective transformations.
Enhance contrast.
Analyze overlaps and stitch images.
Merge final rows/segments into one image.
