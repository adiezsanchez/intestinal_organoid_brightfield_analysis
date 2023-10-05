import os
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tifffile
from skimage import measure
from skimage.color import rgb2gray
from apoc import ObjectSegmenter, ObjectClassifier
import pyclesperanto_prototype as cle  # version 0.24.1
import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.6
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Initialize GPU-acceleration if available
device = cle.select_device("TX")
print("Used GPU: ", device)


def read_images(directory_path):
    """Reads all the images in the input path and organizes them according to the well_id"""
    # Define the directory containing your files
    directory_path = Path(directory_path)

    # Initialize a dictionary to store the grouped (per well) files
    images_per_well = {}

    # Iterate through the files in the directory
    for file_path in directory_path.glob("*.TIF"):
        # Get the filename without the extension
        filename = file_path.stem
        # Remove unwanted files (Plate_R files)
        if "Plate_R" in filename:
            pass
        # Remove maximum projections
        elif "_z" not in filename:
            pass

        else:
            # Extract the last part of the filename (e.g., A06f00d0)
            last_part = filename.split("_")[-1]

            # Get the first three letters to create the group name (well_id)
            well_id = last_part[:3]

            # Check if the well_id exists in the dictionary, if not, create a new list
            if well_id not in images_per_well:
                images_per_well[well_id] = []

            # Append the file to the corresponding group
            images_per_well[well_id].append(str(file_path))

    return images_per_well


def find_focus(images_per_well):
    """Processes all the images and extract the number of organoids in focus from each image"""

    nr_infocus_organoids = {}

    for well_id in tqdm(images_per_well):
        imgs_to_process = images_per_well[well_id]

        for input_img in imgs_to_process:
            # Check if the well_id exists in the dictionary, if not, create a new list
            if well_id not in nr_infocus_organoids:
                nr_infocus_organoids[well_id] = []

            # Load one RGB image and transform it into grayscale (if needed) for APOC
            rgb_img = tifffile.imread(input_img, is_ome=False)
            if len(rgb_img.shape) < 3:
                img = rgb_img
            elif rgb_img.shape[2] == 3:
                img = rgb2gray(rgb_img)
            else:
                print(
                    "Modify the loader to accommodate different file formats",
                    rgb_img.shape,
                )

            # Apply object segmenter from APOC
            try:
                segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
                result = segmenter.predict(image=img)
            except IndexError:
                segmenter = ObjectSegmenter(
                    opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
                )
                result = segmenter.predict(image=img)

            # Closing some holes in the organoid labels
            closed_labels = cle.closing_labels(result, None, radius=4.0)

            # Exclude small labels, cutout in pixel area seems to be below 1000px
            exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
            exclude_small = np.array(
                exclude_small, dtype=np.int32
            )  # Change dtype of closed labels to feed array into nsbatwm.split

            # Splitting organoids into a binary mask
            split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

            # Connected component (cc) labeling
            cc_split_organoids = nsbatwm.connected_component_labeling(
                split_organoids, False
            )

            # Apply object classifier from APOC
            try:
                classifier = ObjectClassifier(opencl_filename="./ObjectClassifier.cl")
                result = classifier.predict(labels=cc_split_organoids, image=img)
            except AttributeError:
                classifier = ObjectClassifier(
                    opencl_filename="./pretrained_APOC/ObjectClassifier.cl"
                )
                result = classifier.predict(labels=cc_split_organoids, image=img)

            # Convert the resulting .cle image into a np.array to count objects within each class
            image_array = np.array(result, dtype=np.int8)

            # Create masks for each class
            background_mask = image_array == 0
            out_of_focus_mask = image_array == 1
            in_focus_mask = image_array == 2

            # Label connected components in each mask
            background_labels = measure.label(background_mask, connectivity=2)
            out_of_focus_labels = measure.label(out_of_focus_mask, connectivity=2)
            in_focus_labels = measure.label(in_focus_mask, connectivity=2)

            # Calculate the number of objects in each class
            num_background_objects = np.max(background_labels)
            num_out_of_focus_objects = np.max(out_of_focus_labels)
            num_in_focus_objects = np.max(in_focus_labels)

            # print(f"Number of Background Objects: {num_background_objects}")
            # print(f"Number of Out-of-Focus Objects: {num_out_of_focus_objects}")
            # print(f"Number of In-Focus Objects: {num_in_focus_objects}")
            try:
                in_focus_percentage = (
                    num_in_focus_objects
                    / (num_in_focus_objects + num_out_of_focus_objects)
                ) * 100
                # print(f"Percentage of In-Focus Objects: {in_focus_percentage}")
            except ZeroDivisionError:
                in_focus_percentage = 0

            nr_infocus_organoids[well_id].append(in_focus_percentage)

    return nr_infocus_organoids


def find_highest_infocus(nr_infocus_organoids):
    """Finds the z-stack with the most organoids in-focus in the dictionary containing said data"""
    max_index_dict = (
        {}
    )  # Create a new dictionary to store the maximum index for each key

    for key, values in nr_infocus_organoids.items():
        max_value = max(values)  # Find the maximum value within the list
        max_index = values.index(max_value)  # Get the index of the maximum value
        max_index_dict[key] = max_index  # Store the index in the new dictionary

    return max_index_dict


def store_imgs(
    images_per_well, max_index_dict, output_dir="./output/in_focus_organoids"
):
    """Stores images in focus"""
    # Create a directory to store the tif files if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through keys in max_index_dict
    for key, index in max_index_dict.items():
        if key in images_per_well:
            file_paths = images_per_well[key]
            if 0 <= index < len(file_paths):
                file_path = file_paths[index]

                # Extract the file name (without the path)
                file_name = os.path.basename(file_path)

                # Construct the output file path
                output_path = os.path.join(output_dir, f"{key}.tif")

                # Copy the file using shutil.copy
                shutil.copy(file_path, output_path)

                # print(f"Saved {key}.tif to {output_path}")
            # else:
            # print(f"Invalid index for key {key}: {index}")
        # else:
        # print(f"Key {key} not found in images_per_well dictionary")


def plot_plate(
    resolution, output_path, img_folder_path, show_fig=True, colormap="gray"
):
    """Plot images in a grid-like fashion according to the well_id position in the plate"""

    # Initialize a dictionary to store images by rows (letters)
    image_dict = {}

    # Iterate through the image files in the folder
    for filename in os.listdir(img_folder_path):
        if filename.endswith(".tif"):
            # Extract the first letter and the number from the filename
            first_letter = filename[0]
            number = int(filename[1:3])

            # Create a dictionary entry for the first letter if it doesn't exist
            if first_letter not in image_dict:
                image_dict[first_letter] = {}

            # Create a dictionary entry for the number if it doesn't exist
            if number not in image_dict[first_letter]:
                image_dict[first_letter][number] = []

            # Append the image filename to the corresponding number
            image_dict[first_letter][number].append(filename)

    # Sort the dictionary by keys (letters) and nested dictionary by keys (numbers)
    sorted_image_dict = {
        letter: dict(sorted(images.items()))
        for letter, images in sorted(image_dict.items())
    }

    # Calculate the number of rows based on the number of letters
    num_rows = len(sorted_image_dict)

    # Calculate the number of columns based on the maximum number
    num_cols = max(max(images.keys()) for images in sorted_image_dict.values())

    # Calculate the figsize based on the number of columns and rows
    fig_width = num_cols * 3  # Adjust the multiplier as needed
    fig_height = num_rows * 2.5  # Adjust the multiplier as needed

    # Create a subplot for each image, using None for empty subplots
    if num_rows == 1:
        fig, axes = plt.subplots(
            1, num_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
    elif num_cols == 1:
        fig, axes = plt.subplots(
            num_rows, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
    else:
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey=True,
        )

    for i, (letter, images) in enumerate(sorted_image_dict.items()):
        for j, (number, filenames) in tqdm(enumerate(images.items())):
            if filenames:
                image_filename = filenames[0]  # Use the first filename in the list
                image_path = os.path.join(img_folder_path, image_filename)
                image = tifffile.imread(image_path, is_ome=False)

                if num_rows == 1:
                    axes[j].imshow(image, cmap=colormap)
                    axes[j].set_title(f"{letter}{number:02d}")
                    axes[j].axis("off")
                elif num_cols == 1:
                    axes[i].imshow(image, cmap=colormap)
                    axes[i].set_title(f"{letter}{number:02d}")
                    axes[i].axis("off")
                else:
                    axes[i, j].imshow(image, cmap=colormap)
                    axes[i, j].set_title(f"{letter}{number:02d}")
                    axes[i, j].axis("off")

            else:
                # If there are no images for a specific letter-number combination, remove the empty subplot
                if num_rows == 1:
                    fig.delaxes(axes[j])
                elif num_cols == 1:
                    fig.delaxes(axes[i])
                else:
                    fig.delaxes(axes[i, j])

    # Adjust the spacing and set aspect ratio to be equal
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save the plot at a higher resolution
    plt.savefig(output_path, format="tif", dpi=resolution, bbox_inches="tight")

    # If False plt.show() is not run to avoid loop stop upon display
    if show_fig:
        # Show the plot (optional)
        plt.show()


def segment_organoids(in_focus_organoids):
    """Processes individual z-stacks inside a folder and returns a dictionary containing organoid object masks stored as StackViewNDArray arrays"""

    segmented_organoids = {}
    images = list(in_focus_organoids.glob("*.TIF"))

    # Iterate through the files in the directory
    for input_img in tqdm(images):
        # Get the filename without the extension
        filename = input_img.stem

        # Load one RGB image and transform it into grayscale (if needed) for APOC
        rgb_img = tifffile.imread(input_img, is_ome=False)
        if len(rgb_img.shape) < 3:
            img = rgb_img
        elif rgb_img.shape[2] == 3:
            img = rgb2gray(rgb_img)
        else:
            print(
                "Modify the loader to accommodate different file formats",
                rgb_img.shape,
            )

        # Apply object segmenter from APOC
        try:
            segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
            result = segmenter.predict(image=img)
        except IndexError:
            segmenter = ObjectSegmenter(
                opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
            )
            result = segmenter.predict(image=img)

        # Closing some holes in the organoid labels
        closed_labels = cle.closing_labels(result, None, radius=4.0)

        # Exclude small labels, cutout in pixel area seems to be below 1000px
        exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
        exclude_small = np.array(
            exclude_small, dtype=np.int32
        )  # Change dtype of closed labels to feed array into nsbatwm.split

        # Splitting organoids into a binary mask
        split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

        # Connected component (cc) labeling
        cc_split_organoids = nsbatwm.connected_component_labeling(
            split_organoids, False
        )

        # Remove labels on edges
        edge_removed = nsbatwm.remove_labels_on_edges(cc_split_organoids)

        segmented_organoids[filename] = edge_removed

    return segmented_organoids


def segment_in_focus_organoids(in_focus_organoids):
    """Processes individual z-stacks inside a folder and returns a dictionary containing in-focus/out-of-focus organoid object masks stored as StackViewNDArray arrays"""

    focus_masks = {}
    images = list(in_focus_organoids.glob("*.TIF"))

    # Iterate through the files in the directory
    for input_img in tqdm(images):
        # Get the filename without the extension
        filename = input_img.stem

        # Load one RGB image and transform it into grayscale (if needed) for APOC
        rgb_img = tifffile.imread(input_img, is_ome=False)
        if len(rgb_img.shape) < 3:
            img = rgb_img
        elif rgb_img.shape[2] == 3:
            img = rgb2gray(rgb_img)
        else:
            print(
                "Modify the loader to accommodate different file formats",
                rgb_img.shape,
            )

        # Apply object segmenter from APOC
        try:
            segmenter = ObjectSegmenter(opencl_filename="./ObjectSegmenter.cl")
            result = segmenter.predict(image=img)
        except IndexError:
            segmenter = ObjectSegmenter(
                opencl_filename="./pretrained_APOC/ObjectSegmenter.cl"
            )
            result = segmenter.predict(image=img)

        # Closing some holes in the organoid labels
        closed_labels = cle.closing_labels(result, None, radius=4.0)

        # Exclude small labels, cutout in pixel area seems to be below 1000px
        exclude_small = cle.exclude_small_labels(closed_labels, None, 1000.0)
        exclude_small = np.array(
            exclude_small, dtype=np.int32
        )  # Change dtype of closed labels to feed array into nsbatwm.split

        # Splitting organoids into a binary mask
        split_organoids = nsbatwm.split_touching_objects(exclude_small, sigma=10.0)

        # Connected component (cc) labeling
        cc_split_organoids = nsbatwm.connected_component_labeling(
            split_organoids, False
        )

        # Remove labels on edges
        edge_removed = nsbatwm.remove_labels_on_edges(cc_split_organoids)

        # Apply object classifier from APOC
        try:
            classifier = ObjectClassifier(opencl_filename="./ObjectClassifier.cl")
            result = classifier.predict(labels=edge_removed, image=img)
        except AttributeError:
            classifier = ObjectClassifier(
                opencl_filename="./pretrained_APOC/ObjectClassifier.cl"
            )
            result = classifier.predict(labels=edge_removed, image=img)

        focus_masks[filename] = result

    return focus_masks


def save_object_mask(segmented_organoids, output_directory):
    # Iterate through the dictionary and save each array as a .tif file
    for filename, object_mask in segmented_organoids.items():
        # Create a filename for the .tif file within the output_directory
        save_filename = os.path.join(output_directory, f"{filename}.tif")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)

        # Save the cc_split_organoid array as a grayscale .tif file
        tifffile.imwrite(
            save_filename,
            object_mask,
            photometric="minisblack",
            dtype=object_mask.dtype,
        )


def random_cmap():
    np.random.seed(42)
    cmap = ListedColormap(np.random.rand(256, 4))
    # value 0 should just be transparent
    cmap.colors[:, 3] = 0.5
    cmap.colors[0, :] = 1
    cmap.colors[0, 3] = 0

    # if image is a mask, color (last value) should be red
    cmap.colors[-1, 0] = 1
    cmap.colors[-1, 1:3] = 0
    return cmap
