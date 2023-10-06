# Library imports
import os
import concurrent.futures
from pathlib import Path
from utils import (
    read_images,
    find_focus,
    find_highest_infocus,
    store_imgs,
    plot_plate,
    segment_organoids,
    random_cmap,
    save_object_mask,
    segment_in_focus_organoids,
)
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap

import warnings

# Filter out the specific warning
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in long_scalars",
)

# ---------------- USER INPUT NEEDED BELOW ---------------- #

# Define your data directory (folder containing the subfolders storing your plate images)
PARENT_FOLDER = Path("./data/Andrew/202309_Org_ApcFlox_Lsd1i_Expt1")

# Define the output resolution of your plate plots and username
RESOLUTION = 300
USERNAME = "Andrew"

# Choose which plate views do you need (i.e. grayscale, organoid_object, in_focus)
PLATE_VIEWS = ["grayscale", "organoid_object", "in_focus"]

# ---------------- USER INPUT NEEDED ABOVE ---------------- #


# Function to process a single folder for multithreading
def process_folder(folder):
    directory_path = PARENT_FOLDER.joinpath(folder)
    print(directory_path)

    # The following function will read all the images contained within the directory_path above
    # and store them grouped by well_id.
    images_per_well = read_images(directory_path)

    # Compute the nr of organoids in focus per well
    nr_infocus_organoids = find_focus(images_per_well)

    # Store a .csv copy of the max_index_dict containing the percentages of organoids in focus
    # Create a Pandas DataFrame
    df = pd.DataFrame(nr_infocus_organoids)

    # Specify the output directory path
    directory = Path(f"./output/{USERNAME}")
    output_directory = directory.joinpath(folder)

    # Check if the output directory already exists and create it if it does not
    try:
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{output_directory}' already exists.")

    # Save the DataFrame as a .csv file
    df.to_csv(
        f"./{str(output_directory)}/Percentage_in_focus_per_well_{str(folder)}.csv",
        index=False,
    )

    # Finding the z-stack with the most organoids in-focus
    max_index_dict = find_highest_infocus(nr_infocus_organoids)

    # In case one of the wells has no detectable organoids in focus, this will substitute the focal plane
    # with an average of all focal planes in the plate

    # Calculate the average of all values in the dictionary
    average_value = round(sum(max_index_dict.values()) / len(max_index_dict))

    # Substitute the 0 values for the average focal plane
    for well, in_focus_stack in max_index_dict.items():
        if in_focus_stack == 0:
            max_index_dict[well] = average_value

    # Storing a copy of each z-stack with the most organoids in focus
    store_imgs(
        images_per_well,
        max_index_dict,
        output_dir=f"{output_directory}/in_focus_organoids",
    )


def save_organoid_segmentation(in_focus_organoids):
    """Reads a folder containing grayscale images, segments the organoids and saves the resulting masks in a new folder"""
    # segment_organoids() returns a dictionary where the organoid labels are stored under each well_id key
    segmented_organoids = segment_organoids(Path(in_focus_organoids))

    # Define the directory path where you want to save the segmented organoid masks
    # Split the in_focus_organoids path to obtain the folder that is one level up (head)
    head, tail = os.path.split(in_focus_organoids)
    output_directory = os.path.join(head, "segmented_organoids")

    # Save the segmented organoid masks contained in segmented_organoids in the above defined output directory
    save_object_mask(segmented_organoids, output_directory)


if __name__ == "__main__":
    # Initialize an empty list to store subfolder names
    subfolder_list = []

    # Iterate over subdirectories in the parent folder
    for subfolder in PARENT_FOLDER.iterdir():
        if subfolder.is_dir() and "4X" not in str(subfolder):
            subfolder_list.append(subfolder.name)

    # Process folders in parallel and extract in-focus images
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_folder, subfolder_list)

    # Plot grayscale images plate view
    if "grayscale" in PLATE_VIEWS:
        for folder in subfolder_list:
            # Specify the output directory path
            directory = Path(f"./output/{USERNAME}")
            output_directory = directory.joinpath(folder)
            # Generate matplotlib plate view and store it
            plot_plate(
                resolution=RESOLUTION,
                output_path=f"./{str(output_directory)}/organoid_greyscale_plot.tif",
                img_folder_path=f"{output_directory}/in_focus_organoids",
                show_fig=False,
            )

    # Generate and save organoid segmentation images, then plot plate view
    if "organoid_object" in PLATE_VIEWS:
        in_focus_org_dirs = []
        org_masks_dirs = []

        for folder in subfolder_list:
            # Specify the in_focus_organoids directory path within output
            directory = Path(f"./output/{USERNAME}")
            output_directory = directory.joinpath(folder)
            in_focus_org_directory = f"{output_directory}/in_focus_organoids"
            organoid_mask_directory = f"{output_directory}/segmented_organoids"
            in_focus_org_dirs.append(in_focus_org_directory)
            org_masks_dirs.append(organoid_mask_directory)

        # Process folders in parallel, extract organoid segmentation masks and store them as .tif files
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(save_organoid_segmentation, in_focus_org_dirs)

        # Plot organoid segmentation plate views using a for loop ()
        cmap = random_cmap()

        print("Generating and storing organoid mask plate views")

        for org_mask_dir in tqdm(org_masks_dirs):
            # Generate the filepath for each organoid mask plate view
            head, tail = os.path.split(org_mask_dir)
            output_path = os.path.join(head, "organoid_object_plot.tif")
            # Generate and save the plot
            plot_plate(
                resolution=RESOLUTION,
                output_path=output_path,
                img_folder_path=org_mask_dir,
                colormap=cmap,
                show_fig=False,
            )

    # Generate and save in/out-of-focus organoid segmentation images, then plot plate view
    if "in_focus" in PLATE_VIEWS:
        in_focus_org_dirs = []
        focus_masks_dirs = []

        for folder in subfolder_list:
            # Specify the in_focus_organoids directory path within output
            directory = Path(f"./output/{USERNAME}")
            output_directory = directory.joinpath(folder)
            in_focus_org_directory = f"{output_directory}/in_focus_organoids"
            focus_mask_directory = f"{output_directory}/in_out_focus_masks"
            in_focus_org_dirs.append(in_focus_org_directory)
            focus_masks_dirs.append(focus_mask_directory)

        # Process folders in parallel, extract organoid segmentation masks and store them as .tif files
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(save_focus_segmentation, in_focus_org_dirs)

        # Plot focus classification using a for loop ()
        # Define the colors for each value
        colors = [
            (0, 0, 0, 0),  # Transparent for 0
            (0.647, 0.165, 0.165, 1),  # Brown for 1 (out of focus)
            (0.678, 0.847, 0.902, 1),
        ]  # Light blue for 2 (in focus)

        # Create a colormap using ListedColormap
        custom_cmap = ListedColormap(colors)

        print("Generating and storing focus classification plate views")

        for focus_mask_dir in tqdm(focus_masks_dirs):
            # Generate the filepath for each organoid mask plate view
            head, tail = os.path.split(focus_mask_dir)
            output_path = os.path.join(head, "focus_classification_plot.tif")

            plot_plate(
                resolution=RESOLUTION,
                output_path=output_path,
                img_folder_path=focus_mask_dir,
                colormap=custom_cmap,
                show_fig=False,
            )

# TODO: Move directory generation loops out of the "PLATE_VIEWS" conditions
