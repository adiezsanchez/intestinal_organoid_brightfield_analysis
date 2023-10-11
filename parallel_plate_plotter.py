# Library imports
import os
import concurrent.futures
from pathlib import Path
from utils import (
    process_folder,
    plot_plate,
    save_organoid_segmentation,
    save_focus_segmentation,
    random_cmap,
)
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap

# Initialize GPU-acceleration if available
import pyclesperanto_prototype as cle  # version 0.24.1

try:
    device = cle.select_device("TX")
    print("Used GPU: ", device)
except:
    print("No GPU acceleration available, script will run on the CPU")

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

# ---------------- USER INPUT NEEDED ABOVE ---------------------------- #

# ---------------- PARALLEL PROCESSING FUNCTIONS ---------------------- #


def process_folder_wrapper(folder):
    """Wrapper function to pass additional arguments to process_folder"""
    process_folder(folder, PARENT_FOLDER, USERNAME)


# ---------------- SCRIPT ---------------------- #

if __name__ == "__main__":
    # Initialize an empty list to store subfolder names
    subfolder_list = []

    # Iterate over subdirectories in the parent folder
    for subfolder in PARENT_FOLDER.iterdir():
        if subfolder.is_dir() and "4X" not in str(subfolder):
            subfolder_list.append(subfolder.name)

    # Process folders in parallel and extract in-focus images
    print("Extracting in-focus z-stacks")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_folder_wrapper, subfolder_list)

    # Generate directory lists
    if len(PLATE_VIEWS) >= 1:
        for folder in subfolder_list:
            # Specify the output directory path
            directory = Path(f"./output/{USERNAME}")
            output_directory = directory.joinpath(folder)

        if "organoid_object" or "in_focus" in PLATE_VIEWS:
            # Create empty lists to hold all the directories
            in_focus_org_dirs = []
            org_masks_dirs = []
            focus_masks_dirs = []

            # Create all necessary subfolders within the output_directory
            in_focus_org_directory = f"{output_directory}/in_focus_organoids"
            organoid_mask_directory = f"{output_directory}/segmented_organoids"
            focus_mask_directory = f"{output_directory}/in_out_focus_masks"
            in_focus_org_dirs.append(in_focus_org_directory)
            org_masks_dirs.append(organoid_mask_directory)
            focus_masks_dirs.append(focus_mask_directory)

    # Plot grayscale images plate view
    if "grayscale" in PLATE_VIEWS:
        print("Generating and storing grayscale images plate views")

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
