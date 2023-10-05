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
import pandas as pd
from matplotlib.colors import ListedColormap

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

    # Plot plate grayscale, object detection and in/out-of-focus organoid masks

    # Grayscale
    if "grayscale" in PLATE_VIEWS:
        plot_plate(
            resolution=RESOLUTION,
            output_path=f"./{str(output_directory)}/organoid_greyscale_plot.tif",
            img_folder_path=f"{output_directory}/in_focus_organoids",
            show_fig=False,
        )

    if "organoid_object" in PLATE_VIEWS:
        # Define the directory path where your in-focus organoid z-stack have been stored
        in_focus_organoids = Path(f"{output_directory}/in_focus_organoids")

        # segment_organoids() returns a dictionary where the organoid labels are stored under each well_id key
        segmented_organoids = segment_organoids(in_focus_organoids)

        # Define the directory path where you want to save the segmented organoid masks
        output_directory_orgm = f"{output_directory}/segmented_organoids/"

        # Save the segmented organoid masks contained in segmented_organoids in the above defined output directory
        save_object_mask(segmented_organoids, output_directory_orgm)

        # Plot organoid object plateview
        cmap = random_cmap()

        plot_plate(
            resolution=RESOLUTION,
            output_path=f"./{str(output_directory)}/organoid_object_plot.tif",
            img_folder_path=f"{output_directory}/segmented_organoids/",
            show_fig=False,
            colormap=cmap,
        )

    if "in_focus" in PLATE_VIEWS:
        # Define the directory path where your in-focus organoid z-stack have been stored
        in_focus_organoids = Path(f"{output_directory}/in_focus_organoids")

        # segment_in_focus_organoids() returns a dictionary where the organoid labels are stored under each well_id key
        segmented_in_focus_organoids = segment_in_focus_organoids(in_focus_organoids)

        # Define the directory path where you want to save the segmented organoid masks
        output_focus_directory = f"{output_directory}/in_out_focus_masks/"

        # Save the in-focus segmented organoid masks contained in segmented_in_focus_organoids in the above defined output directory
        save_object_mask(segmented_in_focus_organoids, output_focus_directory)

        # Define the colors for each value
        colors = [
            (0, 0, 0, 0),  # Transparent for 0
            (0.647, 0.165, 0.165, 1),  # Brown for 1 (out of focus)
            (0.678, 0.847, 0.902, 1),
        ]  # Light blue for 2 (in focus)

        # Create a colormap using ListedColormap
        custom_cmap = ListedColormap(colors)

        plot_plate(
            resolution=RESOLUTION,
            output_path=f"./{str(output_directory)}/organoid_focus_masks_plot.tif",
            img_folder_path=f"{output_directory}/in_out_focus_masks/",
            show_fig=False,
            colormap=custom_cmap,
        )


# Initialize an empty list to store subfolder names
subfolder_list = []

# Iterate over subdirectories in the parent folder
for subfolder in PARENT_FOLDER.iterdir():
    if subfolder.is_dir() and "4X" not in str(subfolder):
        subfolder_list.append(subfolder.name)

# Process folders in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_folder, subfolder_list)
