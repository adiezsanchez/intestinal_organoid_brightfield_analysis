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
