# Library imports
import os
from pathlib import Path
from utils import read_images, find_focus, find_highest_infocus, store_imgs, plot_plate
from tqdm import tqdm
import pandas as pd

# Define your data directory (folder containing the subfolders storing your plate images)
parent_folder = Path("./data/Andrew/202309_Org_ApcFlox_Lsd1i_Expt1")

# Define the output resolution of your plate plots and username
RESOLUTION = 300
USERNAME = "Andrew"

# Initialize an empty list to store subfolder names
subfolder_list = []

# Iterate over subdirectories in the parent folder
for subfolder in parent_folder.iterdir():
    if subfolder.is_dir() and "4X" not in str(subfolder):
        subfolder_list.append(subfolder.name)

# Iterate over each of the folders containing the images and plot the plates
for folder in tqdm(subfolder_list):
    directory_path = parent_folder.joinpath(folder)
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

    plot_plate(
        resolution=RESOLUTION,
        output_path=f"./{str(output_directory)}/organoid_greyscale_plot.tif",
        img_folder_path=f"{output_directory}/in_focus_organoids",
        show_fig=False,
    )
