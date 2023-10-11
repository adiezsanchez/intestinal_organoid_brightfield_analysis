import os
import functools
import threading
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
import pyclesperanto_prototype as cle
from tkinter import (
    ttk,
    Label,
    Entry,
    Button,
    Tk,
    StringVar,
    Checkbutton,
    Text,
    Scrollbar,
)

# Initialize GPU-acceleration if available
import pyclesperanto_prototype as cle  # version 0.24.1

try:
    device = cle.select_device("TX")
    print("Used GPU: ", device)
except:
    print("No GPU acceleration available, script will run on the CPU")
    pass

# Filter out the specific warning
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in long_scalars",
)


def run_parallel_plate_plotter(
    parent_folder, resolution, username, plate_views, output_text, result_label
):
    # Redirect print statements to the Text widget
    def custom_print(*args, **kwargs):
        output_text.insert("end", " ".join(map(str, args)) + "\n")
        output_text.yview("end")

    # Replace the built-in print function with custom_print
    print = custom_print

    # Initialize an empty list to store subfolder names
    subfolder_list = []

    # Iterate over subdirectories in the parent folder
    for subfolder in parent_folder.iterdir():
        if subfolder.is_dir() and "4X" not in str(subfolder):
            subfolder_list.append(subfolder.name)

    # Process folders in parallel and extract in-focus images
    print("Extracting in-focus z-stacks")
    # Create a partial function with fixed parameters
    partial_process_folder = functools.partial(
        process_folder, parent_folder=parent_folder, username=username
    )
    # Use the partial function in executor.map
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(partial_process_folder, subfolder_list)

    # Generate directory lists
    if len(plate_views) >= 1:
        for folder in subfolder_list:
            # Specify the output directory path
            directory = Path(f"./output/{username}")
            output_directory = directory.joinpath(folder)

            if "organoid_object" or "in_focus" in plate_views:
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
    if "grayscale" in plate_views:
        print("Generating and storing grayscale image plate views")

        for folder in subfolder_list:
            # Specify the output directory path
            directory = Path(f"./output/{username}")
            output_directory = directory.joinpath(folder)
            # Generate matplotlib plate view and store it
            plot_plate(
                resolution=resolution,
                output_path=f"./{str(output_directory)}/organoid_greyscale_plot.tif",
                img_folder_path=f"{output_directory}/in_focus_organoids",
                show_fig=False,
            )

    # Generate and save organoid segmentation images, then plot plate view
    if "organoid_object" in plate_views:
        in_focus_org_dirs = []
        org_masks_dirs = []

        for folder in subfolder_list:
            # Specify the in_focus_organoids directory path within output
            directory = Path(f"./output/{username}")
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
                resolution=resolution,
                output_path=output_path,
                img_folder_path=org_mask_dir,
                colormap=cmap,
                show_fig=False,
            )

    # Generate and save in/out-of-focus organoid segmentation images, then plot plate view
    if "in_focus" in plate_views:
        in_focus_org_dirs = []
        focus_masks_dirs = []

        for folder in subfolder_list:
            # Specify the in_focus_organoids directory path within output
            directory = Path(f"./output/{username}")
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
                resolution=resolution,
                output_path=output_path,
                img_folder_path=focus_mask_dir,
                colormap=custom_cmap,
                show_fig=False,
            )


# Tkinter GUI
def on_submit():
    parent_folder = Path(parent_folder_entry.get())
    resolution = resolution_entry.get()
    username = username_entry.get()
    plate_views = [view.get() for view in plate_views_checkbuttons.values()]

    try:
        resolution = int(resolution)
        if 100 <= resolution <= 600:
            # Pass the output_text widget and result_label to the thread
            threading.Thread(
                target=execute_task,
                args=(
                    parent_folder,
                    resolution,
                    username,
                    plate_views,
                    output_text,
                    result_label,
                ),
            ).start()
        else:
            result_label.config(text="Resolution must be between 100 and 600")
    except ValueError:
        result_label.config(text="Resolution must be an integer")


def execute_task(
    parent_folder, resolution, username, plate_views, output_text, result_label
):
    try:
        run_parallel_plate_plotter(
            parent_folder, resolution, username, plate_views, output_text, result_label
        )
        root.after(
            0,
            result_label.config,
            {
                "text": f"Script executed successfully. You can find your data under ./output/{username}"
            },
        )
    except Exception as e:
        root.after(0, result_label.config, {"text": f"Error: {str(e)}"})


root = Tk()
root.title("Parallel Plate Plotter")

# Input fields
Label(root, text="Parent Folder:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
parent_folder_entry = Entry(root, width=100)  # Adjust width here
parent_folder_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

Label(root, text="Resolution in dpi (100-600):").grid(
    row=1, column=0, padx=10, pady=5, sticky="w"
)
resolution_entry = Entry(root)
resolution_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

Label(root, text="Username:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
username_entry = Entry(root)
username_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

# Text label above checkbuttons
checkbuttons_label = Label(root, text="Choose the plate views you wish to generate:")
checkbuttons_label.grid(row=3, column=0, columnspan=2, pady=5)

# Checkbuttons for plate views
plate_views_checkbuttons = {}
plate_views_frame = ttk.Frame(root)
plate_views_frame.grid(row=4, column=0, columnspan=2, pady=10)

for i, view in enumerate(["grayscale", "organoid_object", "in_focus"]):
    var = StringVar(value="")  # Set to empty initially
    checkbutton = Checkbutton(
        plate_views_frame, text=view, variable=var, onvalue=view, offvalue=""
    )
    checkbutton.grid(row=3, column=i, padx=5)
    plate_views_checkbuttons[view] = var

# Submit button
submit_button = Button(root, text="Generate plate views", command=on_submit)
submit_button.grid(row=5, column=0, columnspan=2, pady=10)

# Output Text widget
output_text = Text(root, height=10, width=100, wrap="word")
output_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Scrollbar for the text widget
scrollbar = Scrollbar(root, command=output_text.yview)
scrollbar.grid(row=6, column=2, sticky="nsew")
output_text.config(yscrollcommand=scrollbar.set)

# Result label
result_label = Label(root, text="")
result_label.grid(row=7, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    root.mainloop()
