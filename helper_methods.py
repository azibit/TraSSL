import os, torch, random, glob, shutil
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime

def copy_files_with_limit(source_dir, destination_dir, limit):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Get the list of directories in the source directory
    subdirectories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    all_files = []
    new_subdir = "result"

    # Create corresponding directory in the destination folder
    destination_subdir = os.path.join(destination_dir, new_subdir)
    if not os.path.exists(destination_subdir):
        os.makedirs(destination_subdir)

    # Iterate over each subdirectory
    for subdir in subdirectories:
        source_subdir = os.path.join(source_dir, subdir)

        # Get the list of files in the subdirectory
        files = glob.glob(source_subdir + "/*")
        all_files.extend(files)
    
    random.shuffle(all_files)

    # Iterate over files in the subdirectory
    files_copied = 0
    for filename in all_files:
        local_filename = filename.split("/")[-1]
        destination_file = os.path.join(destination_subdir, local_filename)

        # Check if the file already exists in the destination directory
        if os.path.exists(destination_file):
            # If the file already exists, rename it
            new_filename = local_filename.split('.')[0] + "_copy." + local_filename.split('.')[1]
            destination_file = os.path.join(destination_subdir, new_filename)

        # Copy the file
        shutil.copy2(filename, destination_file)
        files_copied += 1

        # Break if the limit is reached
        if files_copied == limit:
            break

def create_folder_with_timestamp(filename):
    # Get current date and time
    current_datetime = datetime.now()
    
    # Format the date and time as a string (e.g., "2024-02-25_145930")
    timestamp = current_datetime.strftime("%Y-%m-%d_%H")
    
    # Create folder name using the timestamp
    folder_name = f"{filename}_{timestamp}"
    
    # Create the new folder
    os.makedirs(folder_name)
    return folder_name

def count_folders(folder_name):
    try:
        return sum(os.path.isdir(entry.path) for entry in os.scandir(folder_name))
    except OSError as e:
        print(f"Error: {e}")
        return None

def select_samples(dataset_name, select_random, sort_method, number_of_samples):

    selected_samples = []

    if select_random:
        selected_samples = random.sample(glob.glob(f'./DATA/{dataset_name}/train/*/*'), number_of_samples)
    else:
        result_list = f'{dataset_name}_result_list.csv'
        df = pd.read_csv(result_list)

        selected_samples = sort_dataframe_and_get_paths(df=df, sort_method=sort_method, number_of_samples=number_of_samples)

    return selected_samples

def sort_dataframe_and_get_paths(df, sort_method, number_of_samples, dataset_name):

    """
    sort_method:
        0 - np.lexsort((loss_1, confusion_classes)) # Sort by loss after sorting by confusion classes
        1 - np.lexsort((kl_uncertainties, confusion_classes)) # Sort by uncertainty after sorting by confusion classes
        2 - np.argsort(loss_1) # Sort by loss
        3 - np.argsort(kl_uncertainties) # Sort by uncertainty
    """

    df = df[df['path'].str.contains(dataset_name)]

    if sort_method == 0:
        sorted_indices = torch.tensor(np.lexsort((df['individual_losses'], df['individual_correct_preds'])))
    elif sort_method == 1:
        sorted_indices = torch.tensor(np.lexsort((df['individual_kl_scores'], df['individual_correct_preds'])))
    elif sort_method == 2:
        sorted_indices = torch.argsort(torch.tensor(df['individual_losses']))
    elif sort_method == 3:
        sorted_indices = torch.argsort(torch.tensor(df['individual_kl_scores']))
    else:
        raise ValueError("Invalid sort_method value. Supported values are 0, 1, 2, or 3.")

    sorted_indices = sorted_indices.tolist()
    # Reverse the sorting order for higher losses within each confusion class
    sorted_indices = sorted_indices[::-1]

    # Select evenly through the samples
    sampled_indices = sorted_indices[::int(len(sorted_indices) / number_of_samples)][:number_of_samples]

    result = df.loc[sampled_indices, 'path']

    # Return 'path' column for the sorted indices
    return result.tolist()


def write_list_to_file(file_path, data_list):
        """
        Write a list of strings to a file.

        Parameters:
        - file_path: The path to the file.
        - data_list: The list of strings to be written to the file.
        """
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(str(item) + '\n')

def create_empty_file(file_name):
        # Check if the file exists
        if os.path.exists(file_name):
            # If it exists, delete it
            os.remove(file_name)

        # Open the file in write mode, creating it if it doesn't exist
        with open(file_name, 'w'):
            pass  # This is an empty block, creating an empty file

def compute_KL_divergence_score(tensor):
        # Compute the KL divergence from a uniform distribution
        kl_divergence = F.kl_div(F.log_softmax(tensor, dim=0), torch.ones_like(tensor) / tensor.size(0))

        return kl_divergence.item()

def sort_method_switch(sort_method, loss_1, kl_uncertainties, confusion_classes):

    """
    sort_method:
        0 - np.lexsort((loss_1, confusion_classes))
        1 - np.lexsort((kl_uncertainties, confusion_classes))
        2 - np.argsort(loss_1)
        3 - np.argsort(kl_uncertainties)
    """

    if sort_method == 0:
        return torch.lexsort((loss_1, confusion_classes))
    elif sort_method == 1:
        return torch.lexsort((kl_uncertainties, confusion_classes))
    elif sort_method == 2:
        return torch.argsort(loss_1)
    elif sort_method == 3:
        return torch.argsort(kl_uncertainties)
    else:
        raise ValueError("Invalid sort_method value. Supported values are 0, 1, 2, or 3.")


def sort_and_update_file(file_path_for_rotation_loss, output_file_path, sort_method = 0):

    # Read losses from the file
    with open(file_path_for_rotation_loss, 'r') as f:
        losses = f.readlines()

    # Process loss values and names
    loss_1 = []
    name_2 = []
    confusion_classes = []
    kl_uncertainties = []

    for j in losses:
        loss_value = float(j[:-1].split('_')[0])
        loss_1.append(loss_value)

        confusion_class = j[:-1].split('_')[1]
        confusion_classes.append(confusion_class)

        kl_uncertainty_score = j[:-1].split('_')[2]
        kl_uncertainties.append(kl_uncertainty_score)

        name = j[:-1].split('_')[3]
        name_2.append(name)
        
    sort_index = sort_method_switch(sort_method, loss_1, kl_uncertainties, confusion_classes)

    # Reverse the sorting order for higher losses within each confusion class
    sort_index = sort_index[::-1]

    sorted_file_name_list = []

    # Update the file with sorted names
    for k in sort_index:
        sorted_file_name_list.append(name_2[k])

    write_list_to_file(output_file_path, sorted_file_name_list)