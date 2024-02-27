import os
import shutil

def list_folders(path):
    folders = []
    # Iterate over all items in the directory
    for item in os.listdir(path):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders

# # Example usage
# path = "/path/to/your/directory"
# folders = list_folders(path)
# print("Folders in", path, ":")
# for folder in folders:
#     print(folder)

def list_folders_until_files(path):
    folders = []
    # Walk through the directory structure
    for root, dirs, files in os.walk(path):
        if files:
            break  # Exit loop if files are found
        folders.append(root)
    return folders

import os

def leave_only_n_files(folder_path, num_files_to_keep):
    files = os.listdir(folder_path)
    files.sort()  # Sort files based on your preferred criteria
    
    # Determine excess files to remove
    files_to_remove = files[num_files_to_keep:]

    # Remove excess files
    for file_name in files_to_remove:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        print("Removed file:", file_name)
                
# Example usage
path = "./DATA/oct"
folders_until_files_list = list_folders(path)
print("Folders until files are encountered:")
for folder in folders_until_files_list:
    print(f"For folder {folder}")
    sub_folders = list_folders(path + "/" + folder + "/")

    for sub in sub_folders:
        leave_only_n_files(path + "/" + folder + "/" + sub, 30)
