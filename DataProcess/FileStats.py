import os

def count_files_in_folder(folder_path):
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def count_files_in_all_folders(base_directory):
    for root, _, _ in os.walk(base_directory):
        folder_name = os.path.relpath(root, base_directory)
        file_count = count_files_in_folder(root)
        print(f"Folder '{folder_name}': {file_count} files")

if __name__ == "__main__":
    directory_to_scan = "/home/abdkhan/myfsl/dataset/ScanObjectL1"  # Replace with the directory you want to scan
    count_files_in_all_folders(directory_to_scan)
