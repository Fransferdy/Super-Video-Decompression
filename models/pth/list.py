import os

def list_files_in_folder(folder_path):
    """
    Reads all file and directory names from the given folder path and prints them.

    Args:
        folder_path (str): The absolute or relative path to the folder.
    """
    print(f"--- Listing contents of: {folder_path} ---")

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at path: {folder_path}")
        return

    # Check if the path is actually a directory
    if not os.path.isdir(folder_path):
        print(f"Error: Path exists but is not a directory: {folder_path}")
        return

    try:
        # os.listdir() returns a list containing the names of the entries
        # in the directory given by path.
        contents = os.listdir(folder_path)

        if not contents:
            print("The folder is empty.")
            return

        # Print each item found in the directory
        for item in contents:
            full_path = os.path.join(folder_path, item)
            # Differentiate between files and directories for clarity
            if os.path.isfile(full_path):
                print(f"[FILE] {item}")
            elif os.path.isdir(full_path):
                print(f"[DIR]  {item}")
            else:
                print(f"[OTHER] {item}")

    except PermissionError:
        print(f"Error: Permission denied to access folder: {folder_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Change this path to the folder you want to read.
    # '.' refers to the current directory where the script is executed.
    target_folder = "."

    list_files_in_folder(target_folder)

    print("\n--- Listing contents of a specific folder (e.g., your home folder) ---")
    # You can also use a specific path, e.g.:
    # list_files_in_folder("/Users/YourName/Documents") # On Mac/Linux
    # list_files_in_folder("C:\\Users\\YourName\\Documents") # On Windows

    # Example using a common user directory (needs modification by user)
    # import getpass
    # user_name = getpass.getuser()
    # list_files_in_folder(f"/Users/{user_name}")