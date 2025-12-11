import os
import argparse

def rename_webp_files(folder):
    # List all .webp files
    files = [f for f in os.listdir(folder) if f.lower().endswith(".webp")]

    for fname in files:
        parts = fname.split("_")
        if len(parts) > 2:  # ensure there's something to remove
            # Remove the second part
            new_parts = [parts[0]] + parts[2:]
            new_name = "_".join(new_parts)
            
            old_path = os.path.join(folder, fname)
            new_path = os.path.join(folder, new_name)
            
            # Rename only if new name is different
            if old_path != new_path:
                print(f"Renaming: {fname} → {new_name}")
                os.rename(old_path, new_path)
        else:
            print(f"Skipping {fname}, not enough parts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename .webp files by removing second underscore part.")
    parser.add_argument("folder", type=str, help="Folder containing .webp files")
    args = parser.parse_args()

    rename_webp_files(args.folder)