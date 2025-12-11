import os
import subprocess
import sys

def main(folder_path):
    progress_file = "progress.txt"

    # Load last processed index
    start_index = 0

    # Get all .toml files
    files = [f for f in os.listdir(folder_path) if f.endswith(".toml")]
    files.sort()

    print(f"Found {len(files)} toml files. Starting at index {start_index}.")

    for i in range(start_index, len(files)):
        file = files[i]
        config_path = os.path.join(folder_path, file)

        cmd = [
            "./.venv/Scripts/python",
            "optsconvert.py",
            "-opt",
            config_path,
            "--fp16",
            "--optimize",
            "--output", 
            "./onnx/"+file+".onnx"
        ]

        print(f"\n[{i+1}/{len(files)}] Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"Error running {file}: {e}")



    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <folder_with_toml_files>")
        sys.exit(1)

    main(sys.argv[1])