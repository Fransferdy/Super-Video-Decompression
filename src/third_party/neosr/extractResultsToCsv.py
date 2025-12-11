import os
import csv
import re

def extract_metrics_from_file(path):
    """
    Reads the .txt file at 'path' and extracts metrics (psnr, ssim, dists)
    from lines that start with '#'.
    """
    metrics = {}
    pattern = r"#\s*(\w+):\s*([0-9.]+)"  # captures key:value

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "# " in line:  # The lines we want always contain '# '
                match = re.search(pattern, line)
                if match:
                    key = match.group(1).lower()
                    val = match.group(2)
                    metrics[key] = val
    return metrics


def scan_folder(root_folder, output_csv="results.csv"):
    results = []

    for sub in os.listdir(root_folder):
        print(f"Processing {sub}")
        sub_path = os.path.join(root_folder, sub)
        if not os.path.isdir(sub_path):
            continue

        # Find .txt file inside subfolder
        txt_files = [f for f in os.listdir(sub_path) if f.endswith(".log")]
        if not txt_files:
            continue

        txt_path = os.path.join(sub_path, txt_files[0])
        metrics = extract_metrics_from_file(txt_path)
        metrics["folder"] = sub

        results.append(metrics)

    # Determine all keys for CSV header
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())

    # Ensure "folder" is the first column
    header = ["folder"] + sorted([k for k in all_keys if k != "folder"])

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV to: {output_csv}")


# ---------------------------
# Run the script
# ---------------------------

if __name__ == "__main__":
    scan_folder("./experiments/results/", "./metrics_output.csv")