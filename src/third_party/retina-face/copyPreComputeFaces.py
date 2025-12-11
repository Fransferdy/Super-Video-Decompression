import os
import json
import shutil

def copy_images_with_faces(hq_folder, lq_folder, faces_json_path, output_root="./withFaces"):
    # --- Load face positions ---
    with open(faces_json_path, "r") as f:
        faces_pos = json.load(f)

    # --- Prepare output directories ---
    hq_out = os.path.join(output_root, "hq")
    lq_out = os.path.join(output_root, "lq")
    os.makedirs(hq_out, exist_ok=True)
    os.makedirs(lq_out, exist_ok=True)

    copied = 0
    total = 0
    filtered_faces_pos = {}
    for img_name, faces in faces_pos.items():
        total += 1
        if len(faces) == 0:
            continue  # skip images with no faces
        filtered_faces_pos[img_name] = faces
        # Define paths
        hq_src = os.path.join(hq_folder, img_name)
        lq_src = os.path.join(lq_folder, img_name)

        hq_dst = os.path.join(hq_out, img_name)
        lq_dst = os.path.join(lq_out, img_name)

        # Copy only if files exist
        if os.path.exists(hq_src) and os.path.exists(lq_src):
            shutil.copy2(hq_src, hq_dst)
            shutil.copy2(lq_src, lq_dst)
            copied += 1
        else:
            print(f"⚠️ Missing file: {img_name}")
    with open(os.path.join(output_root, "faces_pos.json"), "w") as f:
        json.dump(filtered_faces_pos, f)
    
    print(f"✅ Copied {copied}/{total} images that contain faces.")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    # Example usage:
    hq_folder = r"C:\\Users\\Fernando\\Documents\\Unicamp\\IA_376_GAN\\workspace\\dgm-2025.2\\projects\\super-video-decompression\\data\\processed\\lq_34_2X_splitdb\\train\\hq"
    lq_folder = r"C:\\Users\\Fernando\\Documents\\Unicamp\\IA_376_GAN\\workspace\\dgm-2025.2\\projects\\super-video-decompression\\data\\processed\\lq_34_2X_splitdb\\train\\lq"
    faces_json = r"./faces_pos.json"

    copy_images_with_faces(hq_folder, lq_folder, faces_json)