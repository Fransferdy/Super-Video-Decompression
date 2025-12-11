import os
import shutil
from retinaface import RetinaFace
from tqdm import tqdm

def find_images_with_faces(folder_path, valid_exts={".jpg", ".jpeg", ".png",".webp"}):
    """
    Scans all images in a folder and returns a list of filenames that contain at least one detected face.

    Args:
        folder_path (str): Path to folder containing images.
        valid_exts (set): Allowed image extensions.

    Returns:
        list[str]: Filenames (not full paths) that contain faces.
    """
    print(folder_path)
    image_names_with_faces = []
    for fname in tqdm(os.listdir(folder_path), desc=f"Scanning {os.path.basename(folder_path)}"):
        if not any(fname.lower().endswith(ext) for ext in valid_exts):
            continue
        
        img_path = os.path.join(folder_path, fname)
        print(f"Scanning {img_path}")
        try:
            faces = RetinaFace.detect_faces(img_path)
            if isinstance(faces, dict) and len(faces) > 0:
                image_names_with_faces.append(fname)
        except Exception as e:
            print(f"[WARN] Failed to detect faces in {fname}: {e}")

    return image_names_with_faces


def copy_filtered_images(face_images, lq_folder, hq_folder, output_base):
    """
    Copies matching LQ and HQ images for files that contain faces
    into /imagesWithFaces/lq and /imagesWithFaces/hq.

    Args:
        face_images (list[str]): filenames that contain faces
        lq_folder (str): path to low-quality images folder
        hq_folder (str): path to high-quality images folder
        output_base (str): where filtered images will be copied
    """
    lq_out = os.path.join(output_base, "lq")
    hq_out = os.path.join(output_base, "hq")
    os.makedirs(lq_out, exist_ok=True)
    os.makedirs(hq_out, exist_ok=True)

    for fname in tqdm(face_images, desc="Copying filtered images"):
        for src_folder, dst_folder in [(hq_folder, hq_out), (lq_folder, lq_out)]:
            src = os.path.join(src_folder, fname)
            dst = os.path.join(dst_folder, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"[WARN] Missing file: {src}")


if __name__ == "__main__":
    HQ_FOLDER = "C:\\Users\\Fernando\\Documents\\Unicamp\\IA_376_GAN\\workspace\\dgm-2025.2\\projects\\super-video-decompression\\data\\processed\\lq_28_1xImprov_splitdb\\val\\hq"
    LQ_FOLDER = "C:\\Users\\Fernando\\Documents\\Unicamp\\IA_376_GAN\\workspace\\dgm-2025.2\\projects\\super-video-decompression\\data\\processed\\lq_28_1xImprov_splitdb\\val\\lq"
    OUTPUT_FOLDER = "./imagesWithFaces"

    print("🔍 Scanning HQ images for faces...")
    face_images = find_images_with_faces(HQ_FOLDER)

    print(f"✅ Found {len(face_images)} images with faces.")
    print("📦 Copying corresponding HQ and LQ images...")
    copy_filtered_images(face_images, LQ_FOLDER, HQ_FOLDER, OUTPUT_FOLDER)

    print("✅ Done! Filtered images saved in:", OUTPUT_FOLDER)