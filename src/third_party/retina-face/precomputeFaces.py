import os
import json
from retinaface import RetinaFace
from tqdm import tqdm

def find_faces_in_folder(hq_folder: str):
    """
    Detect faces in all images in the given HQ folder.
    Saves results as a JSON file: parent_dir/faces_pos.json

    Output format:
    {
        "image1.jpg": [
            {"x1": 155, "y1": 81, "x2": 434, "y2": 443},
            {"x1": 500, "y1": 120, "x2": 600, "y2": 240}
        ],
        "image2.png": []
    }
    """
    hq_folder = os.path.abspath(hq_folder)
    parent_dir = os.path.dirname(hq_folder)
    output_path = os.path.join(parent_dir, "faces_pos.json")

    results = {}
    
    total_width = 0.0
    total_height = 0.0
    total_faces = 0

    # Collect all image paths
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",".webp")
    image_files = [f for f in os.listdir(hq_folder) if f.lower().endswith(exts)]

    print(f"🔍 Scanning {len(image_files)} images for faces...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(hq_folder, img_name)

        try:
            detections = RetinaFace.detect_faces(img_path)
            faces = []
            if isinstance(detections, dict):
                for face_info in detections.values():
                    x1, y1, x2, y2 = face_info["facial_area"]
                    
                    w = x2 - x1
                    h = y2 - y1

                    total_width += w
                    total_height += h
                    total_faces += 1
                    
                    faces.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    })

            results[img_name] = faces

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")
            results[img_name] = []
            
    if total_faces > 0:
        avg_w = total_width / total_faces
        avg_h = total_height / total_faces
        avg_ratio = avg_w / avg_h
        print(f"\n📊 Detected {total_faces} faces total")
        print(f"➡️ Average width:  {avg_w:.2f} px")
        print(f"➡️ Average height: {avg_h:.2f} px")
        print(f"➡️ Width/Height ratio: {avg_ratio:.3f}")
    else:
        print("⚠️ No faces detected in the folder.")
        
    # Save results as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Face positions saved to: {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect faces and save bounding boxes as JSON.")
    parser.add_argument("--hq_folder", required=True, help="Path to HQ image folder.")
    args = parser.parse_args()

    find_faces_in_folder(args.hq_folder)