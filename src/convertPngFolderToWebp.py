import os
from PIL import Image

def convert_pngs_to_webp(folder_path):
    """
    Converts all PNG images in the given folder to WEBP.
    Keeps same base filename, deletes PNG after successful save.
    """
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".png"):
            continue

        png_path = os.path.join(folder_path, file_name)
        webp_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + ".webp")

        try:
            with Image.open(png_path) as img:
                img.save(webp_path, "WEBP", quality=95)

            # Verify file saved
            if os.path.exists(webp_path) and os.path.getsize(webp_path) > 0:
                os.remove(png_path)
                print(f"✅ Converted and removed: {file_name}")
            else:
                print(f"⚠️ Failed to verify: {file_name}")

        except Exception as e:
            print(f"❌ Error converting {file_name}: {e}")

if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    if os.path.isdir(folder):
        convert_pngs_to_webp(folder)
    else:
        print("❌ Invalid folder path.")