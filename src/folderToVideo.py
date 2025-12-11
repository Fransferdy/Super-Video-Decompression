import cv2
import os
import argparse
from natsort import natsorted  # optional, sorts filenames naturally

def img_to_mp4(input_folder, output_file, fps=30, resize=None, fileType='.webp'):
    """
    Converts all .img images in a folder to an MP4 video.
    
    Args:
        input_folder (str): path to folder containing .img images
        output_file (str): path to save output .mp4 file
        fps (int): frames per second
        resize (tuple or None): (width, height) to resize frames, or None to keep original
    """
    # Get sorted list of images
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(fileType)]
    files = natsorted(files)
    print (files)
    if not files:
        print("No .img files found in the folder.")
        return

    # Read first image to get frame size
    first_img = cv2.imread(os.path.join(input_folder, files[0]), cv2.IMREAD_UNCHANGED)
    if resize:
        frame_size = resize
        first_img = cv2.resize(first_img, frame_size)
    else:
        h, w = first_img.shape[:2]
        frame_size = (w, h)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # Write frames
    for f in files:
        img_path = os.path.join(input_folder, f)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {f}, skipping.")
            continue
        if resize:
            img = cv2.resize(img, resize)
        # Convert grayscale or RGBA to BGR
        if img.shape[-1] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.write(img)

    out.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .img images to MP4 video.")
    parser.add_argument("input_folder", type=str, help="Folder containing .img images")
    parser.add_argument("output_file", type=str, help="Output mp4 file path")
    parser.add_argument("--file_type", type=str, default=30, help="File Type (such as .png, .webp)")
    parser.add_argument("--fps", type=float, default=24, help="Frames per second")
    parser.add_argument("--width", type=int, help="Resize width")
    parser.add_argument("--height", type=int, help="Resize height")

    args = parser.parse_args()
    resize = (args.width, args.height) if args.width and args.height else None
    if args.file_type != '.webp' and args.file_type != '.png':
        args.file_type = '.webp'
    img_to_mp4(args.input_folder, args.output_file, fps=args.fps, resize=resize, fileType=args.file_type)