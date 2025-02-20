import os
import subprocess

def convert_png_to_yuv(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        # Compute the corresponding output directory
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        print(target_dir)

        for file in files:
            if file.lower().endswith(".png"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(target_dir, os.path.splitext(file)[0] + ".yuv")
                
                # Use FFmpeg to convert PNG to YUV444
                cmd = [
                    "ffmpeg", "-y", "-i", input_file,
                    "-pix_fmt", "yuv444p", output_file
                ]
                # print(cmd)
                
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # print(f"Converted: {input_file} -> {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {input_file}: {e}")


def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def convert_yuv_to_png(image_names, widths, heights, input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for name, width, height in zip(image_names, widths, heights):
        input_file = os.path.join(input_dir, name + ".yuv")
        output_file = os.path.join(output_dir, name + ".png")
        
        # Use FFmpeg to convert YUV444 to PNG with specific width and height
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{width}x{height}", "-i", input_file, output_file
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # print(f"Converted: {input_file} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {input_file}: {e}")

def process_all_rec_yuv_folders(root_directory):
    for dirpath, dirnames, _ in os.walk(root_directory):
        if "rec_yuv" in dirnames:
            input_dir = os.path.join(dirpath, "rec_yuv")
            output_dir = os.path.join(dirpath, "rec_png")
            print(output_dir)
            
            image_names = read_file_lines("image_names.txt")
            widths = read_file_lines("image_widths.txt")
            heights = read_file_lines("image_heights.txt")
            
            convert_yuv_to_png(image_names, widths, heights, input_dir, output_dir)

if __name__ == "__main__":
    # # convert png to yuv444
    # root_path = fr"Y:\data\gaochangsheng\DIICM"
    # # input_root = fr"{root_path}\transformed\inferred\MaskRCNN_Res101_FPN_0.5\1.0_0.8"
    # # output_root = fr"{root_path}\transformed_yuv\inferred\MaskRCNN_Res101_FPN_0.5\1.0_0.8"
    # input_root = fr"{root_path}\transformed\label"
    # output_root = fr"{root_path}\transformed_yuv\label"
    # # input_root = fr"Y:\data\gaochangsheng\DIICM\minVal2014"
    # # output_root = fr"Y:\data\gaochangsheng\DIICM\minVal2014_yuv"
    # convert_png_to_yuv(input_root, output_root)

    # convert yuv444 to png
    # root_path = fr"Y:\data\gaochangsheng\DIICM\compressed\vtm_anchor\qp51"
    root_path = fr"Y:\data\gaochangsheng\DIICM\transformed_compressed\vtm_anchor\inferred\MaskRCNN_Res101_FPN_0.5\1.0_0.8\qp42"  # Change this to your actual root directory
    process_all_rec_yuv_folders(root_path)
