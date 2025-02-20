import os
from PIL import Image

def get_image_info(folder_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Sort files by name
    image_files.sort()
    
    # Define file paths
    names_file = os.path.join(output_folder, "image_names.txt")
    widths_file = os.path.join(output_folder, "image_widths.txt")
    heights_file = os.path.join(output_folder, "image_heights.txt")
    
    # Process images
    with open(names_file, 'w') as f_names, open(widths_file, 'w') as f_widths, open(heights_file, 'w') as f_heights:
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as image:
                    width, height = image.size
                    
                    # Write to files
                    f_names.write(f"{img_name[:-4]}\n")
                    f_widths.write(f"{width}\n")
                    f_heights.write(f"{height}\n")
            except Exception as e:
                print(f"Warning: Cannot read {img_name}, skipping... Error: {e}")
    
    print("Processing completed. Output files saved in:", output_folder)

if __name__ == "__main__":
    folder_path = "/gdata/gaocs/dataset/COCO/minVal2014"  # Modify to your image folder path
    output_folder = "/ghome/gaocs/DIICM/coding/VTM"  # Modify to the folder where output will be saved
    get_image_info(folder_path, output_folder)
