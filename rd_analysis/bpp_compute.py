import os 
from PIL import Image
import numpy as np 
import re
import json


def bpp_jpeg():
    quality_all = ['1', '5', '10', '20', '25', '30', '35', '40', '45', '50', '60', '70', '75', '80']
    root_path = '/data/gaocs/Understanding_Detection/compressed/'
    for quality in quality_all:
        img_path = root_path + quality + '/'
        bpp_filename = '/data/gaocs/Understanding_Detection/bitrate/bpp_compressed_' + quality + '_' + quality + '.txt'
        file = open(bpp_filename, 'w')

        img_files = os.listdir(img_path)
        img_files.sort()
        bpp_all = []
        for idx, img_file in enumerate(img_files):
            img_name = img_path + img_file
            img = Image.open(img_name)
            width, height = img.size[0], img.size[1]
            num_pixles = width * height
            img_size = os.path.getsize(img_name)
            bpp = img_size * 8 / num_pixles 
            bpp_all.append(bpp)
            # print(img_file, bpp)
            file.write(img_file + ' ' + str(img_size) + ' ' + str(width) + ' ' + str(height)  + ' ' + str(bpp))
            file.write('\n')
            img.close()

        bpp_avg = np.sum(bpp_all) / len(bpp_all)
        # print(bpp_avg)
        file.write(str(bpp_avg))
        file.close()

def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def extract_vtm_bits_from_file(file_path):
    """
    Extracts the numerical value before 'bits' in lines starting with 'POC' from the given text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        pattern = r'^POC.*?([0-9\.]+)\s+bits'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                values = float(match.group(1))
        
        return values
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def extract_cheng2020_bpp_from_json(json_path: str):
    """
    Read bpp from a JSON file.

    Expected JSON structure:
    {
      "results": {
        "bpp": <float>
      }
    }

    Returns:
        bpp (float)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "results" not in data:
        raise KeyError(f"'results' not found in {json_path}")

    results = data["results"]

    if "bpp" not in results:
        raise KeyError(f"'bpp' not found in results of {json_path}")

    bpp = float(results["bpp"])

    return bpp

def bpp_vtm():
    data_root = "/gdata1/gaocs/Data_DIICM"
    mask_type = "inferred"
    mask_network = "MaskRCNN_Res101_FPN_0.5"
    mask_path = "/gdata/gaocs/dataset/COCO/minVal2014_GT_mask/"
    arch = "vtm_anchor"
    processing_config = "compressed"
    alpha = 0.5

    quality_all = [27, 32, 37, 42, 47, 51]
    # quality_all = [27]

    for quality in quality_all:
        if processing_config == "transformed_compressed":
            if mask_type == "inferred":
                if arch == "vtm_anchor":
                    log_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/encoding_log"
                elif arch == "cheng2020_anchor":
                    log_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/json"
                else:
                    raise ValueError(f"Unknown arch: {arch}")

                bpp_path = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}"
                bpp_name = f"{bpp_path}/bpp_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}.txt"

            elif mask_type == "label":
                if arch == "vtm_anchor":
                    log_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/qp{quality}/encoding_log"
                elif arch == "cheng2020_anchor":
                    log_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/quality{quality}/json"
                else:
                    raise ValueError(f"Unknown arch: {arch}")

                bpp_path = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}"
                bpp_name = f"{bpp_path}/bpp_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}.txt"
            else:
                raise ValueError(f"Unknown mask_type: {mask_type}")

        elif processing_config == "compressed":
            bpp_path = f"{data_root}/{processing_config}/{arch}/bpp"
            if arch == "vtm_anchor":
                bpp_name = f"{bpp_path}/bpp_{processing_config}_{quality}_{quality}.txt"
                log_path = f"{data_root}/{processing_config}/{arch}/qp{quality}/encoding_log"
            elif arch == "cheng2020_anchor":
                bpp_name = f"{bpp_path}/bpp_{processing_config}_quality{quality}.txt"
                log_path = f"{data_root}/{processing_config}/{arch}/quality{quality}/json"
        else:
            raise ValueError(f"Unknown processing_config: {processing_config}")

        os.makedirs(bpp_path, exist_ok=True)
        file = open(bpp_name, 'w')
        # print(bpp_name)

        bpp_all = []
        image_names = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_names.txt")
        image_widths = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_widths.txt")
        image_heights = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_heights.txt")

        for idx, log_file in enumerate(image_names):
            log_name = f"{log_path}/{log_file}.txt"
            # print(log_name)
            bits = extract_vtm_bits_from_file(log_name)
            bpp = bits / float(image_heights[idx]) / float(image_widths[idx])
            bpp_all.append(bpp)
            # print(img_file, bpp)
            file.write(f"{log_file}.jpg {bpp}\n")

        bpp_avg = np.mean(bpp_all)
        file.write(f"Average bpp: {bpp_avg}")
        file.close()

def bpp_cheng2020():
    data_root = "/gdata1/gaocs/Data_DIICM"
    mask_type = "inferred"
    mask_network = "MaskRCNN_Res101_FPN_0.5"
    mask_path = "/gdata/gaocs/dataset/COCO/minVal2014_GT_mask/"
    arch = "cheng2020_anchor"
    processing_config = "transformed_compressed"
    alpha = 0.5

    quality_all = [1, 2, 3, 4, 5, 6]
    # quality_all = [27]

    for quality in quality_all:
        if processing_config == "transformed_compressed":
            if mask_type == "inferred":
                json_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/json"
                bpp_path = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}"
                bpp_name = f"{bpp_path}/bpp_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}.txt"
            elif mask_type == "label":
                json_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/quality{quality}/json"
                bpp_path = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}"
                bpp_name = f"{bpp_path}/bpp_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}.txt"
            else:
                raise ValueError(f"Unknown mask_type: {mask_type}")

        elif processing_config == "compressed":
            json_path = f"{data_root}/{processing_config}/{arch}/quality{quality}/json"
            bpp_path = f"{data_root}/{processing_config}/{arch}/bpp"
            bpp_name = f"{bpp_path}/bpp_{processing_config}_quality{quality}.txt"
        else:
            raise ValueError(f"Unknown processing_config: {processing_config}")

        os.makedirs(bpp_path, exist_ok=True)
        bpp_file = open(bpp_name, 'w')

        bpp_all = []
        image_names = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_names.txt")

        model_suffix = ['dad2ebff', 'a29008eb', 'e49be189', '98b0b468', '23852949', '4c052b1a']
        for idx, log_file in enumerate(image_names):
            json_name = f"{json_path}/{log_file}-cheng2020-anchor-{quality}-{model_suffix[quality-1]}-ans.json"
            # print(json_name)
            bpp = extract_cheng2020_bpp_from_json(json_name)
            bpp_all.append(bpp)
            # print(img_file, bpp)
            bpp_file.write(f"{log_file}.jpg {bpp}\n")

        bpp_avg = np.mean(bpp_all)
        bpp_file.write(f"Average bpp: {bpp_avg}")
        bpp_file.close()

if __name__ == '__main__':
    # bpp_jpeg()
    # bpp_vtm()
    bpp_cheng2020()