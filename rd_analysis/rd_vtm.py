import os 
import re
import numpy as np 


def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def extract_bpp_from_file(file_path):
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

def get_coding_info_single_image(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, image_name, width, height):
    if processing_config == 'transformed_compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/encoding_log/{image_name}.txt"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/qp{quality}/encoding_log/{image_name}.txt"
    bits = extract_bpp_from_file(log_name)
    bpp = bits / width / height
    return bpp

def get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    image_names = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_names.txt")
    image_widths = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_widths.txt")
    image_heights = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_heights.txt")
    bpp_all = []
    for idx, image_name in enumerate(image_names):
        bpp = get_coding_info_single_image(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, image_name, int(image_widths[idx]), int(image_heights[idx]))
        bpp_all.append(bpp)

    assert len(bpp_all)==5000
    bpp_avg = np.mean(bpp_all)
    return bpp_avg


def extract_mAP_from_file(file_path):
    """
    Extracts the first numerical value after the last colon in the last line of the given text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        if not lines:
            return None
        
        # Get the last line
        last_line = lines[-1].strip()
        
        # Extract the portion after the last colon
        if ':' in last_line:
            values_part = last_line.split(':')[-1].strip()
            
            # Extract values separated by commas
            values = values_part.split(',')
            
            if values:
                return float(values[0].strip())
        
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, network_config):
    dataset_name_prefix = 'coco_minVal2014_5000'
    if processing_config == 'transformed_compressed':
        log_path = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}"
        log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{mask_type}_{mask_network}_{network_config}_1.0_{alpha}_quality{quality}.txt"
    elif processing_config == 'compressed':
        log_path = f"{data_root}/mAP/{processing_config}/{arch}/qp{quality}"
        log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{network_config}_quality{quality}.txt"
    # print(log_name)
    mAP = extract_mAP_from_file(log_name)

    return mAP

def main():
    data_root = "/gdata1/gaocs/Data_DIICM"
    mask_type = 'inferred'
    mask_network = 'MaskRCNN_Res101_FPN_0.5'
    arch = 'vtm_anchor'
    processing_config = 'transformed_compressed'
    alpha = 0.2

    # network_config = 'Keypoints_Res50_FPN'
    # network_config = 'Mask_Res50_C4'
    # network_config = 'Faster_Res50_C4'

    quality_all = [51, 47, 42, 37, 32, 27]
    for quality in quality_all:
        # bpp = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        # mAP = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, network_config)

        bpp = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        mAP_OD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Faster_Res50_C4')
        mAP_IS = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Mask_Res50_C4')
        mAP_KD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Keypoints_Res50_FPN')
        print(f"{bpp:.4f}, {mAP_OD:.4f}, {mAP_IS:.4f}, {mAP_KD:.4f}")


if __name__ == '__main__':
    main()