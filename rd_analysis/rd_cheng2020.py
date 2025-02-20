import os 
import re
import numpy as np 
import matplotlib.pyplot as plt


def extract_bpp_from_file(file_path):
    """
    Extracts the numerical value inside the square brackets for the "bpp" key in the given text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Updated regex pattern to match "bpp": [ value ]
        pattern = r'"bpp":\s*\[\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\]'

        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows matching across multiple lines
        
        if match:
            return float(match.group(1))  # Extract and convert to float
        
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    if processing_config == 'transformed_compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/compress_{mask_type}_{mask_network}_1.0_{alpha}_quality{quality}.txt"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/quality{quality}/compress_quality{quality}.txt"
    bpp = extract_bpp_from_file(log_name)
    return bpp


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
        log_path = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}"
        log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{mask_type}_{mask_network}_{network_config}_1.0_{alpha}_quality{quality}.txt"
    elif processing_config == 'compressed':
        log_path = f"{data_root}/mAP/{processing_config}/{arch}/quality{quality}"
        log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{network_config}_quality{quality}.txt"
    # print(log_name)
    mAP = extract_mAP_from_file(log_name)

    return mAP


def main():
    data_root = "/gdata1/gaocs/Data_DIICM"
    mask_type = 'inferred'
    mask_network = 'MaskRCNN_Res101_FPN_0.5'
    alpha = 0.2
    arch = 'cheng2020_anchor'
    processing_config = 'compressed'

    # network_config = 'Keypoints_Res50_FPN'

    quality_all = [1, 2, 3, 4, 5, 6]
    for quality in quality_all:
        # bpp = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        # mAP = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, network_config)
        # print(f"{bpp:.2f}, {mAP:.4f}")

        bpp = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        mAP_OD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Faster_Res50_C4')
        mAP_IS = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Mask_Res50_C4')
        mAP_KD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Keypoints_Res50_FPN')
        print(f"{bpp:.4f}, {mAP_OD:.4f}, {mAP_IS:.4f}, {mAP_KD:.4f}")


if __name__ == '__main__':
    main()