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
        
        values = float(lines[-1])
        return values
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_bpp(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    if processing_config == 'transformed_compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/bpp/bpp_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}.txt"
        if mask_type == 'label':
            log_name = f"{data_root}/{processing_config}/{arch}/bpp/bpp_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}.txt"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/bpp/bpp_{processing_config}_{quality}_{quality}.txt"
    bpp = extract_bpp_from_file(log_name)
    return bpp


def extract_psnr_from_file(file_path):
    """
    Extracts the numerical value before 'bits' in lines starting with 'POC' from the given text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        psnr_fore, psnr_back, psnr_overall = float(lines[-1].split()[-3]), float(lines[-1].split()[-2]), float(lines[-1].split()[-1])
        return psnr_fore, psnr_back, psnr_overall
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_psnr(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, psnr_config):
    if processing_config == 'transformed_compressed':
        if psnr_config == 'psnr':
            log_name = f"{data_root}/{processing_config}/{arch}/{psnr_config}/inferred_label/psnr_{processing_config}_inferenced_label_{mask_network}_1.0_{alpha}_{quality}_{quality}.log"
        if psnr_config == 'psnr_inv':
            if alpha == 0.5: beta = 2.0
            log_name = f"{data_root}/{processing_config}/{arch}/{psnr_config}/inferred_label/psnr_{processing_config}_inv_inferenced_label_{mask_network}_1.0_{beta}_{quality}_{quality}.log"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/psnr/psnr_{processing_config}_{quality}_{quality}.log"
    psnr_fore, psnr_back, psnr_overall = extract_psnr_from_file(log_name)
    return psnr_fore, psnr_back, psnr_overall


def get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, psnr_config):
    bpp = get_bpp(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
    psnr_fore, psnr_back, psnr_overall = get_psnr(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, psnr_config)
    return bpp, psnr_fore, psnr_back, psnr_overall


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
        log_path = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}"
        log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{mask_type}_{mask_network}_{network_config}_1.0_{alpha}_{quality}_{quality}.log"
        if mask_type == 'label':
            log_path = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}"
            log_name = f"{log_path}/{dataset_name_prefix}_{processing_config}_{mask_type}_{network_config}_1.0_{alpha}_{quality}_{quality}.log"
    elif processing_config == 'compressed':
        log_path = f"{data_root}/mAP/{processing_config}/{arch}"
        log_name = f"{log_path}/{dataset_name_prefix}_jpeg_{processing_config}_label_{network_config}_{quality}_{quality}.log"
    # print(log_name)
    mAP = extract_mAP_from_file(log_name)

    return mAP

def main():
    data_root = "/gdata1/gaocs/Data_DIICM"
    mask_type = 'label'
    mask_network = 'MaskRCNN_Res101_FPN_0.5'
    arch = 'jpeg_anchor'
    processing_config = 'compressed'
    psnr_config = 'psnr'
    alpha = 0.5


    quality_all = [1, 10, 20, 30, 40, 50, 60, 70]
    for quality in quality_all:
        bpp, psnr_fore, psnr_back, psnr_overall = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, psnr_config)
        # print(f"{bpp:.4f}, {psnr_fore:.4f}, {psnr_back:.4f}, {psnr_overall:.4f}")
        mAP_IS_2 = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, 0.2, quality, 'Mask_Res50_C4')
        mAP_IS_5 = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, 0.5, quality, 'Mask_Res50_C4')
        mAP_IS_8 = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, 0.8, quality, 'Mask_Res50_C4')
        print(f"{bpp:.4f}, {mAP_IS_2:.4f}, {mAP_IS_5:.4f}, {mAP_IS_8:.4f}")

        # mAP_OD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Faster_Res50_C4')
        # mAP_IS = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Mask_Res50_C4')
        # mAP_KD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Keypoints_Res50_FPN')
        # # print(f"{mAP:.4f}")
        # print(f"{bpp:.4f}, {mAP_OD:.4f}, {mAP_IS:.4f}, {mAP_KD:.4f}")


if __name__ == '__main__':
    main()