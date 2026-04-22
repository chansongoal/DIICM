import os 
import re
import numpy as np 


def read_file_lines(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def extract_yuv_psnr_from_file(file_path):
    """
    Extracts Y-PSNR, U-PSNR, V-PSNR, YUV-PSNR from the summary table.
    Returns a list: [Y, U, V, YUV]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty lines and header line
            if not line or "Total Frames" in line:
                continue

            # Data lines start with frame index (integer)
            if not re.match(r'^\d+', line):
                continue

            # Extract all floating-point numbers
            numbers = re.findall(r'[-+]?\d*\.\d+', line)

            # The last four floats are PSNR values
            if len(numbers) >= 4:
                y, u, v, yuv = map(float, numbers[-4:])
                return [y, u, v, yuv]

        return None

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
    PSNRs = extract_psnr_from_file(log_name)
    return bpp, PSNRs

def get_coding_info_vtm(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    image_names = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_names.txt")
    image_widths = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_widths.txt")
    image_heights = read_file_lines("/ghome/gaocs/DIICM/coding/VTM/image_heights.txt")
    bpp_all = []; Y_PSNR_all = []; U_PSNR_all = []; V_PSNR_all = []; YUV_PSNR_all = []
    for idx, image_name in enumerate(image_names):
        bpp, PSNRs = get_coding_info_single_image(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, image_name, int(image_widths[idx]), int(image_heights[idx]))
        bpp_all.append(bpp)
        Y_PSNR_all.append(PSNRs[0]); U_PSNR_all.append(PSNRs[1]); V_PSNR_all.append(PSNRs[2]); YUV_PSNR_all.append(PSNRs[3])
    assert len(bpp_all)==5000
    bpp_avg = np.mean(bpp_all)
    Y_PSNR_avg = np.mean(Y_PSNR_all); U_PSNR_avg = np.mean(U_PSNR_all); V_PSNR_avg = np.mean(V_PSNR_all); YUV_PSNR_avg = np.mean(YUV_PSNR_all)
    return bpp_avg, Y_PSNR_avg, U_PSNR_avg, V_PSNR_avg, YUV_PSNR_avg

def extract_bpp_from_file(file_path):
    """
    Extracts the numerical value before 'bits' in lines starting with 'POC' from the given text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    
        avg_bpp = float(lines[-1].split()[-1])
        return avg_bpp

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

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

def get_bpp(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    if processing_config == 'transformed_compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}/bpp_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}.txt"
        if mask_type == 'label':
            log_name = f"{data_root}/{processing_config}/{arch}/bpp/{mask_type}/bpp_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}.txt"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/bpp/bpp_{processing_config}_{quality}_{quality}.txt"
    bpp = extract_bpp_from_file(log_name)
    return bpp

def get_psnr(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    if processing_config == 'transformed_compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/psnr/{mask_type}/psnr_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}.txt"
        if mask_type == 'label':
            log_name = f"{data_root}/{processing_config}/{arch}/psnr/{mask_type}/psnr_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}.txt"
    elif processing_config == 'compressed':
        log_name = f"{data_root}/{processing_config}/{arch}/psnr/psnr_{processing_config}_{quality}_{quality}.txt"
    psnr_fore, psnr_back, psnr_overall = extract_psnr_from_file(log_name)
    # print(log_name)
    return psnr_fore, psnr_back, psnr_overall

def get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality):
    bpp = get_bpp(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
    psnr_fore, psnr_back, psnr_overall = get_psnr(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
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
    alpha = 0.5

    # network_config = 'Keypoints_Res50_FPN'
    # network_config = 'Mask_Res50_C4'
    # network_config = 'Faster_Res50_C4'

    quality_all = [51, 47, 42, 37, 32, 27]
    # quality_all = [27]
    for quality in quality_all:
        bpp, psnr_fore, psnr_back, psnr_overall = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        print(f"{bpp:.4f}, {psnr_fore:.4f}, {psnr_back:.4f}, {psnr_overall:.4f}")

        
        # mAP = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, network_config)

        # bpp, Y_PSNR, U_PSNR, V_PSNR, YUV_PSNR = get_coding_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality)
        # mAP_OD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Faster_Res50_C4')
        # mAP_IS = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Mask_Res50_C4')
        # mAP_KD = get_detectron2_info(data_root, processing_config, arch, mask_type, mask_network, alpha, quality, 'Keypoints_Res50_FPN')
        # print(f"{bpp:.4f}, {mAP_OD:.4f}, {mAP_IS:.4f}, {mAP_KD:.4f}")


if __name__ == '__main__':
    main()