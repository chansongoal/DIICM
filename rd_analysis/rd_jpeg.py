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



for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
        inferenced_img = Image.open(inv_mask_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask_arr
        temp[:,:,1] = inv_mask_arr
        temp[:,:,2] = inv_mask_arr
        inv_mask_arr = temp
        mask_arr = (-inv_mask_arr + 1).astype(np.uint8)

        rec_img = Image.open(rec_path + img_name)
        rec_arr = np.asarray(rec_img).astype(np.float32)

        fore_arr = rec_arr * mask_arr
        back_arr = rec_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * beta + (1-beta)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr
        img_arr_scaled = fore_arr + back_arr_sacled

        # normalization 1: only normalize the values that greater than 255 or less than 0
        # idx1, idx2 = back_arr_sacled>255, back_arr_sacled<0
        # idx_all = idx1 | idx2
        # idx_all = idx_all.astype(int)
        # inv_idx_all = -idx_all + 1
        # img_arr_scaled_ = (img_arr_scaled - np.min(back_arr_sacled))/(np.max(back_arr_sacled)-np.min(back_arr_sacled)) * 255
        # img_arr_scaled = img_arr_scaled*inv_idx_all + img_arr_scaled_ * idx_all

        # normalization 2
        num = img_arr_scaled.shape[0] * img_arr_scaled.shape[1] * img_arr_scaled.shape[2]
        up_per, low_per = np.count_nonzero(img_arr_scaled>255)/num, np.count_nonzero(img_arr_scaled<0)/num
        maximum, minimum, avg, up_avg, low_avg = np.max(img_arr_scaled), np.min(img_arr_scaled),  np.mean(img_arr_scaled), np.mean(img_arr_scaled[img_arr_scaled>255]), np.mean(img_arr_scaled[img_arr_scaled<0])
        write_line = img_name + ' ' +str(up_per)+' ' + str(low_per)+' ' + str(maximum)+' ' + str(minimum)+' ' +str(avg)+' ' +str(up_avg)+' ' +str(low_avg)+' \n'
        inv_statistics_file.write(write_line)
        up_per_all.append(up_per)
        low_per_all.append(low_per)
        maximum_all.append(maximum) 
        minimum_all.append(minimum)
        avg_all.append(avg)
        up_avg_all.append(up_avg)
        low_avg_all.append(low_avg)
        img_arr_scaled[img_arr_scaled>255] = 255
        img_arr_scaled[img_arr_scaled<0] = 0

        # # img_arr_scaled = fore_arr + back_arr_sacled
        # img_arr_scaled = img_arr_scaled.astype(np.uint8)
        # img_scaled = Image.fromarray(img_arr_scaled)
        # img_scaled.save(rec_path+img_name[:-4]+'.png')
    write_line = 'average: ' + ' ' +str(np.mean(up_per_all))+' ' + str(np.mean(low_per_all))+' ' + str(np.mean(maximum_all))+' ' + str(np.mean(minimum_all))+' ' +str(np.mean(avg_all))+' ' +str(np.mean(up_avg_all))+' ' +str(np.mean(low_avg_all))+' \n'
    inv_statistics_file.write(write_line)