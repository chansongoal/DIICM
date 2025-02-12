import os
# from sys import orig_argv

from pycocotools.coco import COCO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np 
import pickle
import math

def psnr_compressed(quality):
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    rec_path = '/data/gaocs/Understanding_Detection/compressed/' + quality.split('_')[0] + '/'
    mask_path = '/data/gaocs/Understanding_Detection/label_mask/'
    psnr_path = '/data/gaocs/Understanding_Detection/psnr/compressed/'
    psnr_filename = 'psnr_compressed_' + quality + '.log'
    if not os.path.exists(psnr_path):
        os.makedirs(psnr_path)
    psnr_file = open(psnr_path + psnr_filename, 'w')

    fore_mse_all = []
    back_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []

    overall_mse_all = []
    overall_psnr_all = []

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        img_name = img_files[idx]
        # get ground-truth mask
        mask_img = Image.open(mask_path + img_name[:-4]+'.png')
        mask_arr = np.asarray(mask_img) 
        inv_mask_arr = -mask_arr + 1

        # get org and rec images
        org_img = Image.open(org_path + img_name)
        rec_img = Image.open(rec_path + img_name)
        org_arr = np.asarray(org_img, dtype=np.float32)
        rec_arr = np.asarray(rec_img, dtype=np.float32)

        # process images with ground-truth mask
        mask_fore_org = org_arr * mask_arr
        mask_back_org = org_arr * inv_mask_arr
        mask_fore_rec = rec_arr * mask_arr
        mask_back_rec = rec_arr * inv_mask_arr

        # compute psnr with ground-truth mask
        PIXEL_MAX = 255.0
        num_pixel_fore = np.sum(mask_arr)
        fore_mse = np.sum((mask_fore_org - mask_fore_rec)**2) / num_pixel_fore
        fore_psnr = 10 * math.log10(PIXEL_MAX**2 / fore_mse)
        num_pixel_back = np.sum(inv_mask_arr)
        back_mse = np.sum((mask_back_org - mask_back_rec)**2) / num_pixel_back
        back_psnr = 10 * math.log10(PIXEL_MAX**2 / back_mse)

        # compute overall psnr
        overall_mse = np.mean((rec_arr - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)

        # write psnr with ground-truth mask
        if (fore_mse < 256*256 and back_mse < 256*256 and fore_mse > 0 and back_mse > 0):
            fore_mse_all.append(fore_mse)
            back_mse_all.append(back_mse)
            overall_mse_all.append(overall_mse)
            fore_psnr_all.append(fore_psnr)
            back_psnr_all.append(back_psnr)
            overall_psnr_all.append(overall_psnr)
        write_line = img_name + ' ' +str(fore_mse)+' ' + str(back_mse)+' ' + str(overall_mse)+' ' + str(fore_psnr)+' ' +str(back_psnr)+' ' +str(overall_psnr)+' \n'
        psnr_file.write(write_line)
    
    write_line = 'average: ' +str(np.mean(fore_mse_all))+' ' + str(np.mean(back_mse_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_all))+' ' +str(np.mean(back_psnr_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnr_file.write(write_line)
    psnr_file.close()


def psnr_label(scale, quality):
    transform_config = 'transformed_compressed'
    label_config = 'label'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    rec_path = '/data/gaocs/Understanding_Detection/'+transform_config+'/'+label_config+'/' + scale + '/' + quality + '/'
    mask_path_label = '/data/gaocs/Understanding_Detection/label_mask/'

    # iou_threshold = '0.5'
    # mask_path_inferenced = '/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'

    psnr_path = '/data/gaocs/Understanding_Detection/psnr/'+transform_config+'/label_label/' #+ scale + '/' + quality + '/'
    psnr_filename = 'psnr_' + transform_config+ '_label_label_' + scale + '_' + quality + '.log'
    if not os.path.exists(psnr_path):
        os.makedirs(psnr_path)
    psnr_file = open(psnr_path + psnr_filename, 'w')

    fore_mse_label_all = []
    back_mse_label_all = []
    fore_psnr_label_all = []
    back_psnr_label_all = []

    # fore_mse_inferenced_all = []
    # back_mse_inferenced_all = []    
    # fore_psnr_inferenced_all = []
    # back_psnr_inferenced_all = []

    overall_mse_all = []
    overall_psnr_all = []

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        img_name = img_files[idx]
        # get ground-truth mask
        mask_img_label = Image.open(mask_path_label + img_name[:-4]+'.png')
        mask_arr_label = np.asarray(mask_img_label) 
        inv_mask_arr_label = -mask_arr_label + 1
        # get inferenced mask
        # inferenced_img = Image.open(mask_path_inferenced + img_name[:-4]+'.png')
        # inferenced_arr = np.asarray(inferenced_img)
        # inferenced_arr = np.sum(inferenced_arr, axis=2)
        # inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        # temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        # temp[:,:,0] = inv_mask_arr
        # temp[:,:,1] = inv_mask_arr
        # temp[:,:,2] = inv_mask_arr
        # inv_mask_arr_label = temp
        # mask_img_label = (-inv_mask_arr_label + 1).astype(np.uint8)

        # get org and rec images
        org_img = Image.open(org_path + img_name)
        rec_img = Image.open(rec_path + img_name)
        org_arr = np.asarray(org_img, dtype=np.float32)
        rec_arr = np.asarray(rec_img, dtype=np.float32)

        # process images with ground-truth mask
        mask_fore_org_label = org_arr * mask_arr_label
        mask_back_org_label = org_arr * inv_mask_arr_label
        mask_fore_rec_label = rec_arr * mask_arr_label
        mask_back_rec_label = rec_arr * inv_mask_arr_label

        # process images with inferenced mask
        # mask_fore_org_inferenced = org_arr * mask_arr_inferenced
        # mask_back_org_inferenced = org_arr * inv_mask_arr_inferenced
        # mask_fore_rec_inferenced = rec_arr * mask_arr_inferenced
        # mask_back_rec_inferenced = rec_arr * inv_mask_arr_inferenced

        # compute psnr with ground-truth mask
        PIXEL_MAX = 255.0
        num_pixel_fore_label = np.sum(mask_arr_label)
        fore_mse_label = np.sum((mask_fore_org_label - mask_fore_rec_label)**2) / num_pixel_fore_label
        fore_psnr_label = 10 * math.log10(PIXEL_MAX**2 / fore_mse_label)
        num_pixel_back_label = np.sum(inv_mask_arr_label)
        back_mse_label = np.sum((mask_back_org_label - mask_back_rec_label)**2) / num_pixel_back_label
        back_psnr_label = 10 * math.log10(PIXEL_MAX**2 / back_mse_label)

        # compute psnr with inferenced mask
        # num_pixel_fore_inferenced = np.sum(mask_arr_inferenced)
        # fore_mse_inferenced = np.sum((mask_fore_org_inferenced - mask_fore_rec_inferenced)**2) / num_pixel_fore_inferenced
        # fore_psnr_inferenced = 10 * math.log10(PIXEL_MAX**2 / fore_mse_inferenced)
        # num_pixel_back_inferenced = np.sum(inv_mask_arr_inferenced)
        # back_mse_inferenced = np.sum((mask_back_org_inferenced - mask_back_rec_inferenced)**2) / num_pixel_back_inferenced
        # back_psnr_inferenced = 10 * math.log10(PIXEL_MAX**2 / back_mse_inferenced)

        # compute overall psnr
        overall_mse = np.mean((rec_arr - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)

        # write psnr with ground-truth mask
        if (fore_mse_label < 256*256 and back_mse_label < 256*256 and fore_mse_label > 0 and back_mse_label > 0):
            fore_mse_label_all.append(fore_mse_label)
            back_mse_label_all.append(back_mse_label)
            overall_mse_all.append(overall_mse)
            fore_psnr_label_all.append(fore_psnr_label)
            back_psnr_label_all.append(back_psnr_label)
            overall_psnr_all.append(overall_psnr)
        write_line = img_name + ' ' +str(fore_mse_label)+' ' + str(back_mse_label)+' ' + str(overall_mse)+' ' + str(fore_psnr_label)+' ' +str(back_psnr_label)+' ' +str(overall_psnr)+' \n'
        psnr_file.write(write_line)

        # write psnr with inferenced mask
        # if (fore_mse_inferenced < 256*256 and back_mse_inferenced < 256*256 and fore_mse_inferenced > 0 and back_mse_inferenced > 0):
        #     fore_mse_inferenced_all.append(fore_mse_inferenced)
        #     back_mse_inferenced_all.append(back_mse_inferenced)
        #     overall_mse_all.append(overall_mse)
        #     fore_psnr_inferenced_all.append(fore_psnr_inferenced)
        #     back_psnr_inferenced_all.append(back_psnr_inferenced)
        #     overall_psnr_all.append(overall_psnr)
        # write_line = img_name + ' ' +str(fore_mse_inferenced)+' ' + str(back_mse_inferenced)+' ' + str(overall_mse)+' ' + str(fore_psnr_inferenced)+' ' +str(back_psnr_inferenced)+' ' +str(overall_psnr)+' \n'
        # psnr_file.write(write_line)
    
    write_line = 'average: ' +str(np.mean(fore_mse_label_all))+' ' + str(np.mean(back_mse_label_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_label_all))+' ' +str(np.mean(back_psnr_label_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnr_file.write(write_line)

    # write_line = 'average: ' +str(np.mean(fore_mse_inferenced_all))+' ' + str(np.mean(back_mse_inferenced_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_inferenced_all))+' ' +str(np.mean(back_psnr_inferenced_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    # psnr_file.write(write_line)

    psnr_file.close()


def psnr_inferenced(scale, quality):
    iou_threshold = '0.5'
    transform_config = 'transformed_compressed'
    label_config = 'inferenced_inferenced/MaskRCNN_Res101_FPN_0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    rec_path = '/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_0.5/'+'/' + scale + '/' + quality + '/'

    # mask_path = '/data/gaocs/Understanding_Detection/label_mask/'
    mask_path = '/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'

    psnr_path = '/data/gaocs/Understanding_Detection/psnr/transformed_compressed/inferenced_inferenced/'
    psnr_filename = 'psnr_' + transform_config+ '_' + 'inferenced_inferenced_MaskRCNN_Res101_FPN_0.5' + '_' + scale + '_' + quality + '.log'
    if not os.path.exists(psnr_path):
        os.makedirs(psnr_path)
    psnr_file = open(psnr_path + psnr_filename, 'w')

    fore_mse_all = []
    back_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []

    overall_mse_all = []
    overall_psnr_all = []

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        img_name = img_files[idx]
        # get ground-truth mask
        # mask_img = Image.open(mask_path + img_name[:-4]+'.png')
        # mask_arr = np.asarray(mask_img) 
        # inv_mask_arr = -mask_arr + 1
        # get inferenced mask
        inferenced_img = Image.open(mask_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask_arr
        temp[:,:,1] = inv_mask_arr
        temp[:,:,2] = inv_mask_arr
        inv_mask_arr = temp
        mask_arr = (-inv_mask_arr + 1).astype(np.uint8)

        # get org and rec images
        org_img = Image.open(org_path + img_name)
        rec_img = Image.open(rec_path + img_name)
        org_arr = np.asarray(org_img, dtype=np.float32)
        rec_arr = np.asarray(rec_img, dtype=np.float32)

        # process images with ground-truth mask
        mask_fore_org = org_arr * mask_arr
        mask_back_org = org_arr * inv_mask_arr
        mask_fore_rec = rec_arr * mask_arr
        mask_back_rec = rec_arr * inv_mask_arr

        # compute psnr with ground-truth mask
        PIXEL_MAX = 255.0
        num_pixel_fore = np.sum(mask_arr)
        fore_mse = np.sum((mask_fore_org - mask_fore_rec)**2) / num_pixel_fore
        fore_psnr = 10 * math.log10(PIXEL_MAX**2 / fore_mse)
        num_pixel_back = np.sum(inv_mask_arr)
        back_mse = np.sum((mask_back_org - mask_back_rec)**2) / num_pixel_back
        back_psnr = 10 * math.log10(PIXEL_MAX**2 / back_mse)

        # compute overall psnr
        overall_mse = np.mean((rec_arr - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)

        # write psnr with ground-truth mask
        if (fore_mse < 256*256 and back_mse < 256*256 and fore_mse > 0 and back_mse > 0):
            fore_mse_all.append(fore_mse)
            back_mse_all.append(back_mse)
            overall_mse_all.append(overall_mse)
            fore_psnr_all.append(fore_psnr)
            back_psnr_all.append(back_psnr)
            overall_psnr_all.append(overall_psnr)
        write_line = img_name + ' ' +str(fore_mse)+' ' + str(back_mse)+' ' + str(overall_mse)+' ' + str(fore_psnr)+' ' +str(back_psnr)+' ' +str(overall_psnr)+' \n'
        psnr_file.write(write_line)
    
    write_line = 'average: ' +str(np.mean(fore_mse_all))+' ' + str(np.mean(back_mse_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_all))+' ' +str(np.mean(back_psnr_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnr_file.write(write_line)
    psnr_file.close()


def psnr_inferenced_inv(scale, quality):
    iou_threshold = '0.5'
    transform_config = 'transformed_compressed_inv'
    label_config = 'inferenced_inferenced/MaskRCNN_Res101_FPN_0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    rec_path = '/data/gaocs/Understanding_Detection/transformed_compressed_inv/inferenced/MaskRCNN_Res101_FPN_0.5/' + scale + '/' + quality + '/'
    
    mask_path = '/data/gaocs/Understanding_Detection/label_mask/'
    # mask_path = '/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'

    # psnr_path = '/data/gaocs/Understanding_Detection/psnr/transformed_compressed_inv/inferenced_label/' #normalization 1: only normalize the values that greater than 255 or less than 0
    psnr_path = '/data/gaocs/Understanding_Detection/psnr/transformed_compressed_inv/inferenced_label_0_255/'   #normalization 2
    psnr_filename = 'psnr_' + transform_config+ '_' + 'inferenced_label_MaskRCNN_Res101_FPN_0.5' + '_' + scale + '_' + quality + '.log'
    if not os.path.exists(psnr_path):
        os.makedirs(psnr_path)
    psnr_file = open(psnr_path + psnr_filename, 'w')

    fore_mse_all = []
    back_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []

    overall_mse_all = []
    overall_psnr_all = []

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        img_name = img_files[idx]
        # get ground-truth mask
        mask_img = Image.open(mask_path + img_name[:-4]+'.png')
        mask_arr = np.asarray(mask_img) 
        inv_mask_arr = -mask_arr + 1
        # get inferenced mask
        # inferenced_img = Image.open(mask_path + img_name[:-4]+'.png')
        # inferenced_arr = np.asarray(inferenced_img)
        # inferenced_arr = np.sum(inferenced_arr, axis=2)
        # inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        # temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        # temp[:,:,0] = inv_mask_arr
        # temp[:,:,1] = inv_mask_arr
        # temp[:,:,2] = inv_mask_arr
        # inv_mask_arr = temp
        # mask_arr = (-inv_mask_arr + 1).astype(np.uint8)

        # get org and rec images
        org_img = Image.open(org_path + img_name)
        rec_img = Image.open(rec_path + img_name[:-4]+'.png')
        org_arr = np.asarray(org_img, dtype=np.float32)
        rec_arr = np.asarray(rec_img, dtype=np.float32)

        # process images with ground-truth mask
        mask_fore_org = org_arr * mask_arr
        mask_back_org = org_arr * inv_mask_arr
        mask_fore_rec = rec_arr * mask_arr
        mask_back_rec = rec_arr * inv_mask_arr

        # compute psnr with ground-truth mask
        PIXEL_MAX = 255.0
        num_pixel_fore = np.sum(mask_arr)
        fore_mse = np.sum((mask_fore_org - mask_fore_rec)**2) / num_pixel_fore
        fore_psnr = 10 * math.log10(PIXEL_MAX**2 / fore_mse)
        num_pixel_back = np.sum(inv_mask_arr)
        back_mse = np.sum((mask_back_org - mask_back_rec)**2) / num_pixel_back
        back_psnr = 10 * math.log10(PIXEL_MAX**2 / back_mse)

        # compute overall psnr
        overall_mse = np.mean((rec_arr - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)

        # write psnr with ground-truth mask
        if (fore_mse < 256*256 and back_mse < 256*256 and fore_mse > 0 and back_mse > 0):
            fore_mse_all.append(fore_mse)
            back_mse_all.append(back_mse)
            overall_mse_all.append(overall_mse)
            fore_psnr_all.append(fore_psnr)
            back_psnr_all.append(back_psnr)
            overall_psnr_all.append(overall_psnr)
        write_line = img_name + ' ' +str(fore_mse)+' ' + str(back_mse)+' ' + str(overall_mse)+' ' + str(fore_psnr)+' ' +str(back_psnr)+' ' +str(overall_psnr)+' \n'
        psnr_file.write(write_line)
    
    write_line = 'average: ' +str(np.mean(fore_mse_all))+' ' + str(np.mean(back_mse_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_all))+' ' +str(np.mean(back_psnr_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnr_file.write(write_line)
    psnr_file.close()


if __name__ == "__main__":
    # scale_all = ['1.0_0.2', '1.0_0.5', '1.0_0.8']
    scale_all = ['1.0_5.0', '1.0_2.0', '1.0_1.25']
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70', '80_80', '90_90']
    # scale_all = ['1.0_5.0']
    # quality_all = ['70_70']
    for quality in quality_all:
        for scale in scale_all:
            # psnr_compressed(quality)
            # psnr_label(scale, quality)
            # psnr_inferenced(scale, quality)
            psnr_inferenced_inv(scale, quality)
