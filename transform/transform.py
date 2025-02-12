import enum
import os

from pycocotools.coco import COCO
from PIL import Image, ImageFilter, ImageEnhance
from matplotlib import pyplot as plt
import numpy as np 
import pickle

def clip(value, upper):
    if value < 0:
        value = 0
    elif value >upper:
        value = upper
    return value


def transform(alpha):
    # org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    # mask_path = r'/data/gaocs/Understanding_Detection/label_mask/'
    # rec_path = r'/data/gaocs/Understanding_Detection/transformed/1.0_' + str(alpha) + '/'

    iou_threshold = '0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    mask_path = r'/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
        mask_img = Image.open(mask_path + img_name[:-4]+'.png')
        mask_arr = np.asarray(mask_img)
        inv_mask_arr = -mask_arr + 1

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * alpha + (1-alpha)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr

        img_arr_scaled = fore_arr + back_arr_sacled
        img_arr_scaled = img_arr_scaled.astype(np.uint8)
        img_scaled = Image.fromarray(img_arr_scaled)
        img_scaled.save(rec_path+img_name[:-4]+'.png')


def transform_inferenced(alpha):
    # iou_thresholds = ['0.2', '0.5', '0.75']
    # for iou_threshold in iou_thresholds:
    iou_threshold = '0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    mask_path = r'/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
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

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * alpha + (1-alpha)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr

        img_arr_scaled = fore_arr + back_arr_sacled
        img_arr_scaled = img_arr_scaled.astype(np.uint8)
        img_scaled = Image.fromarray(img_arr_scaled)
        img_scaled.save(rec_path+img_name[:-4]+'.png')


def inv_transform(alpha, beta, quality):
    iou_threshold = '0.5'
    mask_path = r'/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'
    org_path = r'/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/' + quality + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed_compressed_inv/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(beta) + '/' + quality + '/'
    # org_path = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/'
    # rec_path = r'/data/gaocs/Understanding_Detection/transformed_inv/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(beta) + '/'
    # mask_path = r'/Users/changsheng/Downloads/label_mask/'
    # org_path = r'/Users/changsheng/Downloads/transformed_compressed_label_0.2_20/'
    # rec_path = r'/Users/changsheng/Downloads/transformed_compressed_label_0.2_20_inv/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    inv_statistics_path = '/data/gaocs/Understanding_Detection/inv_statistics/'
    inv_statistics_filename = 'inv_statistics_1.0_' + str(alpha) + '_' + quality + '.txt'
    up_per_all = []
    low_per_all = []
    maximum_all = [] 
    minimum_all = [] 
    avg_all = []
    up_avg_all = [] 
    low_avg_all = []
    if not os.path.exists(inv_statistics_path):
        os.makedirs(inv_statistics_path)
    inv_statistics_file = open(inv_statistics_path + inv_statistics_filename, 'w')

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
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

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img).astype(np.float32)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

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

if __name__ == "__main__":
    alphas = [0.2, 0.5, 0.8]
    betas = [5.0, 2.0, 1.25]
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70', '80_80', '90_90']
    alphas = [0.5]
    betas = [2.0]
    for idx, alpha in enumerate(alphas):
        for quality in quality_all:
            # transform_inferenced(alpha)
            inv_transform(alphas[idx], betas[idx], quality)
