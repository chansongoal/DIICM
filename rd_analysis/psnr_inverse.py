import os

from pycocotools.coco import COCO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np 
import pickle
import math
from PIL import Image, ImageFilter, ImageEnhance

def coco_merge(quality1, quality2):
    json_file = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    org_path = '/data/gaocs/Understanding_Detection/minVal2014/'

    psnrPath = '/data/gaocs/Understanding_Detection/log_psnr_inverse/'
    psnrFileName = 'psnr_inverse_' + str(quality1) + '_' + str(quality2) + '.log'
    psnrFile = open(psnrPath + psnrFileName, 'w')

    fore_mse_all = []
    back_mse_all = []
    overall_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []
    overall_psnr_all = []

    compressed_path1 = r'/data/gaocs/Understanding_Detection/minVal2014/'
    # compressed_path2 = r'/data/gaocs/Understanding_Detection/segm_contrast/segm_contrast_compress_inverse/' + quality1 + '_' + quality2 + '/'
    compressed_path2 = r'/data/gaocs/Understanding_Detection/compressed/50/'
    # rec_path = '/data/gaocs/Understanding_Detection/merged/' + quality1+'_'+quality2 + '/'
    # if not os.path.exists(rec_path):
        # os.mkdir(rec_path)

    coco = COCO(json_file)
    catIds = coco.getCatIds() 
    print('catIds: ', catIds)
    cats = coco.loadCats(coco.getCatIds())
    print(len(cats), catIds)

    imgIds = coco.getImgIds() 
    print(len(imgIds))
    imgIds = imgIds[:100]
    # print(coco.dataset['categories'])
    for idx in range(len(imgIds)):
        img = coco.loadImgs(imgIds[idx])[0]
        org_img = Image.open(org_path + img['file_name'][:-4]+'.jpg')
        wdt = org_img.size[0]
        hgt = org_img.size[1]
        org_arr = np.asarray(org_img, dtype=np.float32)
        org_img.close()

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if (len(anns)==0):
            mask_3d = np.ones((hgt,wdt,3), dtype=np.uint8)
        else:
            mask = coco.annToMask(anns[0])

            for n in range(1,len(anns)):
                mask += coco.annToMask(anns[n])
                # print(i, np.mean(coco.annToMask(anns[i])), np.max(coco.annToMask(anns[i])), np.min(coco.annToMask(anns[i])))

            # print(mask.shape, np.max(mask), np.min(mask), len(np.nonzero(mask)[0]))
            mask[mask>0] = 1
            mask = mask.astype(np.uint8)
            mask_3d = np.zeros((hgt, wdt, 3), dtype=np.float32)
            mask_3d[:,:,0] = mask[:hgt, :wdt]
            mask_3d[:,:,1] = mask[:hgt, :wdt]
            mask_3d[:,:,2] = mask[:hgt, :wdt]

        inverse_mask_3d = -mask_3d + 1

        compressed_img1 = Image.open(compressed_path2 + img['file_name'][:-4]+'.png')
        compressed_img2 = Image.open(compressed_path2 + img['file_name'][:-4]+'.png')

        masked_arr1 = mask_3d * np.asarray(compressed_img1, dtype=np.float32)
        masked_arr2 = inverse_mask_3d * np.asarray(compressed_img2, dtype=np.float32)

        masked_org1 = mask_3d * org_arr
        masked_org2 = inverse_mask_3d * org_arr

        PIXEL_MAX = 255.0
        num_pixel1 = np.count_nonzero(mask_3d)
        fore_mse = np.sum((masked_arr1 - masked_org1)**2) / num_pixel1
        fore_psnr = 10 * math.log10(PIXEL_MAX**2 / fore_mse)

        num_pixel2 = np.count_nonzero(inverse_mask_3d)
        back_mse = np.sum((masked_arr2 - masked_org2)**2) / num_pixel2
        back_psnr = 10 * math.log10(PIXEL_MAX**2 / back_mse)

        merge_arr = masked_arr1 + masked_arr2
        overall_mse = np.mean((merge_arr - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)

        if (fore_mse < 256*256 and back_mse < 256*256 and fore_mse > 0 and back_mse > 0):
            fore_mse_all.append(fore_mse)
            back_mse_all.append(back_mse)
            overall_mse_all.append(overall_mse)
            fore_psnr_all.append(fore_psnr)
            back_psnr_all.append(back_psnr)
            overall_psnr_all.append(overall_psnr)

        write_line = img['file_name'][:-4]+'.png'+' ' +str(fore_mse)+' ' + str(back_mse)+' ' + str(overall_mse)+' ' + str(fore_psnr)+' ' +str(back_psnr)+' ' +str(overall_psnr)+' \n'
        psnrFile.write(write_line)
            # print(img['file_name'][:-4]+'.png', fore_mse, back_mse, overall_mse, fore_psnr, back_psnr, overall_psnr)

    write_line = 'average: ' +str(np.mean(fore_mse_all))+' ' + str(np.mean(back_mse_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_all))+' ' +str(np.mean(back_psnr_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnrFile.write(write_line)
    psnrFile.close()


def psnr_inv_inferenced_inferenced(quality1, quality2):
    iou_threshold = '0.2'
    inferenced_path = inferenced_path = r'/data/gaocs/Understanding_Detection/inferenced/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'  #mask path 
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    rec_path = r'/data/gaocs/Understanding_Detection/segm_contrast/segm_contrast_inferenced_compress/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + quality1+'_'+quality2 + '/' #img path
    psnrPath = '/data/gaocs/Understanding_Detection/log_psnr_inverse/inferenced_inferenced/MaskRCNN_Res101_FPN_'+iou_threshold+'/'
    psnrFileName = 'psnr_inv_inferenced_inferenced_MaskRCNN_Res101_FPN_' + iou_threshold + '_' + str(quality1) + '_' + str(quality2) + '.log'
    if not os.path.exists(psnrPath):
        os.makedirs(psnrPath)
    psnrFile = open(psnrPath + psnrFileName, 'w')

    fore_mse_all = []
    back_mse_all = []
    overall_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []
    overall_psnr_all = []

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get the inferenced mask
        img_name = img_files[idx]
        inferenced_img = Image.open(inferenced_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        inv_mask = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask.shape[0], inv_mask.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask
        temp[:,:,1] = inv_mask
        temp[:,:,2] = inv_mask
        inv_mask = temp
        mask = (-inv_mask + 1).astype(np.uint8)

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img, dtype=np.float32)

        rec_img = Image.open(rec_path + img_name)
        rec_arr = np.asarray(rec_img, dtype=np.float32)

        mask_fore_org = org_arr * mask
        mask_back_org = org_arr * inv_mask
        mask_fore_rec = mask * rec_arr
        mask_back_rec = inv_mask * rec_arr

        PIXEL_MAX = 255.0
        num_pixel_fore = np.count_nonzero(mask)
        fore_mse = np.sum((mask_fore_org - mask_fore_rec)**2) / num_pixel_fore
        fore_psnr = 10 * math.log10(PIXEL_MAX**2 / fore_mse)

        num_pixel_back = np.count_nonzero(inv_mask)
        alpha = 1/float(quality1.split('_')[1])
        mean = np.sum(mask_back_rec) / num_pixel_back
        mask_back_rec_inv = alpha * mask_back_rec + (1-alpha) * mean
        rec_arr_merged = mask_fore_rec + mask_back_rec_inv

        back_mse = np.sum((mask_back_org - mask_back_rec_inv)**2) / num_pixel_back
        back_psnr = 10 * math.log10(PIXEL_MAX**2 / back_mse)

        overall_mse = np.mean((rec_arr_merged - org_arr)**2)
        overall_psnr = 10 * math.log10(PIXEL_MAX**2 / overall_mse)
  
        if (fore_mse < 256*256 and back_mse < 256*256 and fore_mse > 0 and back_mse > 0):
            fore_mse_all.append(fore_mse)
            back_mse_all.append(back_mse)
            overall_mse_all.append(overall_mse)
            fore_psnr_all.append(fore_psnr)
            back_psnr_all.append(back_psnr)
            overall_psnr_all.append(overall_psnr)

        write_line = img_name + ' ' +str(fore_mse)+' ' + str(back_mse)+' ' + str(overall_mse)+' ' + str(fore_psnr)+' ' +str(back_psnr)+' ' +str(overall_psnr)+' \n'
        psnrFile.write(write_line)

    write_line = 'average: ' +str(np.mean(fore_mse_all))+' ' + str(np.mean(back_mse_all))+' ' + str(np.mean(np.mean(overall_mse_all)))+' ' + str(np.mean(fore_psnr_all))+' ' +str(np.mean(back_psnr_all))+' ' +str(np.mean(overall_psnr_all))+' \n'
    psnrFile.write(write_line)
    psnrFile.close()


if __name__ == "__main__":
    quality1_all = ['1.0_0.2', '1.0_0.5', '1.0_0.8']
    quality2_all = ['75_75', '50_50', '20_20']
    for quality1 in quality1_all:
        for quality2 in quality2_all:
            if quality1 != quality2:
                psnr_inv_inferenced_inferenced(quality1, quality2)
    # coco_merge('50', '50')
