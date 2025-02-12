import os

from pycocotools.coco import COCO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np 
import pickle
import math

def coco_merge(quality):
    json_file = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    org_path = '/data/gaocs/Understanding_Detection/minVal2014/'

    psnrPath = '/data/gaocs/Understanding_Detection/log_psnr/'
    psnrFileName = 'psnr_' + str(quality) + '_' + str(quality) + '.log'
    psnrFile = open(psnrPath + psnrFileName, 'w')

    fore_mse_all = []
    back_mse_all = []
    overall_mse_all = []
    fore_psnr_all = []
    back_psnr_all = []
    overall_psnr_all = []

    compressed_path1 = r'/data/gaocs/Understanding_Detection/compressed/' + quality + '/'

    coco = COCO(json_file)
    catIds = coco.getCatIds() 
    print('catIds: ', catIds)
    cats = coco.loadCats(coco.getCatIds())
    print(len(cats), catIds)

    imgIds = coco.getImgIds() 
    print(len(imgIds))
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

        compressed_img1 = Image.open(compressed_path1 + img['file_name'][:-4]+'.jpg')
        # compressed_img2 = Image.open(compressed_path2 + img['file_name'][:-4]+'.jpg')

        masked_arr1 = mask_3d * np.asarray(compressed_img1, dtype=np.float32)
        masked_arr2 = inverse_mask_3d * np.asarray(compressed_img1, dtype=np.float32)

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
    # print('average: ', np.mean(fore_mse_all), np.mean(back_mse_all), np.mean(overall_mse_all), np.mean(fore_psnr_all), np.mean(back_psnr_all), np.mean(overall_psnr_all))


if __name__ == "__main__":
    quality_all = ['24', '20', '16', '10', '5', '1', '7', '2']
    for quality in quality_all:
        coco_merge(quality)
