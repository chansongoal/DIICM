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


def coco_merge(quality1, quality2, method):
    json_file = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    org_path = '/data/gaocs/Understanding_Detection/minVal2014/'
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
        org_img.close()

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # get the ground-truth mask
        if 'bbox' in method:
            if (len(anns)==0):
                mask_3d = np.ones((hgt,wdt,3), dtype=np.uint8)
            else:
                mask_3d = np.zeros((hgt,wdt,3), dtype=np.uint8)
                x,y,w,h = anns[0]['bbox']
                if 'resize' in method:
                    scale = float(method.split('_')[-1])
                    x,y,w,h = int(x-(scale-1)*w/2),int(y-(scale-1)*h/2),int(w*scale), int(h*scale)

                x1,x2,y1,y2 = int(x), int(x+w), int(y), int(y+h)
                x1,x2,y1,y2 = clip(x1,wdt), clip(x2,wdt), clip(y1,hgt), clip(y2,hgt)
                mask_3d[y1:y2,x1:x2,:] = 1
                for n in range(1,len(anns)):
                    x,y,w,h = anns[n]['bbox']
                    if 'resize' in method:
                        scale = float(method.split('_')[-1])
                        x,y,w,h = int(x-(scale-1)*w/2),int(y-(scale-1)*h/2),int(w*scale), int(h*scale)
                    x1,x2,y1,y2 = int(x), int(x+w), int(y), int(y+h)
                    x1,x2,y1,y2 = clip(x1,wdt), clip(x2,wdt), clip(y1,hgt), clip(y2,hgt)
                    mask_3d[y1:y2,x1:x2,:] = 1
                mask_3d[mask_3d>0] = 1
                mask_3d = mask_3d.astype(np.uint8)
        elif 'segm' in method:
            if (len(anns)==0):
                mask_3d = np.ones((hgt,wdt,3), dtype=np.uint8)
            else:
                mask = coco.annToMask(anns[0])
                for n in range(1,len(anns)):
                    mask += coco.annToMask(anns[n])
                mask[mask>0] = 1
                mask = mask.astype(np.uint8)
                mask_3d = np.zeros((hgt, wdt, 3))
                mask_3d[:,:,0] = mask[:hgt, :wdt]
                mask_3d[:,:,1] = mask[:hgt, :wdt]
                mask_3d[:,:,2] = mask[:hgt, :wdt]
        inverse_mask_3d = -mask_3d + 1

        processed_path = '/data/gaocs/Understanding_Detection/segm_contrast/segm_contrast_compress/' + quality1 + '_' + quality2 + '/'
        inverse_processed_path = '/data/gaocs/Understanding_Detection/segm_contrast/segm_contrast_compress_inverse/' + quality1 + '_' + quality2 + '/'
        if not os.path.exists(inverse_processed_path):
            os.makedirs(inverse_processed_path) 

        compressed_img1 = Image.open(processed_path + img['file_name'][:-4]+'.jpg')
        compressed_img2 = Image.open(processed_path + img['file_name'][:-4]+'.jpg')

        if 'contrast' in method:
            enhancer1 = ImageEnhance.Contrast(compressed_img1)
            enhancer2 = ImageEnhance.Contrast(compressed_img2)
            enhanced_img1 = enhancer1.enhance(1)
            enhanced_img2 = enhancer2.enhance(1/float(quality1.split('_')[1]))
            masked_arr1 = mask_3d * np.asarray(enhanced_img1)
            masked_arr2 = inverse_mask_3d * np.asarray(enhanced_img2)
        # masked_arr1 = mask_3d * np.zeros((hgt,wdt,3))
        # masked_arr2 = inverse_mask_3d * np.ones((hgt,wdt,3))*255
        # masked_arr1 = mask_3d * np.ones((hgt,wdt,3)) * int(quality1)
        # masked_arr2 = inverse_mask_3d * np.ones((hgt,wdt,3))*int(quality2)
        mask_arr = masked_arr1 + masked_arr2
        mask_arr = mask_arr.astype(np.uint8)
        mask_img = Image.fromarray(mask_arr)
        mask_img.save(inverse_processed_path+img['file_name'][:-4]+'.png')
        
        # enhanced_img1.save(rec_path+img['file_name'][:-4]+'.png')

        


if __name__ == "__main__":
    quality1_all = ['1_0.8', '1_0.5', '1_0.2']
    quality2_all = ['75_75', '50_50', '20_20']
    for quality1 in quality1_all:
        for quality2 in quality2_all:
            if quality1 != quality2:
                coco_merge(quality1, quality2, 'segm_contrast_compress')
