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

def coco_mask():
    json_file = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    org_path = '/data/gaocs/Understanding_Detection/minVal2014/'
    mask_path = '/data/gaocs/Understanding_Detection/label_mask/'
    coco = COCO(json_file)
    catIds = coco.getCatIds() 
    print('catIds: ', catIds)
    cats = coco.loadCats(coco.getCatIds())
    print(len(cats), catIds)

    imgIds = coco.getImgIds() 
    print(len(imgIds))

    for idx in range(len(imgIds)):
        img = coco.loadImgs(imgIds[idx])[0]
        org_img = Image.open(org_path + img['file_name'][:-4]+'.jpg')
        wdt = org_img.size[0]
        hgt = org_img.size[1]
        org_img.close()

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # get the ground-truth mask  
        if (len(anns)==0):
            mask_3d = np.ones((hgt,wdt,3), dtype=np.uint8)
        else:
            mask = coco.annToMask(anns[0])
            for n in range(1,len(anns)):
                mask += coco.annToMask(anns[n])
            mask[mask>0] = 1
            mask = mask.astype(np.uint8)
            mask_3d = np.zeros((hgt, wdt, 3), dtype=np.uint8)
            mask_3d[:,:,0] = mask[:hgt, :wdt]
            mask_3d[:,:,1] = mask[:hgt, :wdt]
            mask_3d[:,:,2] = mask[:hgt, :wdt]

        mask_img = Image.fromarray(mask_3d)
        mask_img.save(mask_path + img['file_name'][:-4]+'.png')



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

        # merge image according to the method
        compressed_path1 = r'/data/gaocs/Understanding_Detection/minVal2014/'
        compressed_path2 = r'/data/gaocs/Understanding_Detection/minVal2014/' 
        # compressed_path1 = r'/data/gaocs/Understanding_Detection/compressed/75/'
        # compressed_path2 = r'/data/gaocs/Understanding_Detection/compressed/75/' 
        # if 'compress' in method:
        #     if quality1 == '100':
        #         compressed_path1 = r'/data/gaocs/Understanding_Detection/minVal2014/'
        #     else:
        #         compressed_path1 = r'/data/gaocs/Understanding_Detection/compressed/' + quality1 + '/'
        #     compressed_path2 = r'/data/gaocs/Understanding_Detection/compressed/' + quality2 + '/'  
        method_prefix = method.split('_')[0] + '_' + method.split('_')[1]   
        rec_path = '/data/gaocs/Understanding_Detection/' + method_prefix + '/' + method +'/'+ quality1+'_'+quality2 + '/'
        if not os.path.exists(rec_path):
            os.makedirs(rec_path) 

        compressed_img1 = Image.open(compressed_path1 + img['file_name'][:-4]+'.jpg')
        # compressed_img2 = Image.open(compressed_path2 + img['file_name'][:-4]+'.jpg')
        if 'compress' in method:
            masked_arr1 = mask_3d * np.asarray(compressed_img1)
            # masked_arr2 = inverse_mask_3d * np.asarray(compressed_img2)
            masked_arr2 = inverse_mask_3d * np.ones_like(masked_arr1)*255
        # if 'boxblur' in method:
        #     blured_img1 = compressed_img1.filter(ImageFilter.BoxBlur(float(quality1)))
        #     blured_img2 = compressed_img2.filter(ImageFilter.BoxBlur(float(quality2)))
        #     masked_arr1 = mask_3d * np.asarray(blured_img1)
        #     masked_arr2 = inverse_mask_3d * np.asarray(blured_img2)
        # if 'uniformnoise' in method:
        #     noise_fore = (np.random.random_sample((hgt,wdt,3))-0.5) * float(quality1)    #quality here is the factor that adjusts the range of noise
        #     noise_back = (np.random.random_sample((hgt,wdt,3))-0.5) * float(quality2)
        #     masked_arr1 = mask_3d * np.asarray(compressed_img1+noise_fore)
        #     masked_arr2 = inverse_mask_3d * np.asarray(compressed_img2+noise_back)
        # if 'normalnoise' in method:
        #     noise_fore = np.random.normal(0, float(quality1), (hgt,wdt,3))
        #     noise_back = np.random.normal(0, float(quality2), (hgt,wdt,3))
        #     masked_arr1 = mask_3d * np.asarray(compressed_img1+noise_fore)
        #     masked_arr2 = inverse_mask_3d * np.asarray(compressed_img2+noise_back)
        # if 'brightness' in method:
        #     enhancer1 = ImageEnhance.Brightness(compressed_img1)
        #     enhancer2 = ImageEnhance.Brightness(compressed_img2)
        #     enhanced_img1 = enhancer1.enhance(float(quality1))
        #     enhanced_img2 = enhancer2.enhance(float(quality2))
        #     masked_arr1 = mask_3d * np.asarray(enhanced_img1)
        #     masked_arr2 = inverse_mask_3d * np.asarray(enhanced_img2)
        # if 'color' in method:
        #     enhancer1 = ImageEnhance.Color(compressed_img1)
        #     enhancer2 = ImageEnhance.Color(compressed_img2)
        #     enhanced_img1 = enhancer1.enhance(float(quality1))
        #     enhanced_img2 = enhancer2.enhance(float(quality2))
        #     masked_arr1 = mask_3d * np.asarray(enhanced_img1)
        #     masked_arr2 = inverse_mask_3d * np.asarray(enhanced_img2)
        # if 'contrast' in method:
        #     enhancer1 = ImageEnhance.Contrast(compressed_img1)
        #     enhancer2 = ImageEnhance.Contrast(compressed_img2)
        #     enhanced_img1 = enhancer1.enhance(float(quality1))
        #     enhanced_img2 = enhancer2.enhance(float(quality2))
        #     masked_arr1 = mask_3d * np.asarray(enhanced_img1)
        #     masked_arr2 = inverse_mask_3d * np.asarray(enhanced_img2)
        # masked_arr1 = mask_3d * np.zeros((hgt,wdt,3))
        # masked_arr2 = inverse_mask_3d * np.ones((hgt,wdt,3))*255
        # masked_arr1 = mask_3d * np.ones((hgt,wdt,3)) * int(quality1)
        # masked_arr2 = inverse_mask_3d * np.ones((hgt,wdt,3))*int(quality2)
        mask_arr = masked_arr1 + masked_arr2
        mask_arr = mask_arr.astype(np.uint8)
        mask_img = Image.fromarray(mask_arr)
        mask_img.save(rec_path+img['file_name'][:-4]+'.png')
        
        # enhanced_img1.save(rec_path+img['file_name'][:-4]+'.png')

def coco_merge_inferenced(quality1, quality2, method):
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    inferenced_path = r'/data/gaocs/Understanding_Detection/inferenced/MaskRCNN_R101_FPN/org_org_0.2/'
    rec_path = r'/data/gaocs/Understanding_Detection/segm_contrast/segm_contrast_inferenced/MaskRCNN_Res101_FPN/' + quality1+'_'+quality2 + '_0.2/'
    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get the inferenced mask
        img_name = img_files[idx]
        inferenced_img = Image.open(inferenced_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        # print(inferenced_arr.shape, np.max(inferenced_arr), np.mean(inferenced_arr))
        inv_mask = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask.shape[0], inv_mask.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask
        temp[:,:,1] = inv_mask
        temp[:,:,2] = inv_mask
        inv_mask = temp
        mask = (-inv_mask + 1).astype(np.uint8)

        # merge image according to the method 
        method_prefix = method.split('_')[0] + '_' + method.split('_')[1]   

        org_img = Image.open(org_path + img_name)
        
        # if 'contrast' in method:
        enhancer = ImageEnhance.Contrast(org_img)
        enhanced_img = enhancer.enhance(float(quality2))
        org_arr = mask * np.asarray(org_img)
        mask_arr = inv_mask * np.asarray(enhanced_img)
        mask_arr = org_arr + mask_arr
        mask_arr = mask_arr.astype(np.uint8)
        mask_img = Image.fromarray(mask_arr)
        mask_img.save(rec_path+img_name[:-4]+'.png')
        # enhanced_img.save(rec_path+img_name[:-4]+'_enhanced.png')




if __name__ == "__main__":
    # quality1_all = ['1.0']
    # quality2_all = ['0.2', '0.5', '0.8']
    # for quality1 in quality1_all:
    #     for quality2 in quality2_all:
    #         if quality1 != quality2:
    #             coco_merge_inferenced(quality1, quality2, 'segm_contrast_inferenced')
    coco_mask()