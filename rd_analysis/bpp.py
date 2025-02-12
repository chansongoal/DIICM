from cProfile import label
from fileinput import filename
import os 
from PIL import Image
import numpy as np 

def bpp_scaled():
    scale_all = ['1.0_0.2', '1.0_0.5', '1.0_0.8']
    # quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70', '80_80']
    quality_all = ['90_90']
    root_path = r'/data/gaocs/Understanding_Detection/'
    # label_config = 'inferenced/MaskRCNN_Res101_FPN_0.5'
    label_config = 'label'
    transform_config = 'transformed_compressed'

    for scale in scale_all:
        for quality in quality_all:
            img_path = root_path + transform_config + '/' + label_config + '/' + scale + '/' + quality + '/'
            psnr_filename = r'/data/gaocs/Understanding_Detection/bitrate/bpp_' + transform_config + '_' + 'label' + '_' + scale + '_' + quality +'.txt'
            file = open(psnr_filename, 'w')

            img_files = os.listdir(img_path)
            bpp_all = []
            for idx, img_file in enumerate(img_files):
                img_name = img_path + img_file
                img = Image.open(img_name)
                width, height = img.size[0], img.size[1]
                num_pixles = width * height
                img_size = os.path.getsize(img_name)
                bpp = img_size * 8 / num_pixles 
                bpp_all.append(bpp)
                # print(img_file, bpp)
                file.write(img_file + ' ' + str(img_size) + ' ' + str(width) + ' ' + str(height)  + ' ' + str(bpp))
                file.write('\n')
                img.close()

            bpp_avg = np.sum(bpp_all) / len(bpp_all)
            # print(bpp_avg)
            file.write(str(bpp_avg))
            file.close()

def bpp_jpeg():
    quality_all = ['1', '5', '10', '20', '25', '30', '35', '40', '45', '50', '60', '70', '75', '80']
    root_path = '/data/gaocs/Understanding_Detection/compressed/'
    for quality in quality_all:
        img_path = root_path + quality + '/'
        bpp_filename = '/data/gaocs/Understanding_Detection/bitrate/bpp_compressed_' + quality + '_' + quality + '.txt'
        file = open(bpp_filename, 'w')

        img_files = os.listdir(img_path)
        img_files.sort()
        bpp_all = []
        for idx, img_file in enumerate(img_files):
            img_name = img_path + img_file
            img = Image.open(img_name)
            width, height = img.size[0], img.size[1]
            num_pixles = width * height
            img_size = os.path.getsize(img_name)
            bpp = img_size * 8 / num_pixles 
            bpp_all.append(bpp)
            # print(img_file, bpp)
            file.write(img_file + ' ' + str(img_size) + ' ' + str(width) + ' ' + str(height)  + ' ' + str(bpp))
            file.write('\n')
            img.close()

        bpp_avg = np.sum(bpp_all) / len(bpp_all)
        # print(bpp_avg)
        file.write(str(bpp_avg))
        file.close()

if __name__ == '__main__':
    bpp_jpeg()