import os 
from PIL import Image 
import numpy as np 

scale_all = ['1.0_0.5']
iou_threshold = '0.5'

# quality_all = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
quality_all = [15]

for scale in scale_all:
    for quality in quality_all:
        # orgPath = r'/data/gaocs/Understanding_Detection/transformed/label/' + scale 
        # recPath = r'/data/gaocs/Understanding_Detection/transformed_compressed/label/' + scale + '/' + str(quality) + '_' + str(quality)
        orgPath = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + scale 
        recPath = r'/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + scale + '/' + str(quality) + '_' + str(quality)
        if not os.path.exists(recPath):
            os.makedirs(recPath)

        imgNames = os.listdir(orgPath)

        for idx, imgName in enumerate(imgNames):
            img = Image.open(os.path.join(orgPath, imgName))
            img.save(os.path.join(recPath, imgName[:-4]+'.jpg'), quality=quality)
    