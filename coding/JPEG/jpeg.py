import os 
from PIL import Image 
import numpy as np 


def jpeg_compression(org_path, rec_path, quality):
    os.makedirs(rec_path, exist_ok=True)
    img_names = os.listdir(org_path)

    for idx, img_name in enumerate(img_names):
        img = Image.open(os.path.join(org_path, img_name))
        img.save(os.path.join(rec_path, img_name[:-4]+'.jpg'), quality=quality)
    
def anchor_compression():
    org_path = "/gdata/gaocs/dataset/COCO/minVal2014"
    quality_all = [70]

    for quality in quality_all:
        compressed_path = f"/gdata1/gaocs/Data_DIICM/compressed/quality{quality}"
        jpeg_compression(org_path, compressed_path, quality)

def transformed_compression():
    alpha_all = [0.5]
    quality_all = [50, 40]

    for alpha in alpha_all:
        for quality in quality_all:
            transformed_path = f"/gdata1/gaocs/Data_DIICM/transformed/inferred/MaskRCNN_Res101_FPN_0.5/1.0_{alpha}"
            transformed_compressed_path = f"/gdata1/gaocs/Data_DIICM/transformed_compressed/jpeg_anchor/inferred/MaskRCNN_Res101_FPN_0.5/1.0_{alpha}/quality{quality}/rec_jpg"
            jpeg_compression(transformed_path, transformed_compressed_path, quality)

def main():
    transformed_compression()

if __name__ == "__main__":
    main()
