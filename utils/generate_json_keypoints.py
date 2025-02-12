import json
import os
from PIL import Image

orgJsonFile = r'/model/gaocs/Detectron2/json/person_keypoints_val2014.json'
recJosnFile = r'/model/gaocs/Detectron2/json/keypoints_minVal2014_jpg.json'
imgPath = r'/data/gaocs/Understanding_Detection/minVal2014/'
# imgPath99 = '/gdata/gaocs/dataset/COCO/vgg_attend_rec/val2014_16_rec_6400_png/m0.001_v10_100000/'

imgFiles = sorted(os.listdir(imgPath))
orgJson = json.load(open(orgJsonFile, 'r'))
recJson = orgJson
keys = orgJson.keys()
info = orgJson['info']
images = orgJson['images']
licenses = orgJson['licenses']
annotations = orgJson['annotations']

imagesNew = []
annotationsNew = []

n = 0
# print(imgFiles[:640])
for img in imgFiles:
    imgSplit = img[:-4].split('_')
    imgId = imgSplit[-1]
    imgId = int(imgId)
    for image in images:
        if image['file_name']==img[:-4]+'.jpg':
            # image['file_name'] = imgPath99 + img
            image['file_name'] = img
            imagesNew.append(image)
            # print(image['file_name'], n)
            # png = Image.open(imgPath+img)
            # width = png.size[0]
            # height = png.size[1]
            # image['width'] = width
            # image['height'] = height
            # png.close()
    for annotation in annotations:
        if annotation['image_id']==imgId:
            annotationsNew.append(annotation)
            print(imgId, n)
    n += 1

recJson['images'] = imagesNew
recJson['annotations'] = annotationsNew
print(len(recJson['images']))
print(len(recJson['annotations']))
print(n, 'images have been processed')

with open(recJosnFile, "w") as f:
    json.dump(recJson, f)
    print("finished！")