import json 

orgJson = json.load(open('/model/gaocs/Detectron2/json/keypoints_minVal2014_jpg.json', 'r'))

for img in orgJson['images']:
    img['file_name'] = img['file_name'][:-4] + '.png'

with open('/model/gaocs/Detectron2/json/keypoints_minVal2014_png.json', 'w') as f:
    json.dump(orgJson, f)
    