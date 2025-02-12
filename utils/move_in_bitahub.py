import os 

# for scaled
# root_path = '/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_0.5/'
# scales = ['1.0_0.8', '1.0_0.5', '1.0_0.2']
# quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70', '80_80', '90_90']

# root_path_dest = '/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_0.5/temp/'

# imgNames = ['COCO_val2014_000000000139', 'COCO_val2014_000000000785', 'COCO_val2014_000000001000', 'COCO_val2014_000000001268', 'COCO_val2014_000000002006', 'COCO_val2014_000000002261', 'COCO_val2014_000000005037', 'COCO_val2014_000000005992', 'COCO_val2014_000000006040', 'COCO_val2014_000000006954', 'COCO_val2014_000000007108', 'COCO_val2014_000000007816', 'COCO_val2014_000000008532']
# for scaleIdx, scale in enumerate(scales):
#     for qualityIdx, quality in enumerate(quality_all):
#         dest_path = root_path_dest + scale + '/' + quality + '/'
#         if not os.path.exists(dest_path): os.makedirs(dest_path)
#         for imgIdx, imgName in enumerate(imgNames):
#             para = 'cp ' + root_path+scale+'/'+quality+'/'+imgName+'.jpg' + ' ' +dest_path+imgName+'.jpg'
#             os.system(para)


# for compresse_only
root_path = '/data/gaocs/Understanding_Detection/compressed/'
quality_all = ['1', '5', '10', '20', '25', '30', '35', '40', '45', '50', '60', '70', '75', '80', '90']

root_path_dest = '/data/gaocs/Understanding_Detection/compressed/temp/'

imgNames = ['COCO_val2014_000000000139', 'COCO_val2014_000000000785', 'COCO_val2014_000000001000', 'COCO_val2014_000000001268', 'COCO_val2014_000000002006', 'COCO_val2014_000000002261', 'COCO_val2014_000000005037', 'COCO_val2014_000000005992', 'COCO_val2014_000000006040', 'COCO_val2014_000000006954', 'COCO_val2014_000000007108', 'COCO_val2014_000000007816', 'COCO_val2014_000000008532']
for qualityIdx, quality in enumerate(quality_all):
    dest_path = root_path_dest + quality + '/'
    if not os.path.exists(dest_path): os.makedirs(dest_path)
    for imgIdx, imgName in enumerate(imgNames):
        para = 'cp ' + root_path+quality+'/'+imgName+'.jpg' + ' ' +dest_path+imgName+'.jpg'
        os.system(para)