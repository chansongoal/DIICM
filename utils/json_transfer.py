import os 

root_path_server = '/data/gaocs/Understanding_Detection/output_json/'
root_path_local = '/data/gaocs/Understanding_Detection/output_json_list/'
path1_all = os.listdir(root_path_server)
for path1 in path1_all:
    path2_all = os.listdir(root_path_server+path1)
    for path2 in path2_all:
        path3_all = os.listdir(root_path_server+path1+'/'+path2)
        for path3 in path3_all:
            path4_all = os.listdir(root_path_server+path1+'/'+path2+'/'+path3)
            for path4 in path4_all:
                json_server = root_path_server+path1+'/'+path2+'/'+path3+'/'+path4+'/inference/coco_instances_results.json'
                json_local = root_path_local+path1+'/'+path2+'/'+path3+'/'+'coco_instances_results_'+path4+'.json'
                if not os.path.exists(root_path_local+path1+'/'+path2+'/'+path3):
                    os.makedirs(root_path_local+path1+'/'+path2+'/'+path3)     
                if not os.path.exists(json_local):
                    para = 'cp ' + json_server + ' ' + json_local
                    print(para)
                    os.system(para)

# os.mkdir('/data/gaocs/Understanding_Detection/log/bbox_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/log/bbox_compress /data/gaocs/Understanding_Detection/log/bbox_compress_temp/bbox_compress')
# os.system('mv /data/gaocs/Understanding_Detection/log/bbox_compress_resize_1.2 /data/gaocs/Understanding_Detection/log/bbox_compress_temp/bbox_compress_resize_1.2')
# os.system('mv /data/gaocs/Understanding_Detection/log/bbox_compress_resize_0.8 /data/gaocs/Understanding_Detection/log/bbox_compress_temp/bbox_compress_resize_0.8')

# os.mkdir('/data/gaocs/Understanding_Detection/log/segm_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/log/segm_compress /data/gaocs/Understanding_Detection/log/segm_compress_temp/segm_compress')
# os.mkdir('/data/gaocs/Understanding_Detection/log/segm_boxblur_temp')
# os.system('mv /data/gaocs/Understanding_Detection/log/segm_boxblur /data/gaocs/Understanding_Detection/log/segm_boxblur_temp/segm_boxblur')


# os.mkdir('/data/gaocs/Understanding_Detection/bbox_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/bbox_compress /data/gaocs/Understanding_Detection/bbox_compress_temp/bbox_compress')
# os.system('mv /data/gaocs/Understanding_Detection/bbox_compress_resize_1.2 /data/gaocs/Understanding_Detection/bbox_compress_temp/bbox_compress_resize_1.2')
# os.system('mv /data/gaocs/Understanding_Detection/bbox_compress_resize_0.8 /data/gaocs/Understanding_Detection/bbox_compress_temp/bbox_compress_resize_0.8')

# os.mkdir('/data/gaocs/Understanding_Detection/segm_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/segm_compress /data/gaocs/Understanding_Detection/segm_compress_temp/segm_compress')
# os.mkdir('/data/gaocs/Understanding_Detection/segm_boxblur_temp')
# os.system('mv /data/gaocs/Understanding_Detection/segm_boxblur /data/gaocs/Understanding_Detection/segm_boxblur_temp/segm_boxblur')


# os.mkdir('/data/gaocs/Understanding_Detection/output_json/bbox_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/output_json/bbox_compress /data/gaocs/Understanding_Detection/output_json/bbox_compress_temp/bbox_compress')
# os.system('mv /data/gaocs/Understanding_Detection/output_json/bbox_compress_resize_1.2 /data/gaocs/Understanding_Detection/output_json/bbox_compress_temp/bbox_compress_resize_1.2')
# os.system('mv /data/gaocs/Understanding_Detection/output_json/bbox_compress_resize_0.8 /data/gaocs/Understanding_Detection/output_json/bbox_compress_temp/bbox_compress_resize_0.8')

# os.mkdir('/data/gaocs/Understanding_Detection/output_json/segm_compress_temp')
# os.system('mv /data/gaocs/Understanding_Detection/output_json/segm_compress /data/gaocs/Understanding_Detection/output_json/segm_compress_temp/segm_compress')
# os.mkdir('/data/gaocs/Understanding_Detection/output_json/segm_boxblur_temp')
# os.system('mv /data/gaocs/Understanding_Detection/output_json/segm_boxblur /data/gaocs/Understanding_Detection/output_json/segm_boxblur_temp/segm_boxblur')