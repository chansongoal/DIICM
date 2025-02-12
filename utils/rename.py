import os 

path = '/data/gaocs/Understanding_Detection/log_tide/'
filenames = os.listdir(path)
for filename in filenames:
    if not '.log' in filename:
        continue
    if not '50' in filename:
        continue
    filename_split = filename.split('_')
    new_filename = path
    for n in range(len(filename_split)-3):
        new_filename = new_filename + filename_split[n]
        new_filename = new_filename + '_'
    new_filename = new_filename + filename_split[-1][:-4]
    new_filename = new_filename + '_'
    new_filename = new_filename + filename_split[-3]
    new_filename = new_filename + '_'
    new_filename = new_filename + filename_split[-2]
    new_filename = new_filename + '.log'
    # new_filename = '/data/gaocs/Understanding_Detection/log/segm_compress/segm_compress/coco_minVal2014_5000_segm_compress_'+filename_split[3]+'_'+filename_split[4]+'_'+filename_split[5]+'_'+filename_split[7]+'_'+filename_split[8]
    para = 'mv '+path+filename + ' ' +new_filename
    print(para)
    os.system(para)