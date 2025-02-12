#!/usr/local/bin/python
import os 
import re
import glob
import argparse

# from tidecv import TIDE
# import tidecv.datasets as datasets
from matplotlib import pyplot as plt


def ConfigModification(configFileName, DatasetName, LogPath, alpha, quality):
    configFile = open(configFileName, 'r')
    config = configFile.readlines()
    configFile.close()

    configFile = open(configFileName[:-12]+'.yaml', 'w')
    old_str1 = '  TEST'
    old_str2 = 'OUTPUT_DIR:'

    for line in config:
        if re.search(old_str1, line):
            write_line = '  TEST: (\"' + DatasetName + '\", )\n'
            # print(write_line)
            # print(line)
            configFile.write(write_line)
        elif re.search(old_str2, line):
            print(DatasetName)
            write_line = f"OUTPUT_DIR: {LogPath}/\n"
            print(write_line)
            configFile.write(write_line)
        else:
            configFile.write(line)
    configFile.close()

def train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath):
    # TrainFileName = '/ghome/gaocs/detection2_vgg/train_net_backup.py'
    configFile = open(TrainFileName, 'r')
    config = configFile.readlines()
    configFile.close()

    configFile = open(TrainFileName[:-10]+'.py', 'w')
    old_str1 = '        register_coco_instances'

    for line in config:
        if re.search(old_str1, line):
            write_line = '        register_coco_instances(\"' \
                        + DatasetName + '\", {}, \"' \
                        + JsonName + '\", \"' \
                        + DatasetPath +'\")\n'

            # print(write_line)
            # print(line)
            configFile.write(write_line)
        else:
            configFile.write(line)
    configFile.close()

def Faster_Res50_C4(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Faster_Res50_C4'
    DatasetName = f"{DatasetNamePrefix}_{NetworkConfig}_1.0_{alpha}_quality{quality}"
    TrainFileName = '/home/gaocs/projects/ICM-DIICM/Code/machines/detectron2/train_net_backup.py'
    configFileName = '/home/gaocs/projects/ICM-DIICM/Code/machines/detectron2/configs/faster_rcnn_R_50_C4_1x_backup.yaml'
    JsonName = '/home/gaocs/dataset/COCO/json/instances_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, LogPath, alpha, quality)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    # if processing_config == 'transformed':
    #     # log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+mask_type+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    #     log_name = f"{LogPath}/{DatasetNamePrefix}_{processing_config}_{mask_type}_{mask_network}_{NetworkConfig}_1.0_{alpha}.log 2>&1"
    # # elif processing_config = 'transformed_compressed':
    # #     log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    # compress_para = 'cd /home/gaocs/projects/ICM-DIICM/Code/machines/detectron2/; \
    #                 python3 train_net.py \
    #                 --config-file ./configs/faster_rcnn_R_50_C4_1x.yaml \
    #                 --eval-only \
    #                 MODEL.WEIGHTS /home/gaocs/models/detectron2/model_final_721ade_FasterRCNN_R50_C4.pkl ' \
    #                 + '>' + log_name
    eval_para = 'cd /home/gaocs/projects/ICM-DIICM/Code/machines/detectron2/; \
                    python3 train_net.py \
                    --config-file ./configs/faster_rcnn_R_50_C4_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /home/gaocs/models/detectron2/model_final_721ade_FasterRCNN_R50_C4.pkl '
    print(eval_para)
    os.system(eval_para)

def Faster_Res50_FPN(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Faster_Res50_FPN'
    DatasetName = DatasetNamePrefix+'_'+NetworkConfig+'_' + QualityConfig
    TrainFileName = '/model/gaocs/Detectron2/code/train_net_backup.py'
    configFileName = '/model/gaocs/Detectron2/configs/faster_rcnn_R_50_FPN_1x_backup.yaml'
    JsonName = '/model/gaocs/Detectron2/json/instances_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, NetworkConfig, scale, quality, label_config, transform_config)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    temp = label_config.split('/')
    if len(temp) == 1:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+label_config+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    elif len(temp) == 2:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    compress_para = 'cd /model/gaocs/Detectron2/code/; \
                    python3 /model/gaocs/Detectron2/code/train_net.py \
                    --config-file /model/gaocs/Detectron2/configs/faster_rcnn_R_50_FPN_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /model/gaocs/Detectron2/model_final_b275ba_FasterRCNN_R50_FPN.pkl ' \
                    + '>' + log_name
    print(compress_para)
    os.system(compress_para)

def Mask_Res50_C4(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Mask_Res50_C4'
    DatasetName = DatasetNamePrefix+'_'+NetworkConfig+'_' + QualityConfig
    TrainFileName = '/model/gaocs/Detectron2/code/train_net_backup.py'
    configFileName = '/model/gaocs/Detectron2/configs/mask_rcnn_R_50_C4_1x_backup.yaml'
    JsonName = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    ConfigModification(configFileName, DatasetName, NetworkConfig, scale, quality, label_config, transform_config)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    temp = label_config.split('/')
    if len(temp) == 1:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+label_config+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    elif len(temp) == 2:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    compress_para = 'cd /model/gaocs/Detectron2/code/; \
                    python3 /model/gaocs/Detectron2/code/train_net.py \
                    --config-file /model/gaocs/Detectron2/configs/mask_rcnn_R_50_C4_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /model/gaocs/Detectron2/model_final_9243eb_MaskRCNN_R50_C4.pkl ' \
                    + '>' + log_name
    print(compress_para)
    os.system(compress_para)

    # do tide evaluation
    # netPath = NetworkConfig.split('_')[-3]+NetworkConfig.split('_')[-2]+NetworkConfig.split('_')[-1]
    # method_prefix = method.split('_')[0] + '_' + method.split('_')[1]
    # json_file = '/data/gaocs/Understanding_Detection/output_json/'+method_prefix+'/'+method+'/MaskRCNN_Res101_FPN_0.75/' + netPath + '/' + QualityConfig + '/inference/coco_instances_results.json'
    # log_file = '/data/gaocs/Understanding_Detection/log_tide_inferenced/MaskRCNN_Res101_FPN_0.75/'+method+'_'+netPath+'_bboxdAp_'+QualityConfig+'.log 2>&1'
    # tide_para = 'cd /model/gaocs/Detectron2/code/; \
    #              python3 /model/gaocs/Detectron2/code/tide_eval.py \
    #              --json_file=' + json_file + ' \
    #              --ap_mode=bboxdAp >' + log_file
    # print(tide_para)
    # os.system(tide_para)
    # log_file = '/data/gaocs/Understanding_Detection/log_tide_inferenced/MaskRCNN_Res101_FPN_0.75/'+method+'_'+netPath+'_segmdAp_'+QualityConfig+'.log 2>&1'
    # tide_para = 'cd /model/gaocs/Detectron2/code/; \
    #              python3 /model/gaocs/Detectron2/code/tide_eval.py \
    #              --json_file=' + json_file + ' \
    #              --ap_mode=segmdAp >' + log_file
    # print(tide_para)
    # os.system(tide_para)

def Mask_Res50_FPN(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Mask_Res50_FPN'
    DatasetName = DatasetNamePrefix+'_'+NetworkConfig+'_' + QualityConfig
    TrainFileName = '/model/gaocs/Detectron2/code/train_net_backup.py'
    configFileName = '/model/gaocs/Detectron2/configs/mask_rcnn_R_50_FPN_1x_backup.yaml'
    JsonName = '/model/gaocs/Detectron2/json/instances_minVal2014_jpg.json'
    ConfigModification(configFileName, DatasetName, NetworkConfig, scale, quality, label_config, transform_config)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    temp = label_config.split('/')
    if len(temp) == 1:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+label_config+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    elif len(temp) == 2:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    compress_para = 'cd /model/gaocs/Detectron2/code/; \
                    python3 /model/gaocs/Detectron2/code/train_net.py \
                    --config-file /model/gaocs/Detectron2/configs/mask_rcnn_R_50_FPN_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /model/gaocs/Detectron2/model_final_f10217_MaskRCNN_R50_FPN.pkl ' \
                    + '>' + log_name
    print(compress_para)
    os.system(compress_para)

    # do tide evaluation
    # netPath = DatasetName.split('_')[-5]+DatasetName.split('_')[-4]+DatasetName.split('_')[-3]
    # mvPath = DatasetName.split('_')[-2]+'_'+DatasetName.split('_')[-1]
    # method_prefix = method.split('_')[0] + '_' + method.split('_')[1]
    # json_file = '/data/gaocs/Understanding_Detection/output_json/'+method_prefix+'/'+method+'/' + netPath + '/' + mvPath + '/inference/coco_instances_results.json'
    # log_file = '/data/gaocs/Understanding_Detection/log_tide/'+method+'_MaskRes50FPN_bboxdAp_'+mvPath+'.log 2>&1'
    # tide_para = 'cd /model/gaocs/Detectron2/code/; \
    #              python3 /model/gaocs/Detectron2/code/tide_eval.py \
    #              --json_file=' + json_file + ' \
    #              --ap_mode=bboxdAp >' + log_file
    # print(tide_para)
    # os.system(tide_para)
    # log_file = '/data/gaocs/Understanding_Detection/log_tide/'+method+'_MaskRes50FPN_segmdAp_'+mvPath+'.log 2>&1'
    # tide_para = 'cd /model/gaocs/Detectron2/code/; \
    #              python3 /model/gaocs/Detectron2/code/tide_eval.py \
    #              --json_file=' + json_file + ' \
    #              --ap_mode=segmdAp >' + log_file
    # print(tide_para)
    # os.system(tide_para)

def Keypoints_Res50_FPN(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Keypoints_Res50_FPN'
    DatasetName = DatasetNamePrefix+'_'+NetworkConfig+'_' + QualityConfig
    TrainFileName = '/model/gaocs/Detectron2/code/train_net_backup.py'
    configFileName = '/model/gaocs/Detectron2/configs/keypoints_rcnn_R_50_FPN_1x_backup.yaml'
    JsonName = '/model/gaocs/Detectron2/json/keypoints_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, NetworkConfig, scale, quality, label_config, transform_config)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    temp = label_config.split('/')
    if len(temp) == 1:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+label_config+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    elif len(temp) == 2:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    compress_para = 'cd /model/gaocs/Detectron2/code/; \
                    python3 /model/gaocs/Detectron2/code/train_net.py \
                    --config-file /model/gaocs/Detectron2/configs/keypoints_rcnn_R_50_FPN_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /model/gaocs/Detectron2/model_final_04e291_KeypointsRCNN_R50_FPN.pkl ' \
                    + '>' + log_name
    print(compress_para)
    os.system(compress_para)

def Keypoints_Res101_FPN(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config):
    if not os.path.exists(LogPath):
        os.makedirs(LogPath)
    NetworkConfig = 'Keypoints_Res101_FPN'
    DatasetName = DatasetNamePrefix+'_'+NetworkConfig+'_' + QualityConfig
    TrainFileName = '/model/gaocs/Detectron2/code/train_net_backup.py'
    configFileName = '/model/gaocs/Detectron2/configs/keypoint_rcnn_R_101_FPN_3x_backup.yaml'
    JsonName = '/model/gaocs/Detectron2/json/keypoints_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, NetworkConfig, scale, quality, label_config, transform_config)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)
    temp = label_config.split('/')
    if len(temp) == 1:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+label_config+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    elif len(temp) == 2:
        log_name = LogPath + DatasetNamePrefix + '_' +transform_config+'_'+temp[0]+'_'+temp[1]+'_'+NetworkConfig+'_'+QualityConfig+'.log 2>&1'
    compress_para = 'cd /model/gaocs/Detectron2/code/; \
                    python3 /model/gaocs/Detectron2/code/train_net.py \
                    --config-file /model/gaocs/Detectron2/configs/keypoint_rcnn_R_101_FPN_3x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /model/gaocs/Detectron2/model_final_997cc7_KeypointsRCNN_R101_FPN.pkl ' \
                    + '>' + log_name
    print(compress_para)
    os.system(compress_para)

# def eval_all(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, alpha, quality, mask_type, transform_config):
def eval_all(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality):
    Faster_Res50_C4(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)
    # Mask_Res50_C4(DatasetNamePrefix, DatasetNameSuffix, DatasetPath, LogPath, scale, quality, label_config, transform_config)
    # Keypoints_Res50_FPN(DatasetNamePrefix, DatasetNameSuffix, DatasetPath, LogPath, scale, quality, label_config, transform_config)
    # Faster_Res50_FPN(DatasetNamePrefix, DatasetNameSuffix, DatasetPath, LogPath, scale, quality, label_config, transform_config)
    # Mask_Res50_FPN(DatasetNamePrefix, DatasetNameSuffix, DatasetPath, LogPath, scale, quality, label_config, transform_config)
    # Keypoints_Res101_FPN(DatasetNamePrefix, DatasetNameSuffix, DatasetPath, LogPath, scale, quality, label_config, transform_config)


def main(args):
    configs = []
    configs.append(args.config)

    

    data_root = "/home/gaocs/projects/ICM-DIICM/Data"
    mask_type = 'inferred'
    mask_network = 'MaskRCNN_Res101_FPN_0.5'
    # transform_config = 'transformed_compressed'
    processing_config = 'transformed'
    arch = 'cheng2020_anchor'

    alpha_all = [0.2]
    quality_all = [100]

    for alpha in alpha_all:
        for quality in quality_all:
            # QualityConfig = alpha +'_' + quality 
            DatasetNamePrefix = 'coco_minVal2014_5000'

            if processing_config == 'transformed':
                DatasetPath = f"{data_root}/{processing_config}/{mask_type}/{mask_network}/1.0_{alpha}"
                LogPath = f"{data_root}/mAP/{processing_config}/{mask_type}/{mask_network}/1.0_{alpha}"
            elif processing_config == 'compressed':
                DatasetPath = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/quality{quality}_image"
                LogPath = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}/quality{quality}"
            elif processing_config == 'transformed_compressed':
                DatasetPath = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}_image"
                LogPath = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}"

            eval_all(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)


    # for JPEG compressed images
    # scale_all = ['1.0_1.0']
    # quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70', '80_80', '90_90']
    # label_config = 'label'
    # transform_config = 'jpeg_compressed'
    # for scale in scale_all:
    #     for quality in quality_all:
    #         QualityConfig = quality 
    #         DatasetNamePrefix = 'coco_minVal2014_5000'

    #         # transformed_compressed
    #         DatasetPath = '/data/gaocs/Understanding_Detection/compressed/' + quality.split('_')[0] + '/'
    #         LogPath = '/data/gaocs/Understanding_Detection/mAP/compressed/'
    #         eval_all(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config) 


    # quality1_all = ['100']
    # quality2_all = ['100']
    # method = 'org_org'
    # method_prefix = method.split('_')[0] + '_' + method.split('_')[1]

    # for scale in scale_all:
    #     for quality in quality_all:
    #         QualityConfig = scale +'_' + quality 
    #         DatasetNamePrefix = 'coco_minVal2014_5000'
    #         # transformed
    #         DatasetPath = '/data/gaocs/Understanding_Detection/minVal2014'
    #         LogPath = '/data/gaocs/Understanding_Detection/mAP/' 
    #         # transformed_compressed
    #         # DatasetPath = '/data/gaocs/Understanding_Detection/' + transform_config + '/' + label_config + '/' + scale + '/' + quality + '/'
    #         # LogPath = '/data/gaocs/Understanding_Detection/mAP/' + transform_config + '/'  + label_config + '/'
    #         eval_all(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, scale, quality, label_config, transform_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for configration')
    # parser.add_argument('--DatasetName', '-dn', help='dataset_name')
    # parser.add_argument('--DatasetPath', '-dp', help='datase_path')
    # parser.add_argument('--JsonName', '-jn', help='json_name')
    parser.add_argument('--config', '-c', help='configration')
    args = parser.parse_args()
    main(args)