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
    NetworkConfig = 'Faster_Res50_C4'
    DatasetName = f"{DatasetNamePrefix}_{NetworkConfig}_1.0_{alpha}_quality{quality}"
    TrainFileName = '/ghome/gaocs/DIICM/machines/detectron2/train_net_backup.py'
    configFileName = '/ghome/gaocs/DIICM/machines/detectron2/configs/faster_rcnn_R_50_C4_1x_backup.yaml'
    JsonName = '/gdata/gaocs/dataset/COCO/json/instances_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, LogPath, alpha, quality)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)

    log_name = f"{LogPath}/{DatasetNamePrefix}_{processing_config}_{mask_type}_{mask_network}_{NetworkConfig}_1.0_{alpha}_quality{quality}.txt 2>&1"
    eval_para = 'cd /ghome/gaocs/DIICM/machines/detectron2/; \
                    python3 train_net.py \
                    --config-file ./configs/faster_rcnn_R_50_C4_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /gdata/gaocs/pretrained_models/detectron2/model_zoo/model_final_721ade_FasterRCNN_R50_C4.pkl ' \
                    + '>' + log_name
    print(eval_para)
    os.system(eval_para)


def Mask_Res50_C4(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality):
    NetworkConfig = 'Mask_Res50_C4'
    DatasetName = f"{DatasetNamePrefix}_{NetworkConfig}_1.0_{alpha}_quality{quality}"
    TrainFileName = '/ghome/gaocs/DIICM/machines/detectron2/train_net_backup.py'
    configFileName = '/ghome/gaocs/DIICM/machines/detectron2/configs/mask_rcnn_R_50_C4_1x_backup.yaml'
    JsonName = '/gdata/gaocs/dataset/COCO/json/instances_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, LogPath, alpha, quality)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)

    log_name = f"{LogPath}/{DatasetNamePrefix}_{processing_config}_{mask_type}_{mask_network}_{NetworkConfig}_1.0_{alpha}_quality{quality}.txt 2>&1"
    eval_para = 'cd /ghome/gaocs/DIICM/machines/detectron2/; \
                    python3 train_net.py \
                    --config-file ./configs/mask_rcnn_R_50_C4_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /gdata/gaocs/pretrained_models/detectron2/model_zoo/model_final_9243eb_MaskRCNN_R50_C4.pkl ' \
                    + '>' + log_name
    print(eval_para)
    os.system(eval_para)


def Keypoints_Res50_FPN(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality):
    NetworkConfig = 'Keypoints_Res50_FPN'
    DatasetName = f"{DatasetNamePrefix}_{NetworkConfig}_1.0_{alpha}_quality{quality}"
    TrainFileName = '/ghome/gaocs/DIICM/machines/detectron2/train_net_backup.py'
    configFileName = '/ghome/gaocs/DIICM/machines/detectron2/configs/keypoints_rcnn_R_50_FPN_1x_backup.yaml'
    JsonName = '/gdata/gaocs/dataset/COCO/json/keypoints_minVal2014_png.json'
    ConfigModification(configFileName, DatasetName, LogPath, alpha, quality)
    train_net_Modification(TrainFileName, DatasetName, JsonName, DatasetPath)

    log_name = f"{LogPath}/{DatasetNamePrefix}_{processing_config}_{mask_type}_{mask_network}_{NetworkConfig}_1.0_{alpha}_quality{quality}.txt 2>&1"
    eval_para = 'cd /ghome/gaocs/DIICM/machines/detectron2/; \
                    python3 train_net.py \
                    --config-file ./configs/keypoints_rcnn_R_50_FPN_1x.yaml \
                    --eval-only \
                    MODEL.WEIGHTS /gdata/gaocs/pretrained_models/detectron2/model_zoo/model_final_04e291_KeypointsRCNN_R50_FPN.pkl ' \
                    + '>' + log_name
    print(eval_para)
    os.system(eval_para)


# def eval_all(DatasetNamePrefix, QualityConfig, DatasetPath, LogPath, alpha, quality, mask_type, transform_config):
def eval_all(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality):
    Faster_Res50_C4(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)
    Mask_Res50_C4(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)
    Keypoints_Res50_FPN(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)


def main(args):
    data_root = args.data_root; print('data_root: ', data_root)
    mask_type = args.mask_type; print('mask_type: ', mask_type)
    mask_network = args.mask_network; print('mask_network: ', mask_network)
    processing_config = args.processing_config; print('processing_config: ', processing_config)
    arch = args.arch; print('arch: ', arch)

    alpha_all = args.alpha_all; print('alpha_all: ', alpha_all)
    quality_all = args.quality_all; print('quality_all: ', quality_all)

    for alpha in alpha_all:
        for quality in quality_all:
            DatasetNamePrefix = 'coco_minVal2014_5000'

            if processing_config == 'transformed':
                DatasetPath = f"{data_root}/{processing_config}/{mask_type}/{mask_network}/1.0_{alpha}"
                LogPath = f"{data_root}/mAP/{processing_config}/{mask_type}/{mask_network}/1.0_{alpha}"
            elif processing_config == 'compressed':
                DatasetPath = f"{data_root}/{processing_config}/{arch}/quality{quality}/image"
                LogPath = f"{data_root}/mAP/{processing_config}/{arch}/quality{quality}"
            elif processing_config == 'transformed_compressed':
                DatasetPath = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/image"
                LogPath = f"{data_root}/mAP/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}"

            if not os.path.exists(LogPath): os.makedirs(LogPath, exist_ok=True)

            eval_all(DatasetNamePrefix, DatasetPath, LogPath, mask_type, mask_network, processing_config, alpha, quality)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperprior Evaluation Pipeline")
    parser.add_argument('--data_root', type=str, default="/gdata1/gaocs/Data_DIICM", help='Root directory of the dataset')
    parser.add_argument('--mask_type', type=str, choices=['inferred', 'label', 'original'], default='inferred', help='Type of mask used')
    parser.add_argument('--mask_network', type=str, default='MaskRCNN_Res101_FPN_0.5', help='Mask network used (if applicable)')
    parser.add_argument('--processing_config', type=str, choices=['transformed', 'compressed', 'transformed_compressed'], default='transformed', help='Processing configuration')
    parser.add_argument('--arch', type=str, default='cheng2020-anchor', help='Model architecture')
    parser.add_argument('--alpha_all', type=float, nargs='+', default=[0.2, 0.5, 0.8], help='List of alpha values')
    parser.add_argument('--quality_all', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='List of quality levels')
    
    args = parser.parse_args()
    main(args)