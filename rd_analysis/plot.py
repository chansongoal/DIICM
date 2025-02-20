import enum
import os 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import gca
# plt.rcParams['text.usetex'] = True

lowbound_L = 0.1
lowbound_R = 0.115
upbound = 1.2
linewidth = 1.5
markersize = 4


def get_bpp_compressed(quality_all):
    bpp_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/bitrate/'
    bpp_all = []
    for idx, quality in enumerate(quality_all):
        bpp_name = bpp_path + 'bpp_compressed_' + quality + '.txt'
        file = open(bpp_name)
        log = file.readlines()
        bpp = float(log[-1])
        bpp_all.append(bpp)
        file.close()
    return bpp_all

def get_mAP_compressed(quality_all, task):
    mAP_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/mAP/compressed/'
    mAP_all = []
    for idx, quality in enumerate(quality_all):
        if task == 'detection':
            mAP_name = mAP_path + 'coco_minVal2014_5000_jpeg_compressed_label_Faster_Res50_C4_' + quality + '.log'
        elif task == 'segmentation':
            mAP_name = mAP_path + 'coco_minVal2014_5000_Mask_Res50_C4_quality_' + quality + '.log'
        elif task == 'keypoints':
            mAP_name = mAP_path + 'coco_minVal2014_5000_jpeg_compressed_label_Keypoints_Res50_FPN_' + quality + '.log'
        file = open(mAP_name)
        log = file.readlines()
        if task=='detection' or task=='segmentation':
            mAP = float(log[-1].split(',')[-6].split()[-1])
        if task=='keypoints':
            mAP = float(log[-1].split(',')[-5].split()[-1])
        mAP_all.append(mAP)
        file.close()
    return mAP_all

def get_psnr_compressed(quality_all):
    psnr_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/psnr/compressed/'
    psnr_fore_all = []
    psnr_back_all = []
    psnr_overall_all = []
    for idx, quality in enumerate(quality_all):
        psnr_name = psnr_path + 'psnr_compressed_' + quality + '.log'
        file = open(psnr_name)
        log = file.readlines()
        psnr_fore, psnr_back, psnr_overall = float(log[-1].split()[-3]), float(log[-1].split()[-2]), float(log[-1].split()[-1])
        psnr_fore_all.append(psnr_fore)
        psnr_back_all.append(psnr_back)
        psnr_overall_all.append(psnr_overall)
        file.close()
    return psnr_fore_all, psnr_back_all, psnr_overall_all

def get_bpp(transform_config, label_config, scale, quality_all):
    bpp_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/bitrate/'
    bpp_all = []
    for idx, quality in enumerate(quality_all):
        bpp_name = bpp_path + 'bpp_'+ transform_config + '_' + label_config + '_' + scale + '_' + quality + '.txt'
        file = open(bpp_name)
        # print(bpp_name)
        log = file.readlines()
        bpp = float(log[-1])
        bpp_all.append(bpp)
        file.close()
    return bpp_all
    
def get_mAP(transform_config, label_config, scale, quality_all, task):
    mAP_all = []
    for idx, quality in enumerate(quality_all):
        mAP_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/mAP/' + transform_config + '/' + label_config + '/'
        if 'inferenced' in label_config:
            mAP_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/mAP/' + transform_config + '/' + label_config[:10] + '/' + label_config[11:] + '/'
        if task == 'detection':
            mAP_name = mAP_path + 'coco_minVal2014_5000_' + transform_config + '_' + label_config + '_Faster_Res50_C4_' + scale + '_' + quality + '.log'
        elif task == 'segmentation':
            mAP_name = mAP_path + 'coco_minVal2014_5000_' + transform_config + '_' + label_config + '_Mask_Res50_C4_' + scale + '_' + quality + '.log'
        elif task == 'keypoints':
            mAP_name = mAP_path + 'coco_minVal2014_5000_' + transform_config + '_' + label_config + '_Keypoints_Res50_FPN_' + scale + '_' + quality + '.log'
        file = open(mAP_name)
        log = file.readlines()
        if task == 'detection' or task == 'segmentation':
            mAP = float(log[-1].split(',')[-6].split()[-1])
        elif task == 'keypoints':
            mAP = float(log[-1].split(',')[-5].split()[-1])
        mAP_all.append(mAP)
        file.close()
    return mAP_all

def get_psnr(transform_config, label_config, scale, quality_all):
    psnr_fore_all = []
    psnr_back_all = []
    psnr_overall_all = []
    for idx, quality in enumerate(quality_all):
        psnr_path = r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Data/psnr/' + transform_config + '/' + label_config + '/'
        psnr_name = psnr_path + 'psnr_' + transform_config + '_' + label_config + '_' + scale + '_' + quality + '.log'
        if 'inferenced' in label_config:
            psnr_name = psnr_path + 'psnr_' + transform_config + '_' + label_config + '_MaskRCNN_Res101_FPN_0.5_' + scale + '_' + quality + '.log'
        file = open(psnr_name)
        log = file.readlines()
        psnr_fore, psnr_back, psnr_overall = float(log[-1].split()[-3]), float(log[-1].split()[-2]), float(log[-1].split()[-1])
        psnr_fore_all.append(psnr_fore)
        psnr_back_all.append(psnr_back)
        psnr_overall_all.append(psnr_overall)
        file.close()
    return psnr_fore_all, psnr_back_all, psnr_overall_all

def plot_bpp_mAP():
    task = 'keypoints'
    scale = '1.0_0.8'
    pdfName = '/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Paper/DGIICM-TIP/images/bpp_mAP/bpp_mAP_' + task + '_' + scale + '.pdf'
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    # quality_all = ['70_70', '60_60', '50_50', '40_40', '30_30', '20_20', '10_10', '1_1']

    bpp_jpeg_all = get_bpp_compressed(quality_all)
    mAP_jpeg_all = get_mAP_compressed(quality_all, task)
    print(bpp_jpeg_all)
    print(mAP_jpeg_all)

    transform_config = 'transformed_compressed'
    label_config = 'label'
    bpp_label_all = get_bpp(transform_config, label_config, scale, quality_all)
    mAP_label_all = get_mAP(transform_config, label_config, scale, quality_all, task)
    # print(bpp_label_all)
    # print(mAP_label_all)

    label_config = 'inferenced_MaskRCNN_Res101_FPN_0.5'
    bpp_inference_all = get_bpp(transform_config, label_config, scale, quality_all)
    mAP_inference_all = get_mAP(transform_config, label_config, scale, quality_all, task)
    # print(bpp_inference_all)
    # print(mAP_inference_all)

    # plt.plot(bpp_jpeg_all, mAP_jpeg_all, 'co-', label=r"Baseline")
    # plt.plot(bpp_inference_all, mAP_inference_all, 'mo-', label=r"DIICM")
    # plt.plot(bpp_label_all, mAP_label_all, 'bo-', label=r"DIICM-GT")


    #inferred_all_detection C4
    bpp_all = [0.16980591864190625, 0.2535003047294002, 0.35782583417217606, 0.4530488161826729, 0.5361500502300521, 0.6768275349511922, 0.7860528175622425, 0.9077020312209303, 1.0632481252833292, 1.2888761251570147]
    mAP_all = [0.6707, 9.915, 21.622, 26.6619, 28.5734, 30.2107, 31.1511, 32.0242, 32.6979, 33.6007]
    plt.plot(bpp_all, mAP_all, 'ro-', label=r"DIICM")
    # print(bpp_all)
    # print(mAP_all)

    #inferred_all_segmentation  C4
    # bpp_all = [0.16980591864190625, 0.2535003047294002, 0.35782583417217606, 0.4530488161826729, 0.5361500502300521,  0.6768275349511922, 0.7860528175622425, 0.9077020312209303,  1.0632481252833292, 1.2888761251570147]
    # mAP_all = [0.6586, 9.2795, 19.574, 23.9554, 25.5364,  26.8324, 27.7206, 28.6115,  29.3297, 29.9275]
    # plt.plot(bpp_all, mAP_all, 'ro-', label=r"DIICM")
    # print(bpp_all)
    # print(mAP_all)

    #inferred_all_keypoints Res50-FPN
    # bpp_all = [0.16980591864190625, 0.2535003047294002, 0.35782583417217606, 0.4530488161826729, 0.5361500502300521, 0.6194830429824971, 0.7860528175622425, 0.9077020312209303, 1.0632481252833292, 1.2888761251570147]
    # mAP_all = [0.495, 15.9291, 41.9367, 49.8164, 52.9922, 54.4147, 56.4356, 57.8847, 59.0133, 60.4988]
    # plt.plot(bpp_all, mAP_all, 'ro-', label=r"DIICM")

    fs = 24
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('mAP', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    # plt.legend(loc=4, prop=font)
    plt.grid()
    plt.tight_layout()
    # pdfName = '/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Paper/DGIICM-TIP/images/bpp_mAP/bpp_mAP_' + task + '_' + scale + '.pdf'
    pdfName = '/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Paper/DGIICM-TIP/images/bpp_mAP/bpp_mAP_' + task + '_merged' + '.pdf'
    # plt.savefig(pdfName, dpi=600, format='pdf')
    plt.clf()

def plot_bpp_psnr(config, alpha, beta):
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    bpp_all = get_bpp_compressed(quality_all)
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr_compressed(quality_all)
    if config=='fore': plt.plot(bpp_all, psnr_fore_all, 'co-', label='Baseline')
    if config=='back': plt.plot(bpp_all, psnr_back_all, 'co-', label='Baseline')
    if config=='overall': plt.plot(bpp_all, psnr_overall_all, 'co-', label='Baseline')

    transform_config = 'transformed_compressed'
    scale = '1.0_'+alpha
    label_config = 'inferenced_MaskRCNN_Res101_FPN_0.5'
    bpp_all = get_bpp(transform_config, label_config, scale, quality_all)  
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr(transform_config, 'inferenced_label', scale, quality_all)
    if config=='fore': plt.plot(bpp_all, psnr_fore_all, 'bo-', label='DIICM')
    if config=='back': plt.plot(bpp_all, psnr_back_all, 'bo-', label='DIICM')
    if config=='overall': plt.plot(bpp_all, psnr_overall_all, 'bo-', label='DIICM')

    transform_config = 'transformed_compressed_inv'
    scale = '1.0_'+beta
    label_config = 'inferenced_MaskRCNN_Res101_FPN_0.5'
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr(transform_config, 'inferenced_label', scale, quality_all)
    if config=='fore': plt.plot(bpp_all, psnr_fore_all, 'mo-', label='DIICM-Inv')
    if config=='back': plt.plot(bpp_all, psnr_back_all, 'mo-', label='DIICM-Inv')
    if config=='overall': plt.plot(bpp_all, psnr_overall_all, 'mo-', label='DIICM-Inv')

    # plt.ylim([15, 35])

    fs = 24
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('PSNR', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':20,}
    if alpha=='0.8' and config=='overall': plt.legend(loc=4, prop=font)
    plt.grid()
    plt.tight_layout()
    pdfName = '/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Paper/DGIICM-TIP/images/bpp_psnr/bpp_psnr_' + config + '_' + alpha + '.pdf'
    plt.savefig(pdfName, dpi=600, format='pdf')
    plt.clf()

def plot_bpp_psnr_allInOne(alpha, beta):
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    bpp_all = get_bpp_compressed(quality_all)
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr_compressed(quality_all)
    plt.plot(bpp_all, psnr_fore_all, 'c.--', label='baseline')
    plt.plot(bpp_all, psnr_back_all, 'c.:', label='baseline')
    plt.plot(bpp_all, psnr_overall_all, 'c.-', label='baseline')

    transform_config = 'transformed_compressed'
    scale = '1.0_'+alpha
    label_config = 'inferenced_MaskRCNN_Res101_FPN_0.5'
    bpp_all = get_bpp(transform_config, label_config, scale, quality_all)  
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr(transform_config, 'inferenced_label', scale, quality_all)
    plt.plot(bpp_all, psnr_fore_all, 'b.--', label='Transform')
    plt.plot(bpp_all, psnr_back_all, 'b.:', label='Transform')
    plt.plot(bpp_all, psnr_overall_all, 'b.-', label='Transform')

    transform_config = 'transformed_compressed_inv'
    scale = '1.0_'+beta
    label_config = 'inferenced_MaskRCNN_Res101_FPN_0.5'
    psnr_fore_all, psnr_back_all, psnr_overall_all = get_psnr(transform_config, 'inferenced_label', scale, quality_all)
    plt.plot(bpp_all, psnr_fore_all, 'm.--', label='Inv_Transform')
    plt.plot(bpp_all, psnr_back_all, 'm.:', label='Inv_Transform')
    plt.plot(bpp_all, psnr_overall_all, 'm.-', label='Inv_Transform')

    # plt.ylim([15, 35])

    fs = 24
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('PSNR', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':20,}
    # if alpha=='0.8': plt.legend(loc=4, prop=font)
    plt.grid()
    plt.tight_layout()
    figname = '/Users/changsheng/Library/CloudStorage/OneDrive-USTC/PaperSubmit/understanding_detection/images/bpp_psnr_allInOne_'+alpha+'.pdf'
    plt.savefig(figname, dpi=600, format='pdf')
    plt.clf()

def plot_bpp_IoU():
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    bpp_all = [1.39, 1.16, 1.00, 0.86, 0.72, 0.56, 0.36, 0.19]  
    # IoU_all = [37.67, 36.73, 35.80, 34.82, 33.02, 28.55, 16.25, 2.52]   #recall_0.8-1.0
    IoU_all = [13.76, 13.33, 12.78, 12.14, 11.04, 8.77, 4.57, 2.17]   #precision_0.8-1.0
    # IoU_all = [11.71, 12.21, 12.52, 13.18, 14.14, 16.35, 25.75, 68.29]   #recall_0-0.2
    # IoU_all = [52.75, 53.12, 53.66, 54.53, 55.94, 59.71, 71.93, 85.69]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'co-', label=r"Baseline")

    bpp_all = [0.85, 0.71, 0.62, 0.54, 0.45, 0.36, 0.25, 0.17]
    # IoU_all = [33.98, 33.44, 32.82, 32.07, 30.85, 27.69, 17.45, 2.89]   #recall_0.8-1.0
    IoU_all = [15.68, 15.11, 14.52, 13.72, 12.62, 10.33, 6.32, 2.83]   #precision_0.8-1.0
    # IoU_all = [18.98, 19.42, 19.53, 20.06, 20.60, 21.98, 28.50, 65.47]   #recall_0-0.2
    # IoU_all = [45.73, 46.33, 47.05, 48.13, 49.70, 52.52, 62.52, 79.22]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'ro-', label=r"DIICM, $\alpha$=0.2")

    bpp_all = [1.09, 0.91, 0.79, 0.68, 0.57, 0.44, 0.29, 0.17]
    # IoU_all = [36.02, 35.57, 34.71, 33.82, 32.51, 28.43, 16.95, 2.57]   #recall_0.8-1.0
    IoU_all = [14.73, 14.32, 13.74, 12.91, 11.89, 9.38, 5.40, 2.56]   #precision_0.8-1.0
    # IoU_all = [14.63, 15.00, 15.34, 15.84, 16.64, 18.36, 26.37, 68.38]   #recall_0-0.2
    # IoU_all = [48.96, 49.55, 50.10, 51.19, 52.72, 56.50, 66.98, 80.47]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'go-', label=r"DIICM, $\alpha$=0.5")

    bpp_all = [1.29, 1.06, 0.92, 0.80, 0.66, 0.51, 0.33, 0.18]
    # IoU_all = [37.20, 36.55, 35.56, 34.71, 33.09, 28.64, 16.22, 2.46]   #recall_0.8-1.0
    IoU_all = [14.10, 13.76, 13.17, 12.38, 11.36, 8.89, 4.73, 2.31]   #precision_0.8-1.0
    # IoU_all = [12.48, 12.77, 13.20, 13.74, 14.56, 16.70, 25.55, 69.19]   #recall_0-0.2
    # IoU_all = [51.22, 51.87, 52.49, 53.42, 54.92, 58.88, 70.57, 83.75]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'bo-', label=r"DIICM, $\alpha$=0.8")

    # plt.ylim([0, 17])

    fs = 18
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('IoU Precision', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    # plt.legend(loc=1, prop=font)
    plt.grid()
    plt.tight_layout()
    # plt.savefig(r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/PhD/ustcthesis/figures/sec4/images/bpp_IoU_recall_low.pdf', dpi=600, format='pdf')
    pdfName = '/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/ResearchOngoing/ICM-DomainGap/Paper/DGIICM-TIP/images/bpp_IoU/bpp_IoU_precision_high.pdf'
    plt.savefig(pdfName, dpi=600, format='pdf')
    plt.clf()

def plot_bpp_IoU_FPN():
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    bpp_all = [1.39, 1.16, 1.00, 0.86, 0.72, 0.56, 0.36, 0.19]  
    # IoU_all = [43.01, 41.90, 40.62, 39.39, 36.90, 31.38, 16.77, 1.70]   #recall_0.8-1.0
    # IoU_all = [19.38, 19.11, 18.70, 18.12, 17.13, 14.93, 9.96, 2.70]   #precision_0.8-1.0
    # IoU_all = [10.87, 11.70, 12.18, 13.17, 14.62, 18.11, 34.02, 81.89]   #recall_0-0.2
    IoU_all = [45.41, 45.42, 45.35, 45.65, 46.12, 47.83, 55.79, 82.61]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'co-', label=r"JPEG")

    bpp_all = [0.85, 0.71, 0.62, 0.54, 0.45, 0.36, 0.25, 0.17]
    # IoU_all = [37.45, 36.72, 35.96, 34.83, 33.15, 29.25, 17.01, 1.64]
    # IoU_all = [20.92, 20.48, 20.09, 19.54, 18.77, 16.99, 12.39, 2.98]
    # IoU_all = [19.45, 20.01, 20.45, 21.14, 22.30, 25.05, 37.71, 82.17]   #recall_0-0.2
    IoU_all = [40.18, 40.12, 40.20, 40.44, 40.84, 41.39, 46.66, 78.32]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'ro-', label=r"transform_0.2")

    bpp_all = [1.09, 0.91, 0.79, 0.68, 0.57, 0.44, 0.29, 0.17]
    # IoU_all = [40.67, 39.74, 38.83, 37.66, 35.58, 30.73, 17.20, 1.53]
    # IoU_all = [20.33, 19.86, 19.51, 18.79, 17.84, 15.68, 11.36, 2.64]
    # IoU_all = [14.06, 14.57, 15.20, 15.97, 17.19, 20.28, 35.61, 82.95]   #recall_0-0.2
    IoU_all = [42.65, 42.64, 42.81, 43.26, 43.82, 45.01, 49.80, 78.52]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'go-', label=r"transform_0.5")

    bpp_all = [1.29, 1.06, 0.92, 0.80, 0.66, 0.51, 0.33, 0.18]
    # IoU_all = [42.68, 41.53, 40.40, 39.12, 36.74, 31.21, 16.74, 1.52]
    # IoU_all = [19.80, 19.38, 18.98, 18.34, 17.34, 15.07, 10.27, 2.41]
    # IoU_all = [11.57, 12.05, 12.78, 13.60, 14.93, 18.56, 34.59, 82.64]   #recall_0-0.2
    IoU_all = [44.45, 44.37, 44.50, 44.76, 45.33, 47.17, 53.97, 81.14]     #precision_0-0.2
    plt.plot(bpp_all, IoU_all, 'bo-', label=r"transform_0.8")

    # plt.ylim([0, 17])

    fs = 18
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('IoU Precision', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.legend(loc=1, prop=font)
    plt.grid()
    plt.tight_layout()
    plt.savefig(r'/Users/changsheng/Library/CloudStorage/OneDrive-USTC/PaperSubmit/understanding_detection/images/bpp_IoU_precision_low_FPN.pdf', dpi=600, format='pdf')
    plt.clf()

def plot_compress_mAP():
    #MaskRCNN-R50-C4
    # levels = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '1']
    # mAp = [32.15,  31.64, 30.74, 30.03, 29.34, 28.50, 28.50, 25.79, 20.55, 9.32, 0.60]

    #FasterRCNN-R50-C4
    levels = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '1']
    mAp = [35.6770, 35.3251,  34.4216, 33.6238, 32.7431, 31.7012, 30.7382, 28.5960, 22.7924, 9.9858, 0.6359]

    #KeypointsRCNN-R50-C4
    # levels = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '1']
    # mAp = [64.0, 63.3395, 61.7398, 60.7341, 59.0207, 57.7692, 55.9636, 52.3528, 43.6045, 17.0083, 0.4234]

    fs = 24
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('mAP', font)
    plt.xlabel('quality level', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    plt.plot(levels, mAp, 'bo-')

    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    # plt.legend(loc=1, prop=font)
    plt.grid()
    plt.tight_layout()
    plt.savefig(r'/Users/changshenggao/Library/CloudStorage/OneDrive-USTC/PhD/ustcthesis/figures/sec4/images/levels_mAp_detection.pdf', dpi=600, format='pdf')
    print('done')
    plt.clf()

if __name__ == "__main__":
    plot_bpp_mAP()
    # alpha, beta = '0.8', '1.25'
    # plot_bpp_psnr('fore', alpha, beta)
    # plot_bpp_psnr('back', alpha, beta)
    # plot_bpp_psnr('overall', alpha, beta)
    # plot_bpp_psnr_allInOne(alpha, beta)
    # plot_bpp_IoU()
    # plot_bpp_IoU_FPN()

    # plot_compress_mAP()
    # print('done')