import os 
import re
import numpy as np 
import matplotlib.pyplot as plt


def get_bpp_mAP(arch, network_idx, alpha_idx):
    # bpp	mAP	mAP	mAP	bpp	mAP	mAP	mAP	bpp	mAP	mAP	mAP	bpp	mAP	mAP	mAP (compressed, 0.2, 0.5, 0.8) (object detection, instance segmentation, keypoints detection)
    data_jpeg = [[0.19,	0.64,	0.60,	0.42,	0.17,	0.67,	0.66,	0.50,	0.17,	0.64,	0.63,	0.50,	0.18,	0.66,	0.61,	0.50],
                 [0.36,	9.99,	9.32,	17.01,	0.25,	9.92,	9.28,	15.93,	0.29,	10.22,	9.45,	16.49,	0.33,	10.17,	9.24,	16.62],
                 [0.56,	22.79,	20.55,	43.60,	0.36,	21.62,	19.57,	41.94,	0.44,	22.69,	20.36,	43.09,	0.51,	22.99,	20.67,	43.59],
                 [0.72,	28.60,	25.79,	52.35,	0.45,	26.66,	23.96,	49.82,	0.57,	28.07,	25.19,	51.42,	0.66,	28.57,	25.83,	52.15],
                 [0.86,	30.74,	27.53,	55.96,	0.54,	28.57,	25.54,	52.99,	0.68,	30.21,	26.83,	54.62,	0.80,	30.86,	27.59,	55.74],
                 [1.00,	31.70,	28.50,	57.77,	0.62,	29.43,	26.20,	54.41,	0.79,	31.15,	27.72,	56.44,	0.92,	31.83,	28.46,	57.38],
                 [1.16,	32.74,	29.34,	59.02,	0.71,	30.33,	26.99,	55.80,	0.91,	32.02,	28.61,	57.88,	1.06,	32.70,	29.33,	59.01],
                 [1.39,	33.62,	30.03,	60.73,	0.85,	30.88,	27.47,	57.25,	1.09,	32.88,	29.30,	58.99,	1.29,	33.60,	29.93,	60.50]]
    data_cheng2020 = [[0.16,	22.44,	19.94,	44.89,	0.08,	21.12,	18.66,	42.24,	0.10,	22.07,	19.54,	43.61,	0.14,	22.45,	19.93,	44.40],
                      [0.24,	25.81,	23.02,	51.14,	0.12,	23.89,	21.11,	47.83,	0.15,	25.36,	22.25,	49.78,	0.20,	25.80,	22.96,	50.84],
                      [0.33,	28.39,	25.43,	55.15,	0.16,	25.76,	22.93,	51.91,	0.22,	27.76,	24.72,	53.69,	0.29,	28.48,	25.33,	55.03],
                      [0.53,	31.13,	27.89,	58.87,	0.25,	28.04,	25.17,	54.92,	0.36,	30.45,	27.18,	57.25,	0.47,	31.02,	27.80,	58.44],
                      [0.73,	32.99,	29.58,	60.95,	0.36,	29.55,	26.43,	56.95,	0.52,	31.91,	28.42,	59.22,	0.65,	32.82,	29.32,	60.75],
                      [0.97,	34.01,	30.46,	62.31,	0.51,	30.66,	27.53,	58.14,	0.71,	32.92,	29.24,	60.24,	0.88,	33.84,	30.33,	61.65]]
    data_vtm = [[0.03,	6.00,	5.13,	10.14,	0.02,	6.52,	5.92,	9.61,	0.02,	6.05,	5.52,	9.39,	0.03,	6.02,	5.23,	10.29],
                [0.06,	11.60,	10.21,	22.51,	0.03,	12.52,	11.10,	23.16,	0.04,	12.13,	10.64,	22.81,	0.05,	11.99,	10.39,	23.02],
                [0.13,	20.07,	17.67,	39.88,	0.07,	19.97,	17.70,	39.30,	0.08,	20.30,	17.78,	40.23,	0.11,	20.11,	17.82,	40.42],
                [0.29,	27.06,	24.18,	52.79,	0.14,	25.56,	22.54,	50.37,	0.18,	26.92,	23.62,	52.11,	0.25,	27.27,	24.47,	52.81],
                [0.58,	31.75,	28.58,	59.24,	0.27,	29.15,	26.00,	55.58,	0.38,	31.01,	27.61,	57.86,	0.51,	31.98,	28.47,	59.07],
                [1.09,	34.29,	30.70,	62.11,	0.53,	30.98,	27.79,	58.31,	0.75,	33.13,	29.44,	60.49,	0.96,	34.22,	30.49,	61.91]]
    
    if arch == 'JPEG': data = data_jpeg
    elif arch == 'Cheng2020': data = data_cheng2020
    elif arch == 'VVC': data = data_vtm

    data = np.asarray(data)
    bpp = data[:, alpha_idx*4]
    mAP = data[:, network_idx+alpha_idx*4]

    return bpp, mAP

def plot_figure(pdf_name, fs=24, fs_legend=24, loc='best'):
    # fs = 24
    font = {'family': 'Times New Roman', 'weight':'normal', 'size':fs,}
    plt.ylabel('mAP', font)
    plt.xlabel('bpp', font)
    plt.xticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)
    plt.yticks(fontname = 'Times New Roman', weight='ultralight', fontsize=fs)

    font_legend = {'family': 'Times New Roman', 'weight':'normal', 'size':fs_legend,}
    plt.legend(loc=loc, prop=font_legend)
    plt.grid()
    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf')
    plt.clf()

def plot_bpp_mAP():
    arch_all = ['JPEG', 'Cheng2020', 'VVC']

    for arch in arch_all:
        network_config_all = ['Faster_Res50_C4', 'Mask_Res50_C4', 'Keypoints_Res50_FPN']
        alpha_all = [arch, 0.2, 0.5, 0.8]; color_all = ['b', '#ff7f0e', '#2ca02c', '#d62728']   # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for network_idx, network_config in enumerate(network_config_all):
            for alpha_idx, alpha in enumerate(alpha_all):
                bpp, mAP = get_bpp_mAP(arch, network_idx+1, alpha_idx)   
                if alpha_idx == 0: label = alpha
                else: 
                    label = rf'$\alpha$={alpha}'
                plt.plot(bpp, mAP, color=color_all[alpha_idx], marker='o', linestyle='-', label=label)

            pdf_name = f"bpp_mAP_{arch}_{network_config}.pdf"
            plot_figure(pdf_name)

def plot_bpp_IoU_jpeg():
    quality_all = ['1_1', '10_10', '20_20', '30_30', '40_40', '50_50', '60_60', '70_70']#, '80_80', '90_90']
    # data for baseline
    bpp_baseline = [1.39, 1.16, 1.00, 0.86, 0.72, 0.56, 0.36, 0.19]  
    IoU_baseline = [[37.67, 36.73, 35.80, 34.82, 33.02, 28.55, 16.25, 2.52],   #recall_0.8-1.0
                    [13.76, 13.33, 12.78, 12.14, 11.04, 8.77, 4.57, 2.17],   #precision_0.8-1.0
                    [11.71, 12.21, 12.52, 13.18, 14.14, 16.35, 25.75, 68.29],   #recall_0-0.2
                    [52.75, 53.12, 53.66, 54.53, 55.94, 59.71, 71.93, 85.69]]     #precision_0-0.2
    
    # data for alpha=0.2
    bpp_2 = [0.85, 0.71, 0.62, 0.54, 0.45, 0.36, 0.25, 0.17]
    IoU_2 = [[33.98, 33.44, 32.82, 32.07, 30.85, 27.69, 17.45, 2.89],   #recall_0.8-1.0
               [15.68, 15.11, 14.52, 13.72, 12.62, 10.33, 6.32, 2.83],   #precision_0.8-1.0
               [18.98, 19.42, 19.53, 20.06, 20.60, 21.98, 28.50, 65.47],   #recall_0-0.2
               [45.73, 46.33, 47.05, 48.13, 49.70, 52.52, 62.52, 79.22]]     #precision_0-0.2
    
    # data for alpha=0.5
    bpp_5 = [1.09, 0.91, 0.79, 0.68, 0.57, 0.44, 0.29, 0.17]
    IoU_5 = [[36.02, 35.57, 34.71, 33.82, 32.51, 28.43, 16.95, 2.57],   #recall_0.8-1.0
               [14.73, 14.32, 13.74, 12.91, 11.89, 9.38, 5.40, 2.56],   #precision_0.8-1.0
               [14.63, 15.00, 15.34, 15.84, 16.64, 18.36, 26.37, 68.38],   #recall_0-0.2
               [48.96, 49.55, 50.10, 51.19, 52.72, 56.50, 66.98, 80.47]]     #precision_0-0.2
    
    # data for alpha=0.8
    bpp_8 = [1.29, 1.06, 0.92, 0.80, 0.66, 0.51, 0.33, 0.18]
    IoU_8 = [[37.20, 36.55, 35.56, 34.71, 33.09, 28.64, 16.22, 2.46],   #recall_0.8-1.0
               [14.10, 13.76, 13.17, 12.38, 11.36, 8.89, 4.73, 2.31],   #precision_0.8-1.0
               [12.48, 12.77, 13.20, 13.74, 14.56, 16.70, 25.55, 69.19],   #recall_0-0.2
               [51.22, 51.87, 52.49, 53.42, 54.92, 58.88, 70.57, 83.75]]     #precision_0-0.2
    

    config_all = ['recall_high', 'precision_high', 'recall_low', 'precision_low']
    for idx, config in enumerate(config_all):
        plt.plot(bpp_baseline, IoU_baseline[idx], color='b', marker='o', linestyle='-', label=r"JPEG")
        plt.plot(bpp_2, IoU_2[idx], color='#ff7f0e', marker='o', linestyle='-', label=r"$\alpha$=0.2")
        plt.plot(bpp_5, IoU_5[idx], color='#2ca02c', marker='o', linestyle='-', label=r"$\alpha$=0.5")
        plt.plot(bpp_8, IoU_8[idx], color='#d62728', marker='o', linestyle='-', label=r"$\alpha$=0.8")
        pdf_name = f"bpp_mIoU_jpeg_{config}.pdf"
        plot_figure(pdf_name)

def plot_bpp_psnr_jpeg():
    # compressed
    compressed = [[0.1945, 19.8065, 21.5378, 21.0429],
                  [0.3619, 23.9764, 26.1305, 25.4188],
                  [0.5565, 25.9494, 28.2311, 27.4296],
                  [0.7229, 27.1028, 29.3923, 28.5676],
                  [0.8635, 27.9271, 30.1964, 29.3731],
                  [1.0036, 28.6302, 30.8749, 30.0540],
                  [1.1628, 29.2762, 31.4850, 30.6766],
                  [1.3863, 30.3237, 32.4896, 31.7052]]

    # transformed_compressed, alpha=0.5, (bpp, fore_psnr, back_psnr, overall_psnr)
    transformed_compressed = [[0.1744, 19.0083, 17.5359, 17.9307],
                               [0.2915, 22.2667, 19.0162, 19.7526],
                               [0.4380, 23.5758, 19.3847, 20.2686],
                               [0.5666, 24.2677, 19.5418, 20.4988],
                               [0.6768, 24.7307, 19.6313, 20.6383],
                               [0.7861, 25.1023, 19.6996, 20.7396],
                               [0.9077, 25.4301, 19.7623, 20.8311],
                               [1.0900, 25.9053, 19.8256, 20.9426]]       
    # transformed_compressed_inv, alpha=0.5
    transformed_compressed_inv = [[0.1744, 19.2553, 18.7266, 18.9252],
                                  [0.2915, 23.4521, 23.7429, 23.7061],
                                  [0.4380, 25.4338, 26.1568, 25.9542],
                                  [0.5666, 26.5674, 27.3551, 27.1101],
                                  [0.6768, 27.3666, 28.1600, 27.9002],
                                  [0.7861, 28.0407, 28.7738, 28.5205],
                                  [0.9077, 28.6744, 29.3254, 29.0924],
                                  [1.0900, 29.6507, 30.1468, 29.9513]]
    compressed = np.asarray(compressed)
    transformed_compressed = np.asarray(transformed_compressed)
    transformed_compressed_inv = np.asarray(transformed_compressed_inv)

    alpha = 0.5
    config_all = ['fore', 'back', 'overall']
    for idx, config in enumerate(config_all):
        plt.plot(compressed[:,0], compressed[:,idx+1], color='m', marker='o', linestyle='-', label=r"JPEG")
        plt.plot(transformed_compressed[:,0], transformed_compressed[:,idx+1], color='c', marker='o', linestyle='-', label=r"DIICM")
        plt.plot(transformed_compressed_inv[:,0], transformed_compressed_inv[:,idx+1], color='b', marker='o', linestyle='-', label=r"DIICM-Inv")

        pdf_name = f"bpp_psnr_jpeg_{config}_{alpha}.pdf"
        plot_figure(pdf_name, 24, 18, 'best')

def plot_bpp_mAP_jpeg():
    # MaskRCNN_Res_C4, (bpp, 0.2, 0.5, 0.8)
    compressed = [[0.1945, 0.6030, 0.6030, 0.6030],
                  [0.3619, 9.3177, 9.3177, 9.3177],
                  [0.5565, 20.5541, 20.5541, 20.5541],
                  [0.7229, 25.7884, 25.7884, 25.7884],
                  [0.8635, 27.5274, 27.5274, 27.5274],
                  [1.0036, 28.4959, 28.4959, 28.4959],
                  [1.1628, 29.3418, 29.3418, 29.3418],
                  [1.3863, 30.0300, 30.0300, 30.0300]]
    
    # MaskRCNN_Res_C4, (bpp, 0.2, 0.5, 0.8)
    transformed_compressed_inferred = [[0.1744, 0.6586, 0.6345, 0.6140],
                                       [0.2915, 9.2795, 9.4469, 9.2429],
                                       [0.4380, 19.5740, 20.3576, 20.6697],
                                       [0.5666, 23.9554, 25.1866, 25.8335],
                                       [0.6768, 25.5364, 26.8324, 27.5861],
                                       [0.7861, 26.1961, 27.7206, 28.4610],
                                       [0.9077, 26.9908, 28.6115, 29.3297],
                                       [1.0900, 27.4705, 29.2966, 29.9275]]

    # MaskRCNN_Res_C4, (bpp, 0.2, 0.5, 0.8)
    transformed_compressed_label = [[0.1734, 0.6316, 0.5952, 0.5744],
                                    [0.2873, 10.3299, 9.8921, 9.4238],
                                    [0.4304, 23.1408, 22.4656, 21.1642],
                                    [0.5564, 28.8032, 28.0635, 26.2924],
                                    [0.6645, 31.2225, 30.2095, 28.5912],
                                    [0.7715, 32.3139, 31.3699, 29.5567],
                                    [0.8906, 33.5419, 32.3415, 30.3294],
                                    [1.0697, 34.5445, 33.2216, 31.0245]]

    compressed = np.asarray(compressed)
    transformed_compressed_inferred = np.asarray(transformed_compressed_inferred)
    transformed_compressed_label = np.asarray(transformed_compressed_label)

    alpha_all = [0.2, 0.5, 0.8]
    for idx, alpha in enumerate(alpha_all):
        plt.plot(compressed[:,0], compressed[:,idx+1], color='m', marker='o', linestyle='-', label=r"JPEG")
        plt.plot(transformed_compressed_inferred[:,0], transformed_compressed_inferred[:,idx+1], color='c', marker='o', linestyle='-', label=r"DIICM")
        plt.plot(transformed_compressed_label[:,0], transformed_compressed_label[:,idx+1], color='b', marker='o', linestyle='-', label=r"DIICM-GT")

        pdf_name = f"bpp_mAP_ablation_jpeg_segmentation_{alpha}.pdf"
        plot_figure(pdf_name, 24, 22, 'best')

if __name__ == '__main__':
    # plot_bpp_mAP()
    # plot_bpp_IoU_jpeg()
    # plot_bpp_psnr_jpeg()
    plot_bpp_mAP_jpeg()