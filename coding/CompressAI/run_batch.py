import os
import re
import matplotlib.pyplot as plt
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def generate_commands(data_root, mask_type, mask_network, alpha, arch, bitrate_level):
    # generate eval command
    arch_path = arch.replace('-', '_')
    
    # for preprocessed images
    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{data_root}/transformed/{mask_type}/{mask_network}/1.0_{alpha} "    # input
        f"-a {arch} --cuda -v "
        f"-d {data_root}/transformed_compressed/{arch_path}/{mask_type}/{mask_network}/1.0_{alpha}/level{bitrate_level} "  # output
        f"--per-image -p /home/gaocs/models/compressai/{arch_path}/mse/cheng2020-anchor-{bitrate_level}-*.pth.tar "
        f">{data_root}/transformed_compressed/{arch_path}_log/{mask_type}/{mask_network}/1.0_{alpha}/"
        f"compress_{mask_type}_{mask_network}_1.0_{alpha}_level{bitrate_level}.txt 2>&1"
    )

    # # for original images, without preprocessing
    # eval_command = (
    #     f"python -m compressai.utils.eval_model checkpoint "
    #     f"/home/gaocs/dataset/COCO/minVal2014 "    # input
    #     f"-a {arch} --cuda -v "
    #     f"-d {data_root}/compressed/{arch_path}/{mask_type}/{mask_network}/level{bitrate_level} "  # output
    #     f"--per-image -p /home/gaocs/models/compressai/{arch_path}/mse/cheng2020-anchor-{bitrate_level}-*.pth.tar "
    #     f">{data_root}/compressed/{arch_path}_log/{mask_type}/{mask_network}/"
    #     f"compress_{mask_type}_{mask_network}_level{bitrate_level}.txt 2>&1"
    # )

    return eval_command


def hyperprior_train_evaluate_pipeline(data_root, mask_type, mask_network, alpha, arch, bitrate_level):
    eval_cmd = generate_commands(data_root, mask_type, mask_network, alpha, arch, bitrate_level)

    time_start = time.time()
    print("\nEval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)


if __name__ == "__main__":
    data_root = "/home/gaocs/projects/ICM-DIICM/Data"
    mask_type = 'inferred'
    mask_network = 'MaskRCNN_Res101_FPN_0.5'
    alpha = 0.2
    arch = 'cheng2020-anchor'
    bitrate_level = 1

    bitrate_level_all = [1, 2, 3, 4, 5, 6]
    for bitrate_level in bitrate_level_all:
        hyperprior_train_evaluate_pipeline(data_root, mask_type, mask_network, alpha, arch, bitrate_level)