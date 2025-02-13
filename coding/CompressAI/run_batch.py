import os
import re
import time
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def generate_commands(data_root, mask_type, mask_network, alpha, arch, quality):
    # generate eval command
    arch_path = arch.replace('-', '_')
    
    # compress preprocessed images with inferred masks
    if mask_type == 'inferred':
        output_path = f"{data_root}/transformed_compressed/{arch_path}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}"; os.makedirs(output_path, exist_ok=True)
        eval_command = (
            f"python -m compressai.utils.eval_model checkpoint "
            f"{data_root}/transformed/{mask_type}/{mask_network}/1.0_{alpha} "    # input
            f"-a {arch} --cuda -v "
            f"-d {data_root}/transformed_compressed/{arch_path}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality} "  # output
            f"--per-image -p /gdata/gaocs/pretrained_models/compressai/{arch}/mse/{arch}-{quality}-*.pth.tar "
            f">{data_root}/transformed_compressed/{arch_path}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/"
            f"compress_{mask_type}_{mask_network}_1.0_{alpha}_quality{quality}.txt 2>&1"
        )

    # compress preprocessed images with ground-truth masks
    elif mask_type == 'label':
        output_path = f"{data_root}/transformed_compressed/{arch_path}/{mask_type}/1.0_{alpha}/quality{quality}"; os.makedirs(output_path, exist_ok=True)
        eval_command = (
            f"python -m compressai.utils.eval_model checkpoint "
            f"{data_root}/transformed/{mask_type}/1.0_{alpha} "    # input
            f"-a {arch} --cuda -v "
            f"-d {data_root}/transformed_compressed/{arch_path}/{mask_type}/1.0_{alpha}/quality{quality} "  # output
            f"--per-image -p /gdata/gaocs/pretrained_models/compressai/{arch}/mse/{arch}-{quality}-*.pth.tar "
            f">{data_root}/transformed_compressed/{arch_path}/{mask_type}/1.0_{alpha}/quality{quality}/"
            f"compress_{mask_type}_1.0_{alpha}_quality{quality}.txt 2>&1"
        )

    # compress original images without preprocessing
    elif mask_type == 'original':
        output_path = f"{data_root}/compressed/{arch_path}/quality{quality}"; os.makedirs(output_path, exist_ok=True)
        eval_command = (
            f"python -m compressai.utils.eval_model checkpoint "
            f"/gdata/gaocs/dataset/COCO/minVal2014 "    # input
            f"-a {arch} --cuda -v "
            f"-d {data_root}/compressed/{arch_path}/quality{quality} "  # output
            f"--per-image -p /gdata/gaocs/pretrained_models/compressai/{arch}/mse/{arch}-{quality}-*.pth.tar "
            f">{data_root}/compressed/{arch_path}/quality{quality}/"
            f"compress_quality{quality}.txt 2>&1"
        )

    return eval_command


def hyperprior_train_evaluate_pipeline(data_root, mask_type, mask_network, alpha, arch, bitrate_level):
    eval_cmd = generate_commands(data_root, mask_type, mask_network, alpha, arch, bitrate_level)

    time_start = time.time()
    print("\nEval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)

def main():
    parser = argparse.ArgumentParser(description="Hyperprior Evaluation Pipeline")
    parser.add_argument('--data_root', type=str, default="/gdata1/gaocs/Data_DIICM", help='Root directory of the dataset')
    parser.add_argument('--mask_type', type=str, choices=['inferred', 'label', 'original'], default='inferred', help='Type of mask used')
    parser.add_argument('--mask_network', type=str, default='MaskRCNN_Res101_FPN_0.5', help='Mask network used (if applicable)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha parameter')
    parser.add_argument('--arch', type=str, default='cheng2020-anchor', help='Model architecture')
    parser.add_argument('--quality', type=int, default=1, help='Quality level to process')
    
    args = parser.parse_args()
    
    hyperprior_train_evaluate_pipeline(args.data_root, args.mask_type, args.mask_network, args.alpha, args.arch, args.quality)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     data_root = "/gdata1/gaocs/Data_DIICM"
#     mask_type = 'inferred'; mask_network = 'MaskRCNN_Res101_FPN_0.5'
#     # mask_type = 'label'; mask_network = ''
#     # mask_type = 'original'; mask_network = '' # for original images without preprocessing
#     alpha = 0.2
#     arch = 'cheng2020-anchor'

#     quality_all = [6]
#     for quality in quality_all:
#         hyperprior_train_evaluate_pipeline(data_root, mask_type, mask_network, alpha, arch, quality)