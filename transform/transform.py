import enum
import os

# from pycocotools.coco import COCO
# from PIL import ImageFilter, ImageEnhance
from PIL import Image
# from matplotlib import pyplot as plt
import numpy as np 
import pickle

def clip(value, upper):
    if value < 0:
        value = 0
    elif value >upper:
        value = upper
    return value


def transform(alpha):
    # org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    # mask_path = r'/data/gaocs/Understanding_Detection/label_mask/'
    # rec_path = r'/data/gaocs/Understanding_Detection/transformed/1.0_' + str(alpha) + '/'

    iou_threshold = '0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    mask_path = r'/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
        mask_img = Image.open(mask_path + img_name[:-4]+'.png')
        mask_arr = np.asarray(mask_img)
        inv_mask_arr = -mask_arr + 1

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * alpha + (1-alpha)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr

        img_arr_scaled = fore_arr + back_arr_sacled
        img_arr_scaled = img_arr_scaled.astype(np.uint8)
        img_scaled = Image.fromarray(img_arr_scaled)
        img_scaled.save(rec_path+img_name[:-4]+'.png')


def transform_inferenced(alpha):
    # iou_thresholds = ['0.2', '0.5', '0.75']
    # for iou_threshold in iou_thresholds:
    iou_threshold = '0.5'
    org_path = r'/data/gaocs/Understanding_Detection/minVal2014/'
    mask_path = r'/data/gaocs/Understanding_Detection/inferenced_mask/MaskRCNN_R101_FPN/org_org_' + iou_threshold + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
        inferenced_img = Image.open(mask_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask_arr
        temp[:,:,1] = inv_mask_arr
        temp[:,:,2] = inv_mask_arr
        inv_mask_arr = temp
        mask_arr = (-inv_mask_arr + 1).astype(np.uint8)

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * alpha + (1-alpha)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr

        img_arr_scaled = fore_arr + back_arr_sacled
        img_arr_scaled = img_arr_scaled.astype(np.uint8)
        img_scaled = Image.fromarray(img_arr_scaled)
        img_scaled.save(rec_path+img_name[:-4]+'.png')

def inv_transform(alpha, beta, quality):
    iou_threshold = '0.5'
    mask_path = r'/gdata1/gaocs/Data_DIICM/MaskFromOrgImg_MaskRCNN_Res101_FPN_0.5'
    org_path = r'/data/gaocs/Understanding_Detection/transformed_compressed/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(alpha) + '/' + quality + '/'
    rec_path = r'/data/gaocs/Understanding_Detection/transformed_compressed_inv/inferenced/MaskRCNN_Res101_FPN_' + iou_threshold + '/' + '1.0_' + str(beta) + '/' + quality + '/'

    if not os.path.exists(rec_path):
        os.makedirs(rec_path)

    img_files = os.listdir(org_path)
    img_files.sort()

    inv_statistics_path = '/data/gaocs/Understanding_Detection/inv_statistics/'
    inv_statistics_filename = 'inv_statistics_1.0_' + str(alpha) + '_' + quality + '.txt'
    up_per_all = []
    low_per_all = []
    maximum_all = [] 
    minimum_all = [] 
    avg_all = []
    up_avg_all = [] 
    low_avg_all = []
    if not os.path.exists(inv_statistics_path):
        os.makedirs(inv_statistics_path)
    inv_statistics_file = open(inv_statistics_path + inv_statistics_filename, 'w')

    for idx, img_name in enumerate(img_files):
        # get mask
        img_name = img_files[idx]
        inferenced_img = Image.open(mask_path + img_name[:-4]+'.png')
        inferenced_arr = np.asarray(inferenced_img)
        inferenced_arr = np.sum(inferenced_arr, axis=2)
        inv_mask_arr = (inferenced_arr / 255/3).astype(np.uint8)
        temp = np.zeros((inv_mask_arr.shape[0], inv_mask_arr.shape[1],3), dtype=np.uint8)
        temp[:,:,0] = inv_mask_arr
        temp[:,:,1] = inv_mask_arr
        temp[:,:,2] = inv_mask_arr
        inv_mask_arr = temp
        mask_arr = (-inv_mask_arr + 1).astype(np.uint8)

        org_img = Image.open(org_path + img_name)
        org_arr = np.asarray(org_img).astype(np.float32)

        fore_arr = org_arr * mask_arr
        back_arr = org_arr * inv_mask_arr

        mean = np.sum(back_arr) / np.sum(inv_mask_arr)
        back_arr_scaled = back_arr * beta + (1-beta)*mean
        back_arr_sacled = back_arr_scaled * inv_mask_arr
        img_arr_scaled = fore_arr + back_arr_sacled

        # normalization 1: only normalize the values that greater than 255 or less than 0
        # idx1, idx2 = back_arr_sacled>255, back_arr_sacled<0
        # idx_all = idx1 | idx2
        # idx_all = idx_all.astype(int)
        # inv_idx_all = -idx_all + 1
        # img_arr_scaled_ = (img_arr_scaled - np.min(back_arr_sacled))/(np.max(back_arr_sacled)-np.min(back_arr_sacled)) * 255
        # img_arr_scaled = img_arr_scaled*inv_idx_all + img_arr_scaled_ * idx_all

        # normalization 2
        num = img_arr_scaled.shape[0] * img_arr_scaled.shape[1] * img_arr_scaled.shape[2]
        up_per, low_per = np.count_nonzero(img_arr_scaled>255)/num, np.count_nonzero(img_arr_scaled<0)/num
        maximum, minimum, avg, up_avg, low_avg = np.max(img_arr_scaled), np.min(img_arr_scaled),  np.mean(img_arr_scaled), np.mean(img_arr_scaled[img_arr_scaled>255]), np.mean(img_arr_scaled[img_arr_scaled<0])
        write_line = img_name + ' ' +str(up_per)+' ' + str(low_per)+' ' + str(maximum)+' ' + str(minimum)+' ' +str(avg)+' ' +str(up_avg)+' ' +str(low_avg)+' \n'
        inv_statistics_file.write(write_line)
        up_per_all.append(up_per)
        low_per_all.append(low_per)
        maximum_all.append(maximum) 
        minimum_all.append(minimum)
        avg_all.append(avg)
        up_avg_all.append(up_avg)
        low_avg_all.append(low_avg)
        img_arr_scaled[img_arr_scaled>255] = 255
        img_arr_scaled[img_arr_scaled<0] = 0

        # # img_arr_scaled = fore_arr + back_arr_sacled
        # img_arr_scaled = img_arr_scaled.astype(np.uint8)
        # img_scaled = Image.fromarray(img_arr_scaled)
        # img_scaled.save(rec_path+img_name[:-4]+'.png')
    write_line = 'average: ' + ' ' +str(np.mean(up_per_all))+' ' + str(np.mean(low_per_all))+' ' + str(np.mean(maximum_all))+' ' + str(np.mean(minimum_all))+' ' +str(np.mean(avg_all))+' ' +str(np.mean(up_avg_all))+' ' +str(np.mean(low_avg_all))+' \n'
    inv_statistics_file.write(write_line)

def inverse_transform(data_root, org_path, processing_config, arch, mask_type, inv_mask_type, mask_network, inv_mask_network, alpha, beta, quality, num_workers=8):
    # Build paths
    if arch == "vtm_anchor":
        rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/rec_png"
        inverse_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/inverse_png_{inv_mask_type}"
        inv_mask_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/{inv_mask_type}"
    elif arch == "cheng2020_anchor":
        rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/image"
        inverse_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/inverse_png_{inv_mask_type}"
        inv_mask_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/{inv_mask_type}"
    elif arch == "jpeg_anchor":
        rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/rec_jpg"
        inverse_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/inverse_png_{inv_mask_type}"
        inv_mask_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/{inv_mask_type}"
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    os.makedirs(inverse_path, exist_ok=True)

    # Mask directory (keep original behavior)
    if inv_mask_type == 'MaskFromOrgImg': inv_mask_path = "/gdata1/gaocs/Data_DIICM/MaskFromOrgImg_MaskRCNN_Res101_FPN_0.5"

    # Statistics output
    inv_statistics_path = f'/gdata1/gaocs/Data_DIICM/{processing_config}/{arch}/inverse_statistics'
    inv_statistics_name = f"{inv_statistics_path}/{arch}_{inv_mask_type}_1.0_{alpha}_qp{quality}.txt"
    os.makedirs(inv_statistics_path, exist_ok=True)
    # inv_statistics_file = open(inv_statistics_name, 'w')

    # Print info
    print(processing_config, arch, inv_mask_type, inv_mask_network, alpha, beta, quality)
    print(f"rec_path: {rec_path}")
    print(f"inv_mask_path: {inv_mask_path}")
    print(f"inv_png_path: {inverse_path}")
    print(f"inv_statistics_name: {inv_statistics_name}")

    with open(inv_statistics_name, "w", encoding="utf-8") as f:
        f.write(
            "Summary Info:\n"
            f"arch: {arch}\n"
            f"inv_mask_type: {inv_mask_type}\n"
            f"inv_mask_network: {inv_mask_network}\n"
            f"alpha: {alpha}\n"
            f"beta: {beta}\n"
            f"quality: {quality}\n"
            f"rec_path: {rec_path}\n"
            f"inv_mask_path: {inv_mask_path}\n"
            f"inv_png_path: {inverse_path}\n"
            f"inv_statistics_name: {inv_statistics_name}\n\n"
        )


    # Collect images
    if not os.path.exists(rec_path): raise FileNotFoundError(f"Reconstructed image folder not found: {rec_path}")
    img_files = sorted(os.listdir(rec_path))
    if len(img_files) == 0: raise FileNotFoundError(f"No image files found in: {rec_path}")

    # Statistics containers
    up_per_all, low_per_all = [], []
    maximum_all, minimum_all = [], []
    avg_all, up_avg_all, low_avg_all = [], [], []

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    with open(inv_statistics_name, "w", encoding="utf-8") as f:
        f.write(f"img_name up_percentage low_percentage maximum minimum avg up_avg low_avg\n")
        for img_name in img_files:
            # ---------- Load mask ----------
            inv_mask_file = f"{inv_mask_path}/{img_name.rsplit('.', 1)[0]}.png"
            if not os.path.exists(inv_mask_file):
                raise FileNotFoundError(f"Mask not found: {inv_mask_file}")

            inv_mask_img = Image.open(inv_mask_file).convert("RGB")
            inv_mask_arr = np.asarray(inv_mask_img, dtype=np.uint8)

            summed = np.sum(inv_mask_arr, axis=2, dtype=np.uint16)
            inv_mask_2d = (summed / (255.0 * 3.0)).astype(np.uint8)

            inv_mask = np.repeat(inv_mask_2d[:, :, None], 3, axis=2).astype(np.float32)
            fore_mask = 1.0 - inv_mask

            # ---------- Load reconstructed image ----------
            rec_img = Image.open(f"{rec_path}/{img_name}").convert("RGB")
            rec_arr = np.asarray(rec_img, dtype=np.float32)

            # ---------- Foreground / background ----------
            fore_arr = rec_arr * fore_mask
            back_arr = rec_arr * inv_mask

            # ---------- Background mean ----------
            den = np.sum(inv_mask)
            mean_val = float(np.sum(back_arr) / den) if den > 0 else 0.0

            # ---------- Inverse scaling ----------
            # ！！！！！！！！！！
            # Please check if the mean_val is correctly computed. Should the reconstructed mean value or the original mean value be used?
            # back_scaled = back_arr * beta + beta * (1.0 - beta) * mean_val  # 
            back_scaled = back_arr * beta + (1.0 - beta) * mean_val
            back_scaled = back_scaled * inv_mask

            inv_img = fore_arr + back_scaled

            # ---------- Statistics (before clip) ----------
            num = inv_img.size
            up_mask = inv_img > 255.0
            low_mask = inv_img < 0.0

            up_per = np.count_nonzero(up_mask) / num
            low_per = np.count_nonzero(low_mask) / num

            maximum = float(np.max(inv_img))
            minimum = float(np.min(inv_img))
            avg = float(np.mean(inv_img))

            up_vals = inv_img[up_mask]
            low_vals = inv_img[low_mask]
            up_avg = float(np.mean(up_vals)) if up_vals.size > 0 else 0.0
            low_avg = float(np.mean(low_vals)) if low_vals.size > 0 else 0.0

            f.write(
                f"{img_name} {up_per} {low_per} {maximum} {minimum} "
                f"{avg} {up_avg} {low_avg}\n"
            )

            up_per_all.append(up_per)
            low_per_all.append(low_per)
            maximum_all.append(maximum)
            minimum_all.append(minimum)
            avg_all.append(avg)
            up_avg_all.append(up_avg)
            low_avg_all.append(low_avg)

            # ---------- Clip & save ----------
            inv_img = np.clip(inv_img, 0.0, 255.0).astype(np.uint8)
            Image.fromarray(inv_img).save(f"{inverse_path}/{img_name[:-4]}.png")

        # ---------- Dataset-level average ----------
        f.write(f"Average up_percentage low_percentage maximum minimum avg up_avg low_avg\n")
        f.write(
            "Average: "
            f"{np.mean(up_per_all)} "
            f"{np.mean(low_per_all)} "
            f"{np.mean(maximum_all)} "
            f"{np.mean(minimum_all)} "
            f"{np.mean(avg_all)} "
            f"{np.mean(up_avg_all)} "
            f"{np.mean(low_avg_all)}\n"
        )

# def main_inverse_transform(data_root, org_path, mask_type, inv_mask_type, mask_network, inv_mask_network, arch, processing_config, alpha, beta, quality_all) -> None:
def main_inverse_transform() -> None:
    data_root = "/gdata1/gaocs/Data_DIICM"
    org_path = "/gdata/gaocs/dataset/COCO/minVal2014"
    mask_type = "inferred"
    inv_mask_type = "MaskFromRecImg"    # MaskFromOrgImg, MaskFromRecImg
    mask_network = "MaskRCNN_Res101_FPN_0.5"
    inv_mask_network = "MaskRCNN_Res50_C4"  # kept for API compatibility
    arch = "jpeg_anchor"
    processing_config = "transformed_compressed"
    alpha = 0.5
    beta = 2.0

    quality_all = [70, 60, 50, 40, 30, 20, 10, 1]
    # quality_all = [70]
    for quality in quality_all:
        inverse_transform(
            data_root=data_root,
            org_path=org_path,
            processing_config=processing_config,
            arch=arch,
            mask_type=mask_type,
            inv_mask_type=inv_mask_type,
            mask_network=mask_network,
            inv_mask_network=inv_mask_network,
            alpha=alpha,
            beta=beta,
            quality=quality,
            num_workers=8,
        )


if __name__ == "__main__":
    # parser.add_argument("--data_root", type=str, default="/gdata1/gaocs/Data_DIICM")
    # parser.add_argument("--org_path", type=str, default="/gdata/gaocs/dataset/COCO/minVal2014")
    # parser.add_argument("--mask_type", type=str, default="inferred")
    # parser.add_argument("--inv_mask_type", type=str, default="MaskFromRecImg")
    # parser.add_argument("--mask_network", type=str, default="MaskRCNN_Res101_FPN_0.5")
    # parser.add_argument("--inv_mask_network", type=str, default="MaskRCNN_Res50_C4")
    # parser.add_argument("--arch", type=str, default="jpeg_anchor")
    # parser.add_argument("--processing_config", type=str, default="transformed_compressed")
    # parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument("--beta", type=float, default=2.0)
    # parser.add_argument('--quality_all', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='List of quality levels')
    # args = parser.parse_args()

    # main_inverse_transform(args.data_root, args.org_path, args.mask_type, args.inv_mask_type, args.mask_network, args.inv_mask_network, args.arch, args.processing_config, args.alpha, args.beta, args.quality_all)
    main_inverse_transform()
