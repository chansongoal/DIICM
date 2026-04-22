import os
import math
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def _read_image_float32(path: str) -> np.ndarray:
    """Read an image and return float32 array in [0, 255]."""
    with Image.open(path) as im:
        arr = np.asarray(im, dtype=np.float32)
    return arr


def _read_mask(path: str) -> np.ndarray:
    """
    Read a mask image and convert to boolean mask.

    This is robust to masks stored as 0/1.
    Pixels >= threshold are treated as foreground.
    """
    with Image.open(path) as im:
        mask = np.asarray(im, dtype=np.float32)  # keep original channels

    return mask


def _psnr_from_mse(mse: float, pixel_max: float = 255.0, eps: float = 1e-12) -> float:
    """Compute PSNR from MSE with numerical safety."""
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10((pixel_max * pixel_max) / mse)


def _compute_one(img_name: str, org_path: str, rec_path: str, mask_path: str) -> tuple:
    """
    Compute foreground/background/overall MSE+PSNR for one image.

    This version is mathematically consistent with your original implementation:
    - Fore/Back: uses per-element masking in HxWxC (mask is 3-channel), then squared, then summed
    - Overall: mean over all H*W*C elements
    """
    org_file = os.path.join(org_path, img_name)
    rec_file = os.path.join(rec_path, os.path.splitext(img_name)[0] + ".png")
    mask_file = os.path.join(mask_path, os.path.splitext(img_name)[0] + ".png")

    if not (os.path.exists(org_file) and os.path.exists(rec_file) and os.path.exists(mask_file)):
        return (img_name, None)

    org = _read_image_float32(org_file)
    rec = _read_image_float32(rec_file)
    mask = _read_mask(mask_file)  # HxWxC

    if org.shape != rec.shape:
        raise ValueError(f"Shape mismatch: org={org.shape}, rec={rec.shape}, file={img_name}")

    # Ensure mask shape matches org/rec exactly (as you stated)
    if mask.shape != org.shape:
        # A common corner case: mask has 1 channel but org has 3 channels
        if mask.ndim == 3 and org.ndim == 3 and mask.shape[:2] == org.shape[:2] and mask.shape[2] == 1 and org.shape[2] == 3:
            mask = np.repeat(mask, 3, axis=2)
        else:
            raise ValueError(f"Mask shape mismatch: mask={mask.shape}, image={org.shape}, file={img_name}")

    inv_mask = 1.0 - mask

    # Compute squared error once
    diff2 = (org - rec) ** 2  # HxWxC

    # Overall MSE (exactly the same as your original code)
    overall_mse = float(np.mean(diff2))
    overall_psnr = _psnr_from_mse(overall_mse)
    # print(overall_mse, overall_psnr)

    # Foreground/background MSE:
    # Original math:
    # fore_mse = sum((org*mask - rec*mask)^2) / sum(mask)
    #          = sum(diff2 * mask^2) / sum(mask)
    mask2 = mask * mask
    inv2 = inv_mask * inv_mask

    num_fore = float(np.sum(mask))
    num_back = float(np.sum(inv_mask))

    # Handle degenerate masks (all 0 or all 1)
    if num_fore <= 1e-12 or num_back <= 1e-12:
        return (img_name, float("nan"), float("nan"), overall_mse, float("nan"), float("nan"), overall_psnr)

    fore_mse = float(np.sum(diff2 * mask2) / num_fore)
    back_mse = float(np.sum(diff2 * inv2) / num_back)

    fore_psnr = _psnr_from_mse(fore_mse)
    back_psnr = _psnr_from_mse(back_mse)

    return (img_name, fore_mse, back_mse, overall_mse, fore_psnr, back_psnr, overall_psnr)


def psnr_compute_fast(org_path: str, rec_path: str, psnr_name: str, mask_path: str, num_workers: int = 8):
    """
    Compute PSNR metrics for a folder with multiprocessing.

    Notes:
    - Parallelizes I/O + numpy computation using ProcessPoolExecutor.
    - Results are aggregated and written by the main process only.
    """
    img_files = sorted(os.listdir(org_path))
    os.makedirs(os.path.dirname(psnr_name), exist_ok=True)

    fore_mse_all, back_mse_all, overall_mse_all = [], [], []
    fore_psnr_all, back_psnr_all, overall_psnr_all = [], [], []

    with open(psnr_name, "w") as f:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_compute_one, img_name, org_path, rec_path, mask_path) for img_name in img_files]

            for fut in as_completed(futures):
                out = fut.result()

                if len(out) == 2 and out[1] is None:
                    img_name = out[0]
                    f.write(f"{img_name} MISSING\n")
                    continue

                img_name, fore_mse, back_mse, overall_mse, fore_psnr, back_psnr, overall_psnr = out

                f.write(
                    f"{img_name} {fore_mse} {back_mse} {overall_mse} "
                    f"{fore_psnr} {back_psnr} {overall_psnr}\n"
                )

                if (
                    np.isfinite(fore_mse) and np.isfinite(back_mse)
                    and 0.0 < fore_mse < 256.0 * 256.0
                    and 0.0 < back_mse < 256.0 * 256.0
                ):
                    fore_mse_all.append(fore_mse)
                    back_mse_all.append(back_mse)
                    overall_mse_all.append(overall_mse)
                    fore_psnr_all.append(fore_psnr)
                    back_psnr_all.append(back_psnr)
                    overall_psnr_all.append(overall_psnr)

        if len(fore_mse_all) > 0:
            f.write(
                "average: "
                f"{float(np.mean(fore_mse_all))} "
                f"{float(np.mean(back_mse_all))} "
                f"{float(np.mean(overall_mse_all))} "
                f"{float(np.mean(fore_psnr_all))} "
                f"{float(np.mean(back_psnr_all))} "
                f"{float(np.mean(overall_psnr_all))}\n"
            )
        else:
            f.write("average: NA\n")

        print(f"{np.mean(fore_psnr_all):.4f}, {np.mean(back_psnr_all):.4f}, {np.mean(overall_psnr_all):.4f}")

def psnr(data_root, org_path, processing_config, arch, mask_type, mask_network, mask_path, alpha, quality, rec_suffix, num_workers=8):
    psnr_path = None
    rec_path = None
    psnr_name = None

    if processing_config == "transformed_compressed":
        if mask_type == "inferred":
            if arch == "vtm_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/qp{quality}/{rec_suffix}"
            elif arch == "cheng2020_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/{rec_suffix}"
            elif arch == "jpeg_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/{mask_network}/1.0_{alpha}/quality{quality}/{rec_suffix}"
            else:
                raise ValueError(f"Unknown arch: {arch}")

            psnr_path = f"{data_root}/{processing_config}/{arch}/psnr/{mask_type}"
            psnr_name = f"{psnr_path}/psnr_{processing_config}_{mask_type}_{mask_network}_1.0_{alpha}_{quality}_{quality}_{rec_suffix}.txt"

        elif mask_type == "label":
            if arch == "vtm_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/qp{quality}/{rec_suffix}"
            elif arch == "cheng2020_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/quality{quality}/{rec_suffix}"
            elif arch == "jpeg_anchor":
                rec_path = f"{data_root}/{processing_config}/{arch}/{mask_type}/1.0_{alpha}/quality{quality}/{rec_suffix}"
            else:
                raise ValueError(f"Unknown arch: {arch}")

            psnr_path = f"{data_root}/{processing_config}/{arch}/psnr/{mask_type}"
            psnr_name = f"{psnr_path}/psnr_{processing_config}_{mask_type}_1.0_{alpha}_{quality}_{quality}_{rec_suffix}.txt"
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

    elif processing_config == "compressed":
        psnr_path = f"{data_root}/{processing_config}/{arch}/psnr/{mask_type}"
        if arch == "vtm_anchor":
            psnr_name = f"{psnr_path}/psnr_{processing_config}_qp{quality}_{rec_suffix}.txt"
            rec_path = f"{data_root}/{processing_config}/{arch}/qp{quality}/{rec_suffix}"
        elif arch == "cheng2020_anchor":
            psnr_name = f"{psnr_path}/psnr_{processing_config}_quality{quality}_{rec_suffix}.txt"
            rec_path = f"{data_root}/{processing_config}/{arch}/quality{quality}/{rec_suffix}"
    else:
        raise ValueError(f"Unknown processing_config: {processing_config}")

    os.makedirs(psnr_path, exist_ok=True)
    psnr_compute_fast(org_path, rec_path, psnr_name, mask_path, num_workers=num_workers)


def main():
    data_root = "/gdata1/gaocs/Data_DIICM"
    org_path = "/gdata/gaocs/dataset/COCO/minVal2014"
    mask_type = "inferred"
    mask_network = "MaskRCNN_Res101_FPN_0.5"
    mask_path = "/gdata/gaocs/dataset/COCO/minVal2014_GT_mask/"
    arch = "vtm_anchor" # cheng2020_anchor, vtm_anchor
    processing_config = "transformed_compressed"    # compressed, transformed_compressed
    rec_suffix = 'inverse_png_MaskFromRecImg'   # 'rec_png' for vtm, 'image' for cheng2020, 'inverse_png_MaskFromOrgImg' for inv_img
    alpha = 0.5

    print(processing_config, arch, mask_type, mask_network, alpha, rec_suffix)

    # quality_all = [51, 47, 42, 37, 32, 27]
    # for quality in quality_all:
    #     psnr(
    #         data_root=data_root,
    #         org_path=org_path,
    #         processing_config=processing_config,
    #         arch=arch,
    #         mask_type=mask_type,
    #         mask_network=mask_network,
    #         mask_path=mask_path,
    #         alpha=alpha,
    #         quality=quality,
    #         rec_suffix=rec_suffix,
    #         num_workers=8,
    #     )

    # arch = "cheng2020_anchor"
    # quality_all = [1, 2, 3, 4, 5, 6]
    # print(processing_config, arch, mask_type, mask_network, alpha, rec_suffix)
    # for quality in quality_all:
    #     psnr(
    #         data_root=data_root,
    #         org_path=org_path,
    #         processing_config=processing_config,
    #         arch=arch,
    #         mask_type=mask_type,
    #         mask_network=mask_network,
    #         mask_path=mask_path,
    #         alpha=alpha,
    #         quality=quality,
    #         rec_suffix=rec_suffix,
    #         num_workers=8,
    #     )

    arch = "jpeg_anchor"
    quality_all = [70, 60, 50, 40, 30, 20, 10, 1]
    # quality_all = [70]
    print(processing_config, arch, mask_type, mask_network, alpha, rec_suffix)
    for quality in quality_all:
        psnr(
            data_root=data_root,
            org_path=org_path,
            processing_config=processing_config,
            arch=arch,
            mask_type=mask_type,
            mask_network=mask_network,
            mask_path=mask_path,
            alpha=alpha,
            quality=quality,
            rec_suffix=rec_suffix,
            num_workers=8,
        )

if __name__ == "__main__":
    main()
