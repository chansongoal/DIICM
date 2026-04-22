# DIICM: Discrimination-Inspired Image Coding for Machines

## Overview
Official implementation of the paper **“DIICM: Discrimination-Inspired Image Coding for Machines.”** This repository provides the code used to run the DIICM pipeline with multiple codecs and machine-vision tasks.

DIICM studies **image coding for machines (ICM)** from a machine-vision perspective. Instead of only minimizing the gap between original and reconstructed objects, DIICM also considers **inter-object discrimination**, which is important for localization-oriented tasks such as object detection and instance segmentation.

DIICM is a **Task-agnostic**, **Codec-agnostic**, and **Plug-and-Play** ICM method, which achieves 28%–43% BD-BR reduction across three machine vision tasks and three codecs JPEG, Cheng2020, and VVC. In addition, DIICM is the first paper that conducts comparisons with MPEG-VCM and demonstrates its superiority on bitrate, task accuracy, and reconstruction quality tradeoff.

<!-- ## Method Summary

The DIICM framework contains four stages:

1. **Image Forward Transform**  
   Detect OOIs and apply the DIICM transform only to OONIs.
2. **Image Coding**  
   Compress the transformed image using any off-the-shelf codec.
3. **Image Understanding**  
   Run downstream machine-vision analysis on the reconstructed image.
4. **Image Inverse Transform**  
   Optionally enhance decoded images for better perceptual quality.

The forward transform is derived under a Gaussian assumption and can be written as:

```text
x' = alpha * x + (1 - alpha) * mu
```

where `alpha < 1` compresses the variance of OONIs while preserving their mean. In practice, the repository evaluates several shared transform coefficients such as `alpha = 0.2, 0.5, 0.8`. -->

## Environment and Dependencies

This repository combines code from several toolchains. A full run typically requires:

- Python 3.x
- PyTorch
- NumPy
- Pillow
- Detectron2
- CompressAI for the Cheng2020 experiments
- VTM software for the VVC experiments

Because the repository integrates multiple external frameworks, you will usually need to prepare the environment for each backend separately.

### Recommended setup

1. Create a Python environment for DIICM.
2. Install common Python dependencies such as `numpy`, `pillow`, and `torch`.
3. Install Detectron2 following its official instructions.
4. Install the local CompressAI version in editable mode:

```bash
cd coding/CompressAI
pip install -e .
```

5. Prepare your VTM executable environment if you plan to run VVC experiments.

## Data Preparation

From the paper, the main experiments use:

- **COCO minival2017** for general-purpose comparisons,
- **TVD** and **SFU** datasets for preprocessing comparisons.

The code also assumes that **OOI masks** are available, either:

- inferred from an object detector / segmenter, or
- generated from reconstructed images for inverse transform.

In the current codebase, many dataset, mask, and output paths are **hardcoded**. Before running experiments, you should carefully update the paths in scripts such as:

- `transform/transform.py`
- `coding/JPEG/jpeg.py`
- `rd_analysis/*.py`

## Repository Structure

```text
DIICM/
├── coding/
│   ├── CompressAI/          # Learned image compression (e.g., Cheng2020-related pipeline)
│   ├── JPEG/                # JPEG anchor compression scripts
│   └── VTM/                 # VVC / VTM batch scripts and utilities
├── job/                     # Cluster job submission scripts
├── machines/
│   └── detectron2/          # Detectron2-based training/evaluation code and configs
├── rd_analysis/             # RD / RA metric computation and plotting scripts
├── transform/               # Forward transform and inverse transform code
└── utils/                   # Dataset processing, JSON conversion, visualization, evaluation helpers
```

## Pipeline and Key Components

The repository follows the DIICM pipeline from preprocessing to evaluation. The main components are organized according to the practical execution flow below.

### 1. OOI Localization and Machine-Vision Evaluation

DIICM first identifies objects of interest (OOIs), and later evaluates downstream machine-vision performance on reconstructed images. These functions are mainly implemented under:

```bash
machines/detectron2/
```

Notable files include:

- `detectron2_eval_merged.py`
- `train_net.py`
- `configs/`

This part is used for tasks such as:

- object detection,
- instance segmentation,
- person keypoint detection.

In a typical run, this module is used twice: first to obtain OOI masks or localization results for preprocessing, and then to evaluate machine performance after compression and reconstruction.

### 2. Forward and Inverse Transform

After OOI regions are identified, DIICM applies the forward transform to OONIs and, when needed, applies an inverse transform after decoding. The core implementation is in:

```bash
transform/transform.py
```

This script contains both:

- the **forward transform** used before coding, and
- the **inverse transform** used after reconstruction for perceptual enhancement.

In practice, you usually edit paths and transform settings in this file, then run:

```bash
python transform/transform.py
```

### 3. Image Coding Backends

The transformed images can then be compressed by different codecs. The repository includes three backends corresponding to the paper.

#### JPEG

```bash
coding/JPEG/jpeg.py
```

Used for JPEG anchor compression and DIICM-based transformed-image compression.

#### Cheng2020 / CompressAI

```bash
coding/CompressAI/run_batch.py
```

Used for learned image compression experiments based on the local CompressAI branch included in this repository.

Typical usage:

```bash
cd coding/CompressAI
python run_batch.py
```

#### VVC / VTM

```bash
coding/VTM/
```

Contains VVC / VTM-related scripts and helper tools, such as:

- `vtm_anchor.bat`
- `vtm_anchor_transformed.bat`
- `DIICM.bat`
- `DIICM_transformed.bat`
- `png2yuv.py`
- `get_png_info.py`

You can choose one backend depending on which experiments you want to reproduce.

### 4. RD / RA Analysis

After compression and downstream evaluation, rate-distortion and rate-accuracy results can be computed under:

```bash
rd_analysis/
```

<!-- Examples include:

- `bpp_compute.py`
- `psnr.py`
- `psnr_inverse.py`
- `rd_jpeg.py`
- `rd_cheng2020.py`
- `rd_vtm.py`
- `rd_plot.py` -->

These scripts are used to compute bitrate, PSNR, and to generate RD / RA comparisons reported in the paper.

### 5. Utilities

Additional preprocessing, conversion, and visualization tools are placed under:

```bash
utils/
```

These scripts support dataset preparation, COCO-style annotation processing, result conversion, visualization, and other experiment utilities.




<!-- ## Important Notes

- The current repository is a **research codebase**, not yet a fully packaged toolbox.
- Several scripts contain **experiment-specific hardcoded paths** and assumptions about dataset layout.
- Some components depend on **external checkpoints**, dataset annotations, or local VTM binaries that are not bundled here.
- For reproducibility, it is best to follow the folder naming conventions already used in the code. -->

## Experimental Setting in the Paper

The paper evaluates DIICM on three machine-vision tasks:

- object detection,
- instance segmentation,
- person keypoint detection.

It reports results with:

- **JPEG**
- **Cheng2020**
- **VVC**
- **MPEG-VCM**

and studies transform coefficients:

- `alpha = 0.2`
- `alpha = 0.5`
- `alpha = 0.8`

The paper also introduces an inverse transform to improve visual quality after decoding.

<!-- ## Citation

If you use this repository in your research, please cite the DIICM paper.

```bibtex
@article{gao2026diicm,
  title={DIICM: Discrimination-Inspired Image Coding for Machines},
  author={Gao, Changsheng and Li, Li and Liu, Dong and Wu, Feng and Ebrahimi, Touradj},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026}
}
``` -->

## Acknowledgement

This repository builds on widely used research frameworks including Detectron2, CompressAI, and VTM.
