# Prepare a personal video for GaVS

`prepare_video_for_gavs.py` is a new, standalone preprocessing pipeline. It
does not edit `train.py`, the GaVS configuration, or any other GaVS source
file. It creates a scene directory that the unchanged GaVS dataset loader can
read.

## What it creates

The final `--output-dir` has this layout:

```text
scene/
|-- blender.json
|-- images/                 # 864x480 original/undistorted frames
|-- fg_mask/                # 864x480, white=dynamic foreground
|-- inpainted/images/       # 1056x672 border-completed frames
|-- inpainted/fg_mask/      # 1056x672 foreground masks
`-- sparse/0/               # cameras.bin, images.bin, points3D.bin
```

Intermediate extracted frames, masks, the COLMAP database, and logs are put in
`--work-dir` (by default `OUTPUT/_pipeline`). All paths can be overridden from
the command line. The final `images`, `fg_mask`, and `sparse/0` paths are fixed
by GaVS's existing loader, so those three cannot be relocated inside the final
scene.

The script extracts a centered 864x480 sequence at 24 FPS by default. This
matches GaVS's 576x320 training crop scaled by 1.5 and its 864x480 evaluation
configuration. At least 64 registered frames are required by the unchanged
dataset loader.

## Install

The script needs Python 3.10 or newer, these Python packages, and two external
executables:

```text
Python:  numpy==1.26.4, Pillow, opencv-contrib-python==4.10.0.84
Programs: ffmpeg, colmap
```

From the GaVS root on Linux, a clean preprocessing environment is the least
confusing option:

```bash
conda create -n gavs-preprocess python=3.10 -y
conda activate gavs-preprocess
python -m pip install -r requirements-preprocess.txt
conda install -c conda-forge ffmpeg colmap
```

The normal GaVS environment can also run the script if `numpy`, `Pillow`, and
one working `cv2` installation are already present. Do not keep installing
`opencv-python`, `opencv-contrib-python`, and their `*-headless` variants on top
of one another; choose one distribution. `cv2` is only needed by this
preprocessor, while GaVS's training environment still needs its own compatible
PyTorch/CUDA packages.

Optional features:

* `--mask-method maskrcnn` additionally needs the matching `torch` and
  `torchvision` already used by GaVS, and downloads torchvision's pretrained
  Mask R-CNN weights on first use.
* `--mapper glomap` additionally needs a working `glomap` executable.
* `--inpaint-method import` needs externally generated 1056x672 completion
  images and masks. This is the place to use ProPainter or another completion
  system.

Check the installation with:

```bash
python prepare_video_for_gavs.py --help
ffmpeg -version
colmap help
```

## One-command pipeline

Run this from the GaVS repository root. Replace the paths with paths on the
machine where the video and tools actually exist:

```bash
python prepare_video_for_gavs.py \
  --video /Data/yanjiahui/videos/my_video.mp4 \
  --output-dir /Data/yanjiahui/datasets/my_video_gavs \
  --work-dir /Data/yanjiahui/work/my_video_gavs \
  --ffmpeg ffmpeg \
  --colmap colmap \
  --fps 24 \
  --mask-method motion \
  --inpaint-method temporal \
  --steps all
```

`motion` estimates foreground from frame-to-frame residuals after a homography
alignment. It is a usable baseline for a mostly static background, but it is
not the Grounded SAM 2 annotation used by the paper. If the scene has no
moving objects, use `--mask-method static`. If you already have masks, use
`--mask-method import --mask-source-dir /path/to/864x480_masks`, where every
mask is named `00000.png`, `00001.png`, ... and white means dynamic foreground.

For a lens-corrected or normal phone video, the default `--camera-model PINHOLE`
is usually the safest starting point. For footage with noticeable radial lens
distortion, try `--camera-model SIMPLE_RADIAL`; the script undistorts the final
images and writes centered PINHOLE intrinsics to `blender.json`.

COLMAP's SIFT extraction and matching use the CPU by default. If the installed
COLMAP has CUDA support, add `--sfm-gpu`. This is independent of the GPU used
by GaVS training.

## Run one step or resume

Steps are separate functions and can be run independently. They execute in the
pipeline's dependency order even if listed in another order:

```bash
# Extract only
python prepare_video_for_gavs.py --video /path/input.mp4 \
  --output-dir /path/scene --steps extract

# Replace the default masks after reviewing the extracted frames
python prepare_video_for_gavs.py --output-dir /path/scene \
  --steps masks --mask-method import --mask-source-dir /path/masks

# Continue from the existing extracted frames/masks
python prepare_video_for_gavs.py --output-dir /path/scene \
  --steps reconstruct geometry metadata inpaint inpaint-masks assemble validate
```

Extraction and COLMAP reconstruction intentionally require fresh directories;
`--overwrite` never deletes an existing database or silently mixes frames from
different videos. Use a new `--work-dir` after changing the input video.

For a quick fallback completion that does not require an external model:

```bash
python prepare_video_for_gavs.py --output-dir /path/scene \
  --steps inpaint inpaint-masks assemble validate \
  --inpaint-method reflect --overwrite
```

`temporal` and `reflect` are basic fallbacks. They preserve the original center
exactly, but neither is equivalent to ProPainter and neither can create reliable
unseen content in every recording.

## Train with unchanged GaVS

After `validate` succeeds, run training from the GaVS root. Keep the pretrained
RE10K/Flash3D checkpoint separate from the personal scene:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
WANDB_DIR=/Data/yanjiahui/wandb \
python train.py \
  hydra.run.dir=/Data/yanjiahui/results/my_video_gavs \
  +experiment=layered_gavs_overfit \
  dataset.data_path=/Data/yanjiahui/datasets/my_video_gavs \
  train.load_weights_folder=/Data/yanjiahui/datasets/gavs-data/re10k_v2_checkpoints \
  train.num_gpus=1
```

`WANDB_MODE=offline` avoids the interactive W&B choice; `WANDB_DIR` selects
the local W&B storage location. If an existing GaVS checkpoint is present in
the training output directory, the unchanged `train.py` resumes it, so use a
fresh `hydra.run.dir` for a new video unless resuming is intentional.

## Important limitations

An ordinary MP4 does not contain the phone's gyro history. This pipeline cannot
reproduce GaVS's rolling-shutter compensation from gyro data and reports
`rolling_shutter_compensated: false` in `pipeline_report.json`.

The paper's preprocessing used GLOMAP, Grounded SAM 2, and ProPainter. The
standalone defaults use COLMAP, a motion mask, and temporal/Telea or reflection
completion so that the pipeline has no hidden model dependencies. These are
approximations. The generated scene is suitable for trying GaVS, but stable
results depend on texture, camera motion, dynamic-object masks, lens model,
and the quality of the completion images.

The script does not generate UniDepth depths or RAFT flows; unchanged GaVS
computes those during training. It also does not generate the pretrained
`re10k_v2_checkpoints`; those must be downloaded separately as described by the
GaVS repository.
