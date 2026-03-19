# Making Video Models Adhere to User Intent with Minor Adjustments

**Daniel Ajisafe**<sup>1</sup>, **Eric Hedlin**<sup>1</sup>, **Helge Rhodin**<sup>2,1</sup>, **Kwang Moo Yi**<sup>1</sup>

<sup>1</sup>University of British Columbia &nbsp; <sup>2</sup>Bielefeld University

**TMLR 2026**

---

This repository is the official project page for **Making Video Models Adhere to User Intent with Minor Adjustments**. We show that slightly adjusting user-provided bounding boxes to align with a video diffusion model's internal attention maps improves generation quality and control adherence.

[**Project Page**](https://ubc-vision.github.io/MinorAdjustVideo/) &nbsp;|&nbsp; [**Paper**](#) &nbsp;|&nbsp; [**Video**](https://ubc-vision.github.io/MinorAdjustVideo/docs/webpage/resources/videos/short-teaser/Box_adjust_control_short-video.mp4)

---

## Setup

```bash
# 0) clone repo
git clone https://github.com/ubc-vision/MinorAdjustVideo.git
cd MinorAdjustVideo

# 1) create conda environment
conda create -n minor_adjust_video python=3.10
conda activate minor_adjust_video

# 2) set a local cache directory
mkdir .cache; export XDG_CACHE_HOME=.cache

# 3) Install packages
pip install -r requirements.txt

# 4) install Git LFS (for model pointer files)
conda install -c conda-forge git-lfs
git lfs install
```

Download ZeroScope (create Hugging Face account + token if gated):

```bash
export ZEROSCOPE_MODEL_ROOT="$(pwd)/.cache"
git clone https://huggingface.co/cerspense/zeroscope_v2_576w .cache/cerspense/zeroscope_v2_576w
```

---

## Demo

Quick demo (SWAN teaser):
***Box optimization:**

```bash
python3 -m bin.Fwd_CmdTrailBlazer --config config/box_opt_teaser/leveltestD/swan/motion_0001.yaml --run_config config/run_opt_fwd.yaml --validate --validate_dirname teaser --val_model_name "optim" --shared_config config/common_shared.yaml --output-path output --bb_deviate_lambda 0.1 --outside_bbox_loss_scale 10 --width 320 --height 320 --set_global_deterministic
```

***TrailBlazer:**
```bash
python3 -m bin.Fwd_CmdTrailBlazer --config config/box_opt_teaser/leveltestD/swan/motion_0001.yaml --validate --validate_dirname teaser --val_model_name "trailblazer_origin" --shared_config config/common_shared.yaml --output-path output --width 320 --height 320 --set_global_deterministic
```

More demo commands and options → [demo.md](docs/demo.md).  
Evaluation (test set, configs, outputs) → [eval.md](docs/eval.md).

---

## Roadmap

- [x] Project page and video
- [x] Code release
- [ ] Demo and usage instructions
- [ ] Evaluation scripts and data


