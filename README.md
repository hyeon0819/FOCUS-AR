<div align="center">

# Modality-Observable Factorized Text Alignment for <br>RGB-Skeleton Open-Vocabulary Action Recognition

[Sanghyeon Lee](https://sites.google.com/view/sang-hyeon-lee) &nbsp;&nbsp;&nbsp; [Jong Taek Lee](https://sites.google.com/view/k-vislab)†

Kyungpook National University, South Korea
<br><sub>† Corresponding author</sub>

### ACCV 2026

</div>

**TL;DR:** **FOCUS-AR** (**F**actorized **O**bservable **C**ontext **U**nder**S**tanding for **A**ction **R**ecognition) factorizes each action description into shared, motion, and appearance factors and aligns RGB and skeleton features only with the factors each modality can observe.

## Installation

```bash
conda create -n focus-ar python=3.9 -y && conda activate focus-ar
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -U huggingface_hub   # `hf` CLI for downloading the skeletons
```

CLIP ViT-B/16 weights are downloaded to `~/.cache/clip` on first use. All experiments use two GPUs (`torchrun --nproc_per_node=2`); with a different number of GPUs, gradients are accumulated to the same total batch size.

## Data preparation

Videos, skeletons, and skeleton-encoder checkpoints are expected at:

```
data/
  hmdb51/videos/*.avi          hmdb51/skeletons.pkl
  ucf101/videos/*.avi          ucf101/skeletons.pkl
  k400/videos/{train,val}/     k400/skeletons.pkl
  k600/videos/train/           k600/skeletons.pkl     # cross-dataset evaluation only
checkpoints/skeleton/
  hmdb51_base/best.pth  ucf101_base/best.pth  k400_base/best.pth  k400/best.pth
```

**1. Videos.** [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) and [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) as flat folders of `.avi` files; [Kinetics-400/600](https://github.com/cvdfoundation/kinetics-dataset) as `.mp4` files named `<youtube_id>_<start>_<end>.mp4`. Split lists are in [splits/](splits) (`<video> <label>` per line), and the class descriptions are in [labels/](labels): the LLM-generated action descriptions released by [FROSTER](https://github.com/Visual-AI/FROSTER). Videos missing from your copy of a dataset are skipped with a warning.

**2. Skeletons and skeleton encoders.** The 2D skeletons and the pretrained BlockGCN skeleton encoders are on [Hugging Face](https://huggingface.co/datasets/hyeon0819/FOCUS-AR) (skeletons 35.6 GB in total, of which Kinetics-400 is 23.1 GB). Download them into the repository root:

```bash
hf download hyeon0819/FOCUS-AR --repo-type dataset --local-dir .
# or a single benchmark, e.g. HMDB-51
hf download hyeon0819/FOCUS-AR --repo-type dataset --local-dir . --include "data/hmdb51/*"
hf download hyeon0819/FOCUS-AR --repo-type dataset --local-dir . --include "checkpoints/skeleton/hmdb51_base/*"
```

Skeletons were extracted with YOLO11x + ByteTrack + RTMPose-m. The BlockGCN skeleton encoders are kept frozen while training FOCUS-AR; to pretrain them yourself:

```bash
bash scripts/skeleton.sh   # or: python tools/train_skeleton.py --config configs/skeleton/hmdb51_base.yaml
```

## Training and evaluation

There is one config per protocol and dataset in [configs/](configs); `--opts` overrides any entry. 
Training ends with multi-view testing (3 temporal clips × 1 crop) of the final model and writes
`work_dirs/<protocol>/<run>/results.json`.

**Base-to-novel generalization** (16 videos per base class; splits 1, 2, 3)

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/base2novel/hmdb51.yaml --opts split=1
# -> work_dirs/base2novel/hmdb51_split1/{log.txt, best.pth, results.json}
```

**Few-shot recognition** (K = 2, 4, 8, 16 videos per class)

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/fewshot/ucf101.yaml --opts shot=2
```

**Cross-dataset zero-shot recognition** (train on Kinetics-400, test on UCF-101, HMDB-51 and Kinetics-600)

```bash
torchrun --nproc_per_node=2 tools/train.py --config configs/zeroshot/k400.yaml
# -> work_dirs/zeroshot/k400/final.pth (epoch-averaged, ensembled with CLIP in weight space)
```

`scripts/{base2novel,fewshot,zeroshot}.sh` run all splits of a protocol and print the table rows with `tools/summarize.py`. A trained checkpoint is evaluated with

```bash
torchrun --nproc_per_node=2 tools/test.py --config configs/base2novel/hmdb51.yaml --opts split=1 \
    --checkpoint work_dirs/base2novel/hmdb51_split1/best.pth
```

## Acknowledgements

This code builds on [TC-CLIP](https://github.com/naver-ai/tc-clip) and [FROSTER](https://github.com/Visual-AI/FROSTER).

## Citation

```bibtex
@inproceedings{lee2026focusar,
  title     = {Modality-Observable Factorized Text Alignment for RGB-Skeleton Open-Vocabulary Action Recognition},
  author    = {Lee, Sanghyeon and Lee, Jong Taek},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```
