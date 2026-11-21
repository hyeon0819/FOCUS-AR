#!/bin/bash
# Cross-dataset zero-shot recognition (Table 4): train on Kinetics-400, test on UCF-101, HMDB-51 and Kinetics-600.
GPUS=${GPUS:-2}
torchrun --nproc_per_node=$GPUS tools/train.py --config configs/zeroshot/k400.yaml
python tools/summarize.py --task zeroshot
