#!/bin/bash
# Few-shot recognition (Table 3): K = 2, 4, 8, 16 labeled videos per class.
GPUS=${GPUS:-2}
for dataset in hmdb51 ucf101; do
  for shot in 2 4 8 16; do
    torchrun --nproc_per_node=$GPUS tools/train.py --config configs/fewshot/$dataset.yaml --opts shot=$shot
  done
done
python tools/summarize.py --task fewshot
