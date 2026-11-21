#!/bin/bash
# Base-to-novel generalization (Table 2): 3 base splits for each dataset.
GPUS=${GPUS:-2}
for dataset in k400 hmdb51 ucf101; do
  for split in 1 2 3; do
    torchrun --nproc_per_node=$GPUS tools/train.py --config configs/base2novel/$dataset.yaml --opts split=$split
  done
done
python tools/summarize.py --task base2novel
