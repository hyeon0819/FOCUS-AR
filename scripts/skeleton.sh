#!/bin/bash
# Skeleton-encoder (BlockGCN) pretraining for every protocol.
for config in hmdb51_base ucf101_base k400_base k400; do
  python tools/train_skeleton.py --config configs/skeleton/$config.yaml
done
