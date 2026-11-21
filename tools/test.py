"""Evaluate a trained FOCUS-AR checkpoint with multi-view testing.

    torchrun --nproc_per_node=2 tools/test.py --config configs/base2novel/hmdb51.yaml \
        --checkpoint work_dirs/base2novel/hmdb51_split1/best.pth --opts split=1
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focus_ar.config import load_config  # noqa: E402
from focus_ar.engine import init_distributed, is_main, load_checkpoint, logger, setup_logger  # noqa: E402
from focus_ar.models import build_model  # noqa: E402
from focus_ar.tasks import log_metrics, run_tests, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opts", nargs="*", default=[])
    parser.add_argument("--out", default=None, help="optional path of a JSON file for the metrics")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    init_distributed(cfg.seed)
    setup_logger(cfg.work_dir, "test_log.txt")
    logger.info(f"Config: {args.config} {' '.join(args.opts)} | checkpoint: {args.checkpoint}")

    model = build_model(cfg)
    model.load_skeleton_encoder(cfg.skeleton.checkpoint)
    load_checkpoint(model, args.checkpoint)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])

    metrics, _ = summarize(cfg, run_tests(cfg, model, num_clips=cfg.num_test_clips))
    log_metrics(cfg, metrics, f"Test ({cfg.num_test_clips} views)")
    if is_main() and args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
