"""Train FOCUS-AR.

    torchrun --nproc_per_node=2 tools/train.py --config configs/base2novel/hmdb51.yaml --opts split=1
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focus_ar.config import load_config  # noqa: E402
from focus_ar.datasets import CutmixMixupBlending, build_loader, load_class_descriptions  # noqa: E402
from focus_ar.engine import (average_checkpoints, build_optimizer, build_scheduler, evaluate,  # noqa: E402
                             init_distributed, is_main, load_checkpoint, logger, save_checkpoint, setup_logger,
                             train_one_epoch, unwrap, weight_space_ensemble)
from focus_ar.models import build_init_reference, build_model  # noqa: E402
from focus_ar.tasks import log_metrics, run_tests, summarize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. split=2 optim.lr=1e-6")
    parser.add_argument("--resume", action="store_true", help="resume from <work_dir>/last.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    local_rank = init_distributed(cfg.seed)
    work_dir = cfg.work_dir
    setup_logger(work_dir)
    if is_main():
        with open(os.path.join(work_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
    logger.info(f"Config: {args.config} {' '.join(args.opts)}")

    # Model
    model = build_model(cfg)
    model.load_skeleton_encoder(cfg.skeleton.checkpoint)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank],
                                                      broadcast_buffers=False, find_unused_parameters=True)

    # Data
    train = cfg.data.train
    train_loader = build_loader(cfg, train.ann_file, train.label_file, cfg.data.video_root,
                                cfg.data.skeleton_file, train=True)
    train_classes = load_class_descriptions(train.label_file)
    mixup = CutmixMixupBlending(num_classes=len(train_classes), smoothing=cfg.loss.label_smoothing,
                                mixup_alpha=cfg.loss.mixup, cutmix_alpha=cfg.loss.cutmix,
                                switch_prob=cfg.loss.switch_prob)
    val_loader = None
    if cfg.task == "zeroshot":
        val = cfg.data.val
        val_loader = build_loader(cfg, val.ann_file, val.label_file, cfg.data.video_root,
                                  cfg.data.skeleton_file, train=False)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    start_epoch, best_score = 0, -1.0
    last_path = os.path.join(work_dir, "last.pth")
    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        unwrap(model).load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_score = ckpt["epoch"] + 1, ckpt["best_score"]
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # Cross-dataset training stops after the last epoch that enters weight averaging.
    end_epoch = cfg.average_epochs[1] + 1 if cfg.task == "zeroshot" else cfg.optim.epochs
    for epoch in range(start_epoch, end_epoch):
        train_loader.sampler.set_epoch(epoch)
        unwrap(model).set_classes(train_classes)
        train_one_epoch(cfg, model, train_loader, optimizer, scheduler, scaler, mixup, epoch)

        if cfg.task == "zeroshot":
            evaluate(cfg, model, val_loader, train_classes, num_clips=1, tag=f"[Epoch {epoch}] val")
            save_checkpoint(model, os.path.join(work_dir, f"epoch_{epoch}.pth"), epoch=epoch)
        else:
            metrics, score = summarize(cfg, run_tests(cfg, model, num_clips=1, tag=f"[Epoch {epoch}] "))
            log_metrics(cfg, metrics, f"[Epoch {epoch}]")
            if score > best_score:
                best_score = score
                save_checkpoint(model, os.path.join(work_dir, "best.pth"), epoch=epoch, metrics=metrics)
                logger.info(f"Saved best.pth (epoch {epoch})")
        if is_main():
            torch.save({"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best_score": best_score}, last_path)
        torch.distributed.barrier()

    # Final model
    if cfg.task == "zeroshot":
        first, last = cfg.average_epochs
        state = average_checkpoints([os.path.join(work_dir, f"epoch_{e}.pth") for e in range(first, last + 1)])
        state = weight_space_ensemble(state, build_init_reference(cfg), cfg.wise_alpha)
        unwrap(model).load_state_dict(state, strict=False)
        save_checkpoint(model, os.path.join(work_dir, "final.pth"))
        logger.info(f"Averaged epochs {first}-{last} and ensembled with the initial model (alpha={cfg.wise_alpha})")
    else:
        load_checkpoint(model, os.path.join(work_dir, "best.pth"))
    torch.distributed.barrier()

    # Multi-view test
    metrics, _ = summarize(cfg, run_tests(cfg, model, num_clips=cfg.num_test_clips))
    log_metrics(cfg, metrics, f"Test ({cfg.num_test_clips} views)")
    if is_main():
        with open(os.path.join(work_dir, "results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if os.path.isfile(last_path):
            os.remove(last_path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
