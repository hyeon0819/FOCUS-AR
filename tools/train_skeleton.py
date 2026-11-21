"""Pretrain the BlockGCN skeleton encoder by classifying the training classes.

    python tools/train_skeleton.py --config configs/skeleton/hmdb51_base.yaml
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focus_ar.config import load_config  # noqa: E402
from focus_ar.datasets.skeleton import SkeletonClipDataset  # noqa: E402
from focus_ar.models.blockgcn import BlockGCN  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)

    os.makedirs(cfg.work_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(cfg.work_dir, "log.txt"))])
    log = logging.getLogger()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    train_set = SkeletonClipDataset(cfg.data.skeleton_file, cfg.data.train, cfg.num_person, cfg.window, train=True)
    val_set = SkeletonClipDataset(cfg.data.skeleton_file, cfg.data.val, cfg.num_person, cfg.window, train=False)
    log.info(f"train {len(train_set)} / val {len(val_set)} clips")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True)

    model = BlockGCN(num_class=cfg.num_class, num_point=17, num_person=cfg.num_person, in_channels=3).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True,
                                weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        if epoch <= cfg.warmup_epochs:
            lr = cfg.lr * epoch / cfg.warmup_epochs
        else:
            lr = cfg.lr * 0.1 ** sum(epoch > s for s in cfg.lr_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data, label in train_loader:
            data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                logits, _ = model(data)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * label.size(0)
            correct += (logits.argmax(1) == label).sum().item()
            total += label.size(0)
        log.info(f"Epoch {epoch:03d} lr {lr:.4f} loss {total_loss / total:.4f} train acc {correct / total:.4f}")

        if epoch >= cfg.epochs - 50 or epoch % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad(), torch.autocast("cuda"):
                for data, label in val_loader:
                    logits, _ = model(data.cuda(non_blocking=True))
                    correct += (logits.argmax(1).cpu() == label).sum().item()
                    total += label.size(0)
            acc = correct / total
            log.info(f"Epoch {epoch:03d} val acc {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "epoch": epoch, "acc": acc},
                           os.path.join(cfg.work_dir, "best.pth"))
    log.info(f"Best val acc {best_acc:.4f}")


if __name__ == "__main__":
    main()
