"""Optimization, training and evaluation loops."""

import datetime
import logging
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler

logger = logging.getLogger("focus_ar")


# ---------------------------------------------------------------------------
# Distributed / logging helpers
# ---------------------------------------------------------------------------
def init_distributed(seed):
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun, e.g. `torchrun --nproc_per_node=2 tools/train.py ...`")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Packing the skeleton cache of a large dataset on rank 0 can take a while.
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = True
    return local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_logger(work_dir, filename="log.txt"):
    os.makedirs(work_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if is_main():
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(work_dir, filename))):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, path, **extra):
    if is_main():
        state = {k: v for k, v in unwrap(model).state_dict().items() if not k.startswith("skeleton_encoder.")}
        torch.save({"model": state, **extra}, path + ".tmp")
        os.replace(path + ".tmp", path)


def load_checkpoint(model, path):
    state = torch.load(path, map_location="cpu")["model"]
    unwrap(model).load_state_dict(state, strict=False)


def average_checkpoints(paths):
    """Uniform average of model weights (checkpoint averaging over epochs)."""
    avg = None
    for p in paths:
        state = torch.load(p, map_location="cpu")["model"]
        if avg is None:
            avg = {k: v.float().clone() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    return {k: v / len(paths) for k, v in avg.items()}


def weight_space_ensemble(finetuned, reference, alpha):
    """theta = (1 - alpha) * theta_CLIP + alpha * theta_finetuned for parameters inherited from CLIP."""
    out = dict(finetuned)
    for k, v in reference.items():
        out[k] = (1 - alpha) * v.float() + alpha * finetuned[k].float()
    return out


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def build_optimizer(cfg, model):
    """AdamW with separate learning rates for gates, FOCUS-AR modules, prompts, and the CLIP backbone."""
    model = unwrap(model)
    o = cfg.optim
    gate = [model.skeleton_gate.prior, model.w_mot, model.w_app]
    focus = [p for m in (model.skeleton_refine, model.skeleton_context, model.factorization, model.skeleton_gate.mlp)
             for p in m.parameters()]
    prompt = [model.prompt_learner.ctx] + list(model.text_encoder.conditioner.parameters())
    assigned = {id(p) for p in gate + focus + prompt}
    clip = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
    groups = [
        {"params": gate, "lr": o.lr_gate, "weight_decay": 0.0},
        {"params": focus, "lr": o.lr_focus, "weight_decay": o.weight_decay_focus},
        {"params": prompt, "lr": o.lr * o.lr_prompt_mult, "weight_decay": o.weight_decay},
        {"params": clip, "lr": o.lr, "weight_decay": o.weight_decay},
    ]
    return torch.optim.AdamW(groups, betas=tuple(o.betas), eps=1e-8)


def build_scheduler(cfg, optimizer, iters_per_epoch):
    o = cfg.optim
    return CosineLRScheduler(optimizer, t_initial=int(o.epochs * iters_per_epoch), lr_min=o.lr * o.lr_min_ratio,
                             warmup_lr_init=0, warmup_t=int(o.warmup_epochs * iters_per_epoch),
                             cycle_limit=1, t_in_epochs=False)


def _to_cuda(batch):
    return (batch["video"].cuda(non_blocking=True), batch["skeleton"].cuda(non_blocking=True),
            batch["has_skeleton"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True))


def train_one_epoch(cfg, model, loader, optimizer, scheduler, scaler, mixup, epoch):
    model.train()
    criterion = SoftTargetCrossEntropy()
    accum = max(1, cfg.optim.total_batch_size // (dist.get_world_size() * cfg.batch_size))
    num_steps = len(loader)
    loss_meter, start = [], time.time()
    optimizer.zero_grad()
    for idx, batch in enumerate(loader):
        video, skeleton, has_skel, label = _to_cuda(batch)
        video, target = mixup(video, label)
        last = (idx + 1) % accum == 0 or idx + 1 == num_steps
        with (model.no_sync() if not last else nullcontext()):
            with torch.autocast("cuda", enabled=cfg.use_amp):
                out = model(video, skeleton, has_skel)
                l = cfg.loss
                loss = (criterion(out["logits"], target) + l.lambda_coh * out["losses"]["coh"]
                        + l.lambda_orth * out["losses"]["orth"] + l.lambda_div * out["losses"]["div"])
            scaler.scale(loss / accum).backward()
        if last:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step_update(epoch * num_steps + idx)
        loss_meter.append(loss.detach())
        if is_main() and (idx % cfg.print_freq == 0 or idx == num_steps - 1):
            lr = optimizer.param_groups[-1]["lr"]
            logger.info(f"Epoch [{epoch}/{cfg.optim.epochs}][{idx}/{num_steps}] "
                        f"loss {torch.stack(loss_meter[-cfg.print_freq:]).mean().item():.4f} lr {lr:.2e} "
                        f"gate {out['gate'].mean().item():.3f} time {time.time() - start:.0f}s")


@torch.no_grad()
def evaluate(cfg, model, loader, descriptions, num_clips, tag=""):
    """Top-1 / top-5 accuracy; class probabilities are averaged over ``num_clips`` temporal views."""
    model.eval()
    unwrap(model).set_classes(descriptions)
    correct1 = correct5 = total = 0
    for batch in loader:
        video, skeleton, has_skel, label = _to_cuda(batch)
        B, T = video.size(0), cfg.num_frames
        video = video.view(B, num_clips, T, *video.shape[2:])
        skeleton = skeleton.view(B, 3, num_clips, T, *skeleton.shape[3:])
        prob = 0
        for i in range(num_clips):
            with torch.autocast("cuda", enabled=cfg.use_amp):
                logits = model(video[:, i], skeleton[:, :, i].contiguous(), has_skel)["logits"]
            prob = prob + logits.float().softmax(dim=-1)
        top5 = prob.topk(5, dim=-1)[1]
        correct1 += (top5[:, 0] == label).sum().item()
        correct5 += (top5 == label[:, None]).any(dim=1).sum().item()
        total += B
    stats = torch.tensor([correct1, correct5, total], dtype=torch.float64, device="cuda")
    dist.all_reduce(stats)
    acc1, acc5 = (100.0 * stats[:2] / stats[2]).tolist()
    logger.info(f"{tag} Acc@1 {acc1:.2f}  Acc@5 {acc5:.2f}  (n={int(stats[2])})")
    return acc1, acc5
