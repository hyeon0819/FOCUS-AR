"""Evaluation protocols: base-to-novel, few-shot, and cross-dataset zero-shot recognition."""

import numpy as np

from .datasets import build_loader, load_class_descriptions
from .engine import evaluate, logger


def run_tests(cfg, model, num_clips, sets=None, tag=""):
    """Evaluate on every split of ``cfg.data.test``; returns ``{name: [top-1 per split]}``."""
    results = {}
    for name, test in cfg.data.test.items():
        if sets is not None and name not in sets:
            continue
        results[name] = []
        for i, split in enumerate(test.splits):
            loader = build_loader(cfg, split.ann_file, split.label_file,
                                  video_root=test.get("video_root", cfg.data.video_root),
                                  skeleton_file=test.get("skeleton_file", cfg.data.skeleton_file),
                                  train=False, num_clips=num_clips)
            descriptions = load_class_descriptions(split.label_file)
            label = f"{tag}{name}" + (f"/split{i + 1}" if len(test.splits) > 1 else "")
            acc1, _ = evaluate(cfg, model, loader, descriptions, num_clips, tag=label)
            results[name].append(acc1)
            del loader
    return results


def summarize(cfg, results):
    """Headline metrics of a protocol and the score used for checkpoint selection."""
    if cfg.task == "base2novel":
        base, novel = results["base"][0], results["novel"][0]
        hm = 2 * base * novel / (base + novel)
        return {"base": base, "novel": novel, "hm": hm}, hm
    if cfg.task == "fewshot":
        acc = results[cfg.dataset][0]
        return {"top1": acc}, acc
    metrics = {name: {"mean": float(np.mean(accs)), "std": float(np.std(accs))} for name, accs in results.items()}
    avg = float(np.mean([m["mean"] for m in metrics.values()]))
    metrics["average"] = avg
    return metrics, avg


def format_metrics(cfg, metrics):
    if cfg.task == "base2novel":
        return f"Base {metrics['base']:.2f} | Novel {metrics['novel']:.2f} | HM {metrics['hm']:.2f}"
    if cfg.task == "fewshot":
        return f"Top-1 {metrics['top1']:.2f}"
    parts = [f"{k} {v['mean']:.2f} +/- {v['std']:.2f}" for k, v in metrics.items() if k != "average"]
    return " | ".join(parts) + f" | Avg {metrics['average']:.2f}"


def log_metrics(cfg, metrics, prefix):
    logger.info(f"{prefix}: {format_metrics(cfg, metrics)}")
