"""YAML configs with ``_base_`` inheritance and ``key=value`` command-line overrides."""

import copy
import os

import yaml


class Config(dict):
    """Dictionary with attribute access to (nested) keys."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = _merge(_load_yaml(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def _format_paths(obj, values):
    if isinstance(obj, str):
        return obj.format(**values)
    if isinstance(obj, dict):
        return {k: _format_paths(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_paths(v, values) for v in obj]
    return obj


def load_config(path: str, overrides=None) -> Config:
    """Load a config file. ``overrides`` are ``key.subkey=value`` strings (values parsed as YAML)."""
    cfg = _load_yaml(path)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    # Path templates such as "{split}" / "{shot}" are filled from top-level scalars.
    scalars = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    cfg["data"] = _format_paths(cfg.get("data", {}), scalars)
    if "skeleton" in cfg:
        cfg["skeleton"] = _format_paths(cfg["skeleton"], scalars)
    if "work_dir" in cfg:
        cfg["work_dir"] = _format_paths(cfg["work_dir"], scalars)
    return _to_config(cfg)
