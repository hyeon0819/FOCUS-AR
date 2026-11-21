# CLIP with a TC vision encoder.
# Adapted from CLIP (https://github.com/openai/CLIP, MIT License) and
# TC-CLIP (https://github.com/naver-ai/tc-clip, Copyright (c) 2024-present NAVER Cloud Corp., CC BY-NC 4.0).

import hashlib
import os
import urllib.request
import warnings
from typing import List, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .layers import LayerNorm, TCVisionTransformer, TextTransformer
from .tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()

_MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int, vision_width: int,
                 vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int,
                 transformer_heads: int, transformer_layers: int, vision_cfg: dict):
        super().__init__()
        self.context_length = context_length
        self.visual = TCVisionTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_width // 64, output_dim=embed_dim, cfg=vision_cfg)
        self.transformer = TextTransformer(transformer_width, transformer_layers, transformer_heads,
                                           attn_mask=self.build_attention_mask())
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


def _convert_weights_to_fp16(model: nn.Module):
    def _convert(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
        if isinstance(module, nn.MultiheadAttention):
            for attr in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
                         "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(module, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            attr = getattr(module, name, None)
            if attr is not None:
                attr.data = attr.data.half()

    model.apply(_convert)


def build_model(state_dict: dict, vision_cfg: dict, context_length: int = 77) -> CLIP:
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    vision_cfg = dict(vision_cfg, num_patches=grid_size * grid_size)

    transformer_width = state_dict["ln_final.weight"].shape[0]
    model = CLIP(
        embed_dim=state_dict["text_projection"].shape[1],
        image_resolution=vision_patch_size * grid_size,
        vision_layers=vision_layers, vision_width=vision_width, vision_patch_size=vision_patch_size,
        context_length=context_length, vocab_size=state_dict["token_embedding.weight"].shape[0],
        transformer_width=transformer_width, transformer_heads=transformer_width // 64,
        transformer_layers=len({k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")}),
        vision_cfg=vision_cfg,
    )

    # The TC vision blocks use separate q/k/v projections.
    state_dict = {k: v for k, v in state_dict.items() if k not in ("input_resolution", "context_length", "vocab_size")}
    for name in list(state_dict):
        if name.startswith("visual.transformer.resblocks") and "attn.in_proj_" in name:
            param = state_dict.pop(name)
            suffix = "weight" if name.endswith("weight") else "bias"
            prefix = name[: -len(f"in_proj_{suffix}")]
            for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                state_dict[f"{prefix}{proj}.{suffix}"] = param[i * vision_width:(i + 1) * vision_width]

    _convert_weights_to_fp16(model)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "local_global_bias_table" not in k]
    if missing:
        warnings.warn(f"CLIP weights not found for: {missing}")
    return model.eval()


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")) -> str:
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, os.path.basename(url))
    expected_sha256 = url.split("/")[-2]
    if os.path.isfile(target):
        if hashlib.sha256(open(target, "rb").read()).hexdigest() == expected_sha256:
            return target
        warnings.warn(f"{target} exists, but the SHA256 checksum does not match; re-downloading")
    with urllib.request.urlopen(url) as source, open(target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded CLIP checkpoint does not match its SHA256 checksum")
    return target


def load_clip(arch: str, vision_cfg: dict, context_length: int = 77) -> CLIP:
    """Build CLIP with the TC vision encoder and load the pretrained OpenAI weights (on CPU)."""
    path = _download(_MODELS[arch])
    try:
        state_dict = torch.jit.load(path, map_location="cpu").eval().state_dict()
    except RuntimeError:
        state_dict = torch.load(path, map_location="cpu")
    return build_model(state_dict, vision_cfg, context_length)


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot_token] + _tokenizer.encode(text) + [eot_token]
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {text} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result
