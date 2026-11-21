"""FOCUS-AR: Factorized Observable Context UnderStanding for Action Recognition.

Components (section numbers follow the paper):
  3.3  RGB-skeleton visual encoding with RGB-guided skeleton refinement
  3.4  Sample-conditioned text encoding (prompt tokens attend to RGB + skeleton context)
  3.5  Modality-observable factorized text alignment (shared / motion / appearance factors)
  3.6  Factor constraints (coherence, orthogonality, diversity)
"""

import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blockgcn import BlockGCN
from .clip import load_clip, tokenize
from .clip.layers import LayerNorm, QuickGELU


# ---------------------------------------------------------------------------
# 3.4  Sample-conditioned text encoding
# ---------------------------------------------------------------------------
class PromptLearner(nn.Module):
    """Learnable context tokens ``p_c`` followed by the fixed class-description tokens."""

    def __init__(self, cfg, token_embedding: nn.Embedding):
        super().__init__()
        self.n_ctx = cfg.n_ctx
        self.context_length = cfg.context_length
        ctx_init = cfg.ctx_init.replace("_", " ")
        n_init = len(ctx_init.split())
        assert n_init == self.n_ctx, "ctx_init must have n_ctx words"
        with torch.no_grad():
            ctx = token_embedding(tokenize(ctx_init, self.context_length))[0, 1:1 + n_init].clone()
        self.ctx = nn.Parameter(ctx)
        self.prompt_prefix = ctx_init
        self.register_buffer("token_embedding_weight", token_embedding.weight.detach().clone(), persistent=False)
        self.register_buffer("token_prefix", torch.empty(0), persistent=False)
        self.register_buffer("token_suffix", torch.empty(0), persistent=False)
        self.register_buffer("tokenized_prompts", torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def set_classes(self, descriptions):
        """Tokenize ``"<ctx> <description>."`` for every class description."""
        # Split CamelCase class names (e.g. "ApplyEyeMakeup") into words.
        descriptions = [re.sub(r"([A-Z])", r" \1", d).strip().replace("_", " ") for d in descriptions]
        prompts = [f"{self.prompt_prefix} {d}." for d in descriptions]
        tokens = tokenize(prompts, self.context_length).to(self.token_embedding_weight.device)
        embedding = F.embedding(tokens, self.token_embedding_weight)
        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]
        self.tokenized_prompts = tokens

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.token_prefix.size(0), -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class PromptConditioner(nn.Module):
    """Eqs. (2)-(3): prompt tokens cross-attend to the concatenated RGB and skeleton context tokens."""

    def __init__(self, d_model: int, n_head: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.ln_1_kv = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x, context):
        kv = self.ln_1_kv(context)
        x = x + self.attn(self.ln_1(x), kv, kv, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class SampleConditionedTextEncoder(nn.Module):
    """CLIP text encoder whose prompt tokens are updated by ``PromptConditioner`` before a given layer."""

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float16 if cfg.use_amp else torch.float32
        self.context_length = cfg.context_length
        self.n_ctx = cfg.n_ctx
        self.condition_layer = cfg.condition_layer - 1

        # Initialized from the CLIP text layer it precedes.
        self.conditioner = PromptConditioner(clip_model.ln_final.weight.shape[0])
        state = clip_model.transformer.resblocks[self.condition_layer].state_dict()
        state["ln_1_kv.weight"] = state["ln_1.weight"].clone()
        state["ln_1_kv.bias"] = state["ln_1.bias"].clone()
        self.conditioner.load_state_dict(state)

    def forward(self, prompts, tokenized_prompts, context):
        """
        Args:
            prompts:           (C, L, D) prompt embeddings
            tokenized_prompts: (C, L) token ids
            context:           (K, D) RGB + skeleton context tokens of one sample
        Returns:
            tokens:       (C, L, D) token-level features after ln_final
            eos:          (C, D) projected EOS embedding (holistic text feature)
            content_mask: (C, L) 1 for description tokens (no SOS / EOS / padding)
        """
        x = prompts + self.positional_embedding[:self.context_length].type(self.dtype)
        x = x.permute(1, 0, 2)
        for i, block in enumerate(self.transformer.resblocks):
            if i == self.condition_layer:
                ctx = x[1:1 + self.n_ctx]
                ctx = self.conditioner(ctx, context.unsqueeze(1).expand(-1, ctx.size(1), -1))
                x = torch.cat([x[:1], ctx, x[1 + self.n_ctx:]], dim=0)
            x = block(x)
        x = self.ln_final(x.permute(1, 0, 2)).type(self.dtype)

        rows = torch.arange(x.shape[0], device=x.device)
        eos_idx = tokenized_prompts.argmax(dim=-1)
        eos = x[rows, eos_idx] @ self.text_projection

        content_mask = (tokenized_prompts != 0).float()
        content_mask[:, 0] = 0.0
        content_mask[rows, eos_idx] = 0.0
        return x, eos, content_mask


# ---------------------------------------------------------------------------
# 3.3  RGB-guided skeleton refinement
# ---------------------------------------------------------------------------
class SkeletonRefinement(nn.Module):
    """Eq. (1): F~S = LN(F^S + MHCA(LN(F^S), LN(F^R))) with F^S = H^S W_S."""

    def __init__(self, d_skel: int, d_model: int, n_heads: int = 4):
        super().__init__()
        self.skel_proj = nn.Linear(d_skel, d_model)
        nn.init.xavier_uniform_(self.skel_proj.weight)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, skel_tokens, rgb_tokens):
        f_s = self.skel_proj(skel_tokens)
        kv = self.ln_kv(rgb_tokens)
        return self.ln_out(f_s + self.attn(self.ln_q(f_s), kv, kv)[0])


# ---------------------------------------------------------------------------
# 3.5  Modality-observable factorization
# ---------------------------------------------------------------------------
def _factor_head(d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))


class TokenFactorization(nn.Module):
    """Eqs. (4)-(5) and (10): soft token memberships and shared / motion / appearance text factors."""

    def __init__(self, d: int):
        super().__init__()
        self.phi_rgb = nn.Linear(d, d, bias=False)   # phi^R
        self.phi_skel = nn.Linear(d, d, bias=False)  # phi^S
        self.scale = d ** -0.5
        self.temperature = nn.Parameter(torch.ones(1))
        self.head_sh = _factor_head(d)               # q_sh
        self.head_mot = _factor_head(d)              # q_mot
        self.head_app = _factor_head(d)              # q_app

    def forward(self, tokens, content_mask, eos, text_projection, v_rgb, v_skel):
        tau = self.temperature.clamp(min=0.1)
        mu_app = torch.sigmoid((self.phi_rgb(tokens) * v_rgb).sum(-1) * self.scale / tau) * content_mask
        mu_mot = torch.sigmoid((self.phi_skel(tokens) * v_skel).sum(-1) * self.scale / tau) * content_mask
        mu_sh = mu_app * mu_mot  # soft intersection

        def factor(mu, head):
            pooled = (tokens * mu.unsqueeze(-1)).sum(1) / mu.sum(1, keepdim=True).clamp(min=1e-6)
            return F.normalize(head(pooled @ text_projection.float()), dim=-1)

        e_all = F.normalize(eos, dim=-1)
        e_sh = factor(mu_sh, self.head_sh)
        e_mot = factor(mu_mot, self.head_mot)
        e_app = factor(mu_app, self.head_app)

        losses = {
            # inclusion-exclusion: e_all ~ e_mot + e_app - e_sh
            "coh": 1.0 - (F.normalize(e_mot + e_app - e_sh, dim=-1) * e_all).sum(-1).mean(),
            # modality-specific residuals are orthogonal
            "orth": ((e_mot - e_sh) * (e_app - e_sh)).sum(-1).pow(2).mean(),
            # tokens should not all collapse into the shared factor
            "div": mu_sh.sum(-1).mean(),
        }
        return e_all, e_sh, e_mot, e_app, losses


class SkeletonGate(nn.Module):
    """Per-sample skeleton gate g^S = h * sigmoid(prior) * G^S(v^S)."""

    def __init__(self, d: int, prior_init: float, bottleneck: int = 64, dropout: float = 0.2, init_bias: float = 2.0):
        super().__init__()
        self.prior = nn.Parameter(torch.tensor(float(prior_init)))
        self.mlp = nn.Sequential(nn.Linear(d, bottleneck), nn.ReLU(), nn.Dropout(dropout), nn.Linear(bottleneck, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, v_skel, has_skel):
        return torch.sigmoid(self.prior) * torch.sigmoid(self.mlp(v_skel)) * has_skel.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# FOCUS-AR
# ---------------------------------------------------------------------------
class FOCUSAR(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        d = clip_model.ln_final.weight.shape[0]
        self.dtype = torch.float16 if cfg.use_amp else torch.float32

        # Video encoder (TC-CLIP) and sample-conditioned text encoder
        self.video_encoder = clip_model.visual
        self.prompt_learner = PromptLearner(cfg, clip_model.token_embedding)
        self.text_encoder = SampleConditionedTextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        last_layer = len(clip_model.visual.transformer.resblocks) - 1
        self.context_layer = self.text_encoder.condition_layer
        self.return_layers = sorted({self.context_layer, last_layer})

        # Skeleton encoder (BlockGCN, pretrained and frozen)
        self.skeleton_encoder = BlockGCN(num_class=cfg.skeleton.num_class, num_point=17,
                                         num_person=cfg.skeleton.num_person, in_channels=3)
        for p in self.skeleton_encoder.parameters():
            p.requires_grad = False
        self.skeleton_refine = SkeletonRefinement(256, d, n_heads=4)
        self.skeleton_context = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        nn.init.xavier_uniform_(self.skeleton_context[0].weight)

        # Factorized alignment
        self.factorization = TokenFactorization(d)
        self.skeleton_gate = SkeletonGate(d, prior_init=cfg.gate.skeleton_prior_init)
        self.w_mot = nn.Parameter(torch.tensor(float(cfg.gate.motion_init)))
        self.w_app = nn.Parameter(torch.tensor(float(cfg.gate.appearance_init)))

    def load_skeleton_encoder(self, path):
        state = torch.load(path, map_location="cpu")
        state = {k: v for k, v in state.get("model", state).items() if not k.startswith("fc.")}
        self.skeleton_encoder.load_state_dict(state, strict=False)

    def set_classes(self, descriptions):
        self.prompt_learner.set_classes(descriptions)

    def train(self, mode: bool = True):
        super().train(mode)
        self.skeleton_encoder.eval()
        return self

    def forward(self, video, skeleton, has_skel):
        """
        Args:
            video:    (B, T, 3, H, W)
            skeleton: (B, 3, T, 17, M), temporally aligned with ``video``
            has_skel: (B,) bool, whether a skeleton was extracted for the sample
        Returns:
            dict with ``logits`` (B, C) and the factor ``losses``.
        """
        B = video.shape[0]
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        # 3.3 RGB / skeleton encoding
        cls_tokens, context_tokens = self.video_encoder(video.type(self.dtype), self.return_layers)
        f_rgb = cls_tokens[:, -1].float()                                  # F^R  (B, T, D)
        v_rgb = F.normalize(f_rgb.mean(dim=1), dim=-1)                     # v^R
        c_rgb = context_tokens[:, self.return_layers.index(self.context_layer)]  # C^R

        with torch.no_grad():
            _, skel_map = self.skeleton_encoder.encode(skeleton, return_map=True)
        h_skel = skel_map.mean(1).permute(0, 2, 1).float()                 # H^S  (B, T', 256)
        f_skel = self.skeleton_refine(h_skel, f_rgb)                       # F~S  (B, T', D)
        c_skel = self.skeleton_context(f_skel)                             # C^S
        v_skel = F.normalize(f_skel.mean(1), dim=-1)                       # v^S

        g_skel = self.skeleton_gate(f_skel.mean(1), has_skel)              # (B, 1)
        w_mot = torch.sigmoid(self.w_mot)
        w_app = torch.sigmoid(self.w_app)

        # 3.4 / 3.5 per-sample conditioned text encoding and factorized alignment
        logits, losses = [], []
        for i in range(B):
            context = torch.cat([c_rgb[i].detach(), c_skel[i]], dim=0)
            tokens, eos, content_mask = self.text_encoder(prompts, tokenized_prompts, context)
            e_all, e_sh, e_mot, e_app, loss_i = self.factorization(
                tokens, content_mask, eos, self.text_encoder.text_projection, v_rgb[i], v_skel[i])

            l_all = logit_scale * v_rgb[i] @ e_all.t()
            l_app = logit_scale * v_rgb[i] @ e_app.t()
            l_sh = logit_scale * v_skel[i] @ e_sh.t()
            l_mot = logit_scale * v_skel[i] @ e_mot.t()
            g = g_skel[i]
            logits.append((1.0 - g) * l_all + g * (l_sh + w_mot * l_mot) + w_app * l_app)
            losses.append(loss_i)

        losses = {k: torch.stack([l[k] for l in losses]).mean() for k in losses[0]}
        return {"logits": torch.stack(logits), "losses": losses, "gate": g_skel.detach()}


def build_model(cfg):
    clip_model = load_clip(cfg.clip_arch, vision_cfg={
        "temporal_length": cfg.num_frames,
        "context_token_k": cfg.context_token_k,
        "seed_token_a": cfg.seed_token_a,
        "local_global_bias": True,
        "tome_r": cfg.tome_r,
        "tome_d": 0,
    }, context_length=cfg.context_length)
    model = FOCUSAR(cfg, clip_model)
    return model.float()


def build_init_reference(cfg):
    """Weights at initialization (pretrained CLIP and the initial token factorization), used for
    weight-space ensembling. Prompts, the conditioner, and the skeleton modules are not ensembled."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = build_model(cfg)
    return {k: v for k, v in model.state_dict().items()
            if k.startswith(("video_encoder.", "text_encoder.transformer.", "text_encoder.positional_embedding",
                             "text_encoder.ln_final.", "text_encoder.text_projection", "logit_scale",
                             "factorization."))
            and "local_global_bias" not in k}
