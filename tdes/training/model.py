"""A deliberately tiny decoder-only transformer.

The model is a prop.  The assignment is about the data system around it, and
the design notes is blunt about this - "training is just fake training, send to a
loop and come back".  What the model must do is be *real enough* that the
numbers the data system records mean something:

*   a genuine cross-entropy per token, so per-token perplexity is a measurement
    rather than a placeholder;
*   genuine gradients, so OPUS can score a candidate by how its gradient aligns
    with a proxy direction, and so gradient norms in the learning ledger are
    real;
*   an initial loss near ln(vocab_size), which is the free sanity check that the
    labels are shifted correctly and the loss mask is not inverted.

The one non-standard piece is that attention takes `segment_ids` and confines
itself to a single packed document.  Without that, packing would quietly teach
the model that an unrelated document is a natural continuation.
"""

from __future__ import annotations

import math

from .determinism import torch


def build_attention_bias(segment_ids, dtype):
    """Additive mask: 0 where attention is allowed, -inf where it is not.

    Combines two constraints in one tensor - causal (no looking forward) and
    block-diagonal over segments (no looking into another packed document).
    Padding attends to itself only, so the softmax stays finite; those rows are
    discarded by the loss mask anyway.
    """
    t = torch()
    batch, length = segment_ids.shape

    causal = t.tril(t.ones(length, length, dtype=t.bool)).unsqueeze(0)
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    allowed = causal & same_segment
    # keep the diagonal alive so no row is entirely masked
    eye = t.eye(length, dtype=t.bool).unsqueeze(0).expand(batch, -1, -1)
    allowed = allowed | eye

    bias = t.zeros(batch, 1, length, length, dtype=dtype)
    bias.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
    return bias


def _module_bases():
    t = torch()
    return t.nn.Module


class TinyGPT:
    """Factory wrapper so torch is imported lazily rather than at module load."""

    def __new__(cls, config):
        t = torch()
        nn = t.nn

        class _CausalSelfAttention(nn.Module):
            def __init__(self, d_model, n_head, dropout):
                super().__init__()
                assert d_model % n_head == 0
                self.n_head = n_head
                self.head_dim = d_model // n_head
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
                self.proj = nn.Linear(d_model, d_model, bias=False)
                self.dropout = dropout

            def forward(self, x, bias):
                B, L, D = x.shape
                qkv = self.qkv(x).view(B, L, 3, self.n_head, self.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores = scores + bias
                weights = t.softmax(scores, dim=-1)
                out = (weights @ v).transpose(1, 2).reshape(B, L, D)
                return self.proj(out)

        class _Block(nn.Module):
            def __init__(self, d_model, n_head, dropout):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = _CausalSelfAttention(d_model, n_head, dropout)
                self.ln2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                )

            def forward(self, x, bias):
                x = x + self.attn(self.ln1(x), bias)
                x = x + self.mlp(self.ln2(x))
                return x

        class _TinyGPT(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg
                self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
                self.pos_emb = nn.Embedding(cfg["max_position"], cfg["d_model"])
                self.blocks = nn.ModuleList(
                    [
                        _Block(cfg["d_model"], cfg["n_head"], cfg["dropout"])
                        for _ in range(cfg["n_layer"])
                    ]
                )
                self.ln_f = nn.LayerNorm(cfg["d_model"])
                self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
                # Tied embeddings: fewer parameters, and the standard choice.
                self.lm_head.weight = self.token_emb.weight
                self.apply(self._init)

            @staticmethod
            def _init(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def forward(self, token_ids, position_ids, segment_ids):
                x = self.token_emb(token_ids) + self.pos_emb(position_ids)
                bias = build_attention_bias(segment_ids, x.dtype)
                for block in self.blocks:
                    x = block(x, bias)
                return self.lm_head(self.ln_f(x))

            # -- losses ----------------------------------------------------

            def token_losses(self, token_ids, position_ids, segment_ids, loss_mask):
                """Per-target cross-entropy, plus the mask that selects targets.

                Shifted so that logits at position i predict token i+1, which is
                why `loss_mask[0]` is never a target and why the returned arrays
                are one shorter than the input.
                """
                logits = self(token_ids, position_ids, segment_ids)
                predictions = logits[:, :-1, :]
                targets = token_ids[:, 1:]
                target_mask = loss_mask[:, 1:].to(predictions.dtype)

                flat = t.nn.functional.cross_entropy(
                    predictions.reshape(-1, predictions.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).view(targets.shape)
                return flat, target_mask, targets

            def masked_loss(self, token_ids, position_ids, segment_ids, loss_mask):
                """Mean loss over loss-bearing tokens only.

                Dividing by the number of *graded* tokens rather than by the
                number of positions is what stops padding and context spans from
                diluting the number - a model can look like it is improving
                simply by being served more padding.
                """
                per_token, mask, _ = self.token_losses(
                    token_ids, position_ids, segment_ids, loss_mask
                )
                denom = mask.sum().clamp(min=1.0)
                return (per_token * mask).sum() / denom, denom

            def parameter_count(self) -> int:
                seen, total = set(), 0
                for parameter in self.parameters():
                    if id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    total += parameter.numel()
                return total

        return _TinyGPT(config)


def model_config(vocab_size: int, cfg) -> dict:
    return {
        "vocab_size": vocab_size,
        "d_model": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "dropout": cfg.dropout,
        "max_position": cfg.max_position,
    }


def expected_initial_loss(vocab_size: int) -> float:
    """ln(V): the loss of a model that has learned nothing.

    An untrained output layer is roughly uniform, so it assigns about 1/V to
    whatever token appears and the loss starts near ln(V).  A first step far
    from this means something is wrong before training began - shifted labels,
    an inverted mask, or an output layer that is not actually random.
    """
    return math.log(vocab_size)
