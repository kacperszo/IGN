"""Move a trained encoder into a fresh IGN, and freeze it if asked.

**IGN's boundary is the cleanest in the roster, and it is the authors' own.** `IGN.encode`
runs the covalent AttentiveFP, then the interaction graph, then a weighted edge readout, and
returns a genuine complex-level vector:

    atom_feats  = self.cov_graph(...)
    bond_feats3 = self.noncov_graph(...)
    h_g_sum, w  = self.readout(bond_feats3, edge_batch, num_graphs)   # 200-d per complex

`forward` is that followed by `self.FC(h_g_sum)`. So `FC` is the head and everything else is
the encoder — no pooling had to be invented, unlike SS-GNN, PLANET and EquiBind where the head
runs per edge and no complex vector exists at all. `models/ign/test_encoder_transfer.py` already
holds the boundary to a round trip.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The affinity head. Must agree with `encoder.head` for ign.torch in harness/registry.toml.
HEAD_PREFIXES = ("FC",)


def is_head(name: str) -> bool:
    """Whether a parameter belongs to the task head, matched at module boundaries."""
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    """Load an encoder file, accepting the shapes it arrives in.

    IGN's own checkpoints wrap the weights as `{"model_state_dict": ...}`, so that spelling is
    accepted alongside the harness's `{"encoder": ...}`. `weights_only=True` throughout.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False) -> dict[str, object]:
    """Overlay an encoder onto a fresh model, leaving the head at its initialisation."""
    encoder = read_encoder(path)
    target = model.state_dict()

    head_keys = [k for k in encoder if is_head(k)]
    if head_keys:
        raise SystemExit(
            f"{path} carries {len(head_keys)} head tensors ({head_keys[:3]}). That is a whole "
            f"checkpoint, not an encoder — cut it with `gnnb encoder split` first."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". Check graph_feat_size and outdim_g3 against how the encoder was trained."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the head")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Put the model in the right mode, keeping a frozen encoder deterministic.

    Not a formality here: dropout runs at 0.25 through the AttentiveFP layers and the
    interaction convolution, and both batch-norm layers keep updating their running statistics
    in train mode whatever `requires_grad` says. A head fitted against a representation that is
    re-sampled every step is a different experiment from the one `gnnb probe` measures.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()
