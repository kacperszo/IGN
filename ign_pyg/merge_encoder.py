"""Make a loadable IGN checkpoint out of a pretrained encoder plus any FC head.

`predict.py` loads with `strict=True`, so an encoder on its own is refused — but the
embedding is `IGN.encode()`, which touches only cov_graph, noncov_graph and readout. The
head's weights cannot reach it. So borrowing FC from an author checkpoint produces
embeddings that are a function of the pretrained encoder and nothing else, with no training
of any kind in between.

That matters for what this is for: measuring how a pretraining draw changes the
representation. Training a head to read the encoder would put SGD back in the instrument and
SGD is where this project's run-to-run spread lives.

    python merge_encoder.py --encoder enc.pt --donor author.pth --out full.pt
"""

from __future__ import annotations

import argparse
import torch

ENCODER_PREFIXES = ("cov_graph.", "noncov_graph.", "readout.")
HEAD_PREFIX = "FC."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, help="outputs/encoder.pt from a pretrain run")
    p.add_argument("--donor", required=True, help="a full checkpoint, for its FC weights only")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    blob = torch.load(args.encoder, map_location="cpu", weights_only=True)
    enc = blob["encoder"] if isinstance(blob, dict) and "encoder" in blob else blob
    enc = {k: v for k, v in enc.items() if k.startswith(ENCODER_PREFIXES)}
    if not enc:
        raise SystemExit(f"{args.encoder} holds no cov_graph/noncov_graph/readout tensors")

    donor = torch.load(args.donor, map_location="cpu", weights_only=True)
    donor = donor.get("model_state_dict", donor)
    head = {k: v for k, v in donor.items() if k.startswith(HEAD_PREFIX)}
    if not head:
        raise SystemExit(f"{args.donor} holds no FC tensors")

    # every key the donor has outside FC must be supplied by the encoder, or the strict load
    # downstream fails with a message about the model rather than about this merge
    missing = sorted(k for k in donor if not k.startswith(HEAD_PREFIX) and k not in enc)
    if missing:
        raise SystemExit(f"encoder is missing {len(missing)} keys the model needs, "
                         f"first few: {missing[:5]}")

    merged = {**enc, **head}
    torch.save(merged, args.out)
    note = blob.get("source") if isinstance(blob, dict) else None
    print(f"{len(enc)} encoder + {len(head)} head = {len(merged)} tensors -> {args.out}"
          + (f"  [pretext: {note}]" if note else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
