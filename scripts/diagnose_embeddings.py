#!/usr/bin/env python3
"""Phase 2 — diagnose why Gate 1 failed: insufficient plasticity (A) vs
forgetting (B).

Only run this if train_19class.py's Phase 1 (frozen encoder) run failed
Gate 1. It extracts 256-d embeddings for the watch classes from the frozen
encoder, checks whether zh/yue are separable in embedding space at all
(independent of the trained head), and inspects the classifier's row norms
to check for the unnormalised-softmax scale-mismatch failure mode.

Requires umap-learn + matplotlib for the projection plot; both are optional
— if missing, the script still prints the numeric diagnostics (cosine
similarity, row norms) that actually drive the Phase 2 decision, and skips
only the plot.

Usage:
    python scripts/diagnose_embeddings.py \\
        --checkpoint ./runs/phase1_head_only/best.pt \\
        --val-manifest /home/.../manifests_19class/val.csv \\
        --output-dir ./runs/phase1_head_only/diagnostics \\
        --device cuda:0
"""

import argparse
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

from lid_common import DEFAULT_SOURCE, load_finetuned_model, read_manifest_csv
from train_19class import ManifestDataset, collate_fn, encode

DEFAULT_CLASSES = ("zh", "yue")


@torch.no_grad()
def extract_embeddings(model, loader, idx_to_code, device):
    """Returns (embeddings: (N,256) tensor, labels: list[str])."""
    all_emb, all_labels = [], []
    for wavs, rel_lengths, labels in loader:
        wavs, rel_lengths = wavs.to(device), rel_lengths.to(device)
        emb = encode(model, wavs, rel_lengths, has_unfrozen_encoder=False)
        emb = emb.squeeze(1)  # (batch, 256)
        all_emb.append(emb.cpu())
        all_labels.extend(idx_to_code[i] for i in labels.tolist())
    return torch.cat(all_emb, dim=0), all_labels


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--savedir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--max-samples-per-class", type=int, default=500)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    savedir = args.savedir or "./pretrained_model_cache"

    print(f"Loading fine-tuned model from {args.checkpoint} ...")
    model, idx_to_code = load_finetuned_model(args.checkpoint, args.source, savedir, device)
    code_to_idx = {c: i for i, c in enumerate(idx_to_code)}

    missing = [c for c in args.classes if c not in code_to_idx]
    if missing:
        sys.exit(f"error: classes not in this checkpoint's idx_to_code: {missing}")

    rng = random.Random(args.seed)
    rows = read_manifest_csv(args.val_manifest)
    by_class = {c: [] for c in args.classes}
    for row in rows:
        if row["label"] in by_class:
            by_class[row["label"]].append(row)
    subset = []
    for c, class_rows in by_class.items():
        if len(class_rows) > args.max_samples_per_class:
            class_rows = rng.sample(class_rows, args.max_samples_per_class)
        subset.extend(class_rows)
        print(f"  {c}: {len(class_rows)} samples")

    ds = ManifestDataset(subset, code_to_idx, chunk_seconds=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=args.num_workers)

    print(f"Extracting embeddings for {len(subset)} samples ...")
    embeddings, labels = extract_embeddings(model, loader, idx_to_code, device)

    centroids = {}
    for c in args.classes:
        idxs = [i for i, l in enumerate(labels) if l == c]
        centroids[c] = embeddings[idxs].mean(dim=0)

    print("\n--- Centroid cosine similarity ---")
    report_lines = ["# Phase 2 diagnostics\n"]
    for i, c1 in enumerate(args.classes):
        for c2 in args.classes[i + 1:]:
            sim = cosine_sim(centroids[c1], centroids[c2])
            line = f"cosine({c1}, {c2}) = {sim:.4f}"
            print(f"  {line}")
            report_lines.append(line + "\n")

    # Nearest-centroid check: for each yue-family class, is its own centroid
    # actually its nearest neighbor among the watched classes?
    print("\n--- Nearest-centroid check (per class) ---")
    for c in args.classes:
        idxs = [i for i, l in enumerate(labels) if l == c]
        sims_to_own = torch.stack([
            torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), centroids[c].unsqueeze(0))
            for i in idxs
        ]).mean().item()
        nearest_other, nearest_sim = None, -2.0
        for other in args.classes:
            if other == c:
                continue
            s = cosine_sim(centroids[c], centroids[other])
            if s > nearest_sim:
                nearest_other, nearest_sim = other, s
        line = (f"{c}: mean-sim-to-own-centroid={sims_to_own:.4f}  "
                f"nearest-other-centroid={nearest_other} (sim={nearest_sim:.4f})")
        print(f"  {line}")
        report_lines.append(line + "\n")

    # Classifier row-norm check (Weight Aligning diagnostic).
    print("\n--- Classifier row norms (all classes) ---")
    out_linear = model.mods.classifier.out.w
    row_norms = {idx_to_code[i]: out_linear.weight[i].norm().item()
                 for i in range(len(idx_to_code))}
    mean_norm_other = sum(v for k, v in row_norms.items() if k not in args.classes) / max(
        1, len(row_norms) - len(args.classes))
    report_lines.append("\nclassifier.out.w row norms:\n")
    for c in idx_to_code:
        tag = " *watched" if c in args.classes else ""
        line = f"  {c}: {row_norms[c]:.4f}{tag}"
        print(line)
        report_lines.append(line + "\n")
    report_lines.append(f"\nmean norm of non-watched rows: {mean_norm_other:.4f}\n")
    print(f"\nmean norm of non-watched rows: {mean_norm_other:.4f}")

    for c in args.classes:
        ratio = row_norms[c] / mean_norm_other if mean_norm_other else float("inf")
        flag = " <-- grew large relative to other rows; suggests Weight Aligning is needed" \
            if ratio > 1.5 else ""
        line = f"{c} row norm / mean-other-norm = {ratio:.2f}x{flag}"
        print(f"  {line}")
        report_lines.append(line + "\n")

    # UMAP projection (optional — numeric diagnostics above are what drive the decision).
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=args.seed)
        proj = reducer.fit_transform(embeddings.numpy())

        fig, ax = plt.subplots(figsize=(7, 6))
        for c in args.classes:
            idxs = [i for i, l in enumerate(labels) if l == c]
            ax.scatter(proj[idxs, 0], proj[idxs, 1], label=c, alpha=0.6, s=10)
        ax.legend()
        ax.set_title("Frozen-encoder embeddings (cosine UMAP)")
        plot_path = os.path.join(args.output_dir, "umap_projection.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved UMAP plot to {plot_path}")
        report_lines.append(f"\nUMAP plot: {plot_path}\n")
    except ImportError as e:
        print(f"\n[info] skipping UMAP plot (missing dependency: {e}). "
              f"pip install umap-learn matplotlib scikit-learn to enable it. "
              f"The numeric diagnostics above are what actually drive the Phase 2 decision.")
        report_lines.append(f"\n[UMAP skipped: missing dependency]\n")

    report_path = os.path.join(args.output_dir, "diagnostics.txt")
    with open(report_path, "w") as fh:
        fh.writelines(report_lines)
    print(f"\nWrote {report_path}")

    print("\n=== Phase 2 decision guide ===")
    print("Problem A (yue low, insufficient plasticity): if the nearest-centroid check "
          "shows yue's nearest neighbor is zh with high cosine similarity (embeddings "
          "genuinely overlap even before the head) -> proceed to Phase 3.")
    print("Problem B (zh dropped, forgetting): if embeddings are separable (low cosine "
          "similarity between zh/yue centroids) but the yue row norm is >1.5x the mean "
          "of other rows -> apply Weight Aligning (rescale classifier.out.w[yue]) and "
          "re-run evaluate_19class.py BEFORE touching the encoder.")


if __name__ == "__main__":
    main()
