#!/usr/bin/env python3
"""Phase 1.4 / Phase 4 — evaluate a fine-tuned 18-way checkpoint and
calibrate the unknown-class rejection threshold.

Loads the checkpoint produced by train_19class.py, evaluates it on the test
manifest, and writes a .txt report in the same style as evaluate_lid.py:
overall accuracy, per-class recall, and a confusion summary — plus a
dedicated zh/yue/vi/th 4x4 confusion block (the four classes the measured
zero-shot baseline flagged as at risk), and the Phase 4 threshold
calibration + accuracy-vs-coverage curve.

Usage:
    python scripts/evaluate_19class.py \\
        --checkpoint ./runs/phase1_head_only/best.pt \\
        --test-manifest /home/.../manifests_19class/test.csv \\
        --val-manifest /home/.../manifests_19class/val.csv \\
        --output results/eval_19class.txt \\
        --device cuda:0
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from lid_common import DEFAULT_SOURCE, load_finetuned_model, read_manifest_csv
from train_19class import ManifestDataset, collate_fn, forward_batch

WATCH_CLASSES = ("zh", "yue", "vi", "th")


@torch.no_grad()
def run_inference(model, loader, idx_to_code, device):
    """Returns a list of (true_code, pred_code, max_prob) for every sample."""
    results = []
    for wavs, rel_lengths, labels in loader:
        wavs, rel_lengths = wavs.to(device), rel_lengths.to(device)
        log_probs = forward_batch(model, wavs, rel_lengths, has_unfrozen_encoder=False)
        probs = log_probs.exp()
        max_probs, preds = probs.max(dim=-1)
        for true_idx, pred_idx, mp in zip(labels.tolist(), preds.tolist(), max_probs.tolist()):
            results.append((idx_to_code[true_idx], idx_to_code[pred_idx], mp))
    return results


def compute_frr(val_results, tau):
    """Fraction of in-set val samples that would be wrongly rejected at this tau."""
    if not val_results:
        return None
    rejected = sum(1 for _, _, mp in val_results if mp < tau)
    return rejected / len(val_results)


def calibrate_threshold(val_results, target_frr, steps=200):
    """Largest tau such that FRR(tau) <= target_frr — maximizes out-of-set
    rejection power while staying within the false-rejection budget."""
    best_tau = 0.0
    for i in range(steps + 1):
        tau = i / steps
        frr = compute_frr(val_results, tau)
        if frr is not None and frr <= target_frr:
            best_tau = tau
    return best_tau


def accuracy_vs_coverage(test_results, steps=50):
    rows = []
    for i in range(steps + 1):
        tau = i / steps
        accepted = [(t, p) for t, p, mp in test_results if mp >= tau]
        coverage = len(accepted) / len(test_results) if test_results else 0.0
        acc = (sum(1 for t, p in accepted if t == p) / len(accepted)) if accepted else None
        rows.append((tau, coverage, acc))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--savedir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--val-manifest", default=None,
                   help="used for threshold calibration; if omitted, threshold "
                        "calibration and its section of the report are skipped")
    p.add_argument("--target-frr", type=float, default=0.05,
                   help="target false-rejection rate for calibrating the unknown threshold")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"error: checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.test_manifest):
        sys.exit(f"error: test manifest not found: {args.test_manifest}")

    device = torch.device(args.device)
    savedir = args.savedir or "./pretrained_model_cache"

    print(f"Loading fine-tuned model from {args.checkpoint} ...")
    model, idx_to_code = load_finetuned_model(args.checkpoint, args.source, savedir, device)
    code_to_idx = {c: i for i, c in enumerate(idx_to_code)}
    print(f"idx_to_code = {idx_to_code}")

    test_rows = read_manifest_csv(args.test_manifest)
    test_ds = ManifestDataset(test_rows, code_to_idx, chunk_seconds=None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers)

    print(f"Running inference on {len(test_rows):,} test files ...")
    test_results = run_inference(model, test_loader, idx_to_code, device)

    val_results = None
    if args.val_manifest and os.path.exists(args.val_manifest):
        val_rows = read_manifest_csv(args.val_manifest)
        val_ds = ManifestDataset(val_rows, code_to_idx, chunk_seconds=None)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)
        print(f"Running inference on {len(val_rows):,} val files (for threshold calibration) ...")
        val_results = run_inference(model, val_loader, idx_to_code, device)

    # ---- Aggregate metrics ----
    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = defaultdict(Counter)
    for true_code, pred_code, _ in test_results:
        per_class_total[true_code] += 1
        confusion[true_code][pred_code] += 1
        if pred_code == true_code:
            per_class_correct[true_code] += 1

    overall_total = sum(per_class_total.values())
    overall_correct = sum(per_class_correct.values())
    micro_acc = overall_correct / overall_total if overall_total else 0.0
    per_class_acc = {c: (per_class_correct[c] / per_class_total[c] if per_class_total[c] else None)
                      for c in idx_to_code}
    macro_acc = (sum(a for a in per_class_acc.values() if a is not None) /
                 sum(1 for a in per_class_acc.values() if a is not None))

    watch_present = [c for c in WATCH_CLASSES if c in idx_to_code]

    threshold_result = None
    if val_results is not None:
        tau = calibrate_threshold(val_results, args.target_frr)
        achieved_frr = compute_frr(val_results, tau)
        curve = accuracy_vs_coverage(test_results)
        threshold_result = {"tau": tau, "achieved_frr": achieved_frr, "curve": curve}

    # ---- Write report ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("# 19-class LID evaluation (18-way trained head + threshold-based unknown)\n")
        fh.write(f"# checkpoint:    {args.checkpoint}\n")
        fh.write(f"# test manifest: {args.test_manifest}\n")
        fh.write(f"# files evaluated: {overall_total:,}\n")
        fh.write(f"# overall accuracy: micro={micro_acc*100:.2f}%  macro={macro_acc*100:.2f}%\n")
        fh.write("#\n")

        fh.write("## Per-class accuracy\n")
        lang_w = max(len(c) for c in idx_to_code)
        for c in idx_to_code:
            n = per_class_total[c]
            corr = per_class_correct[c]
            acc = per_class_acc[c]
            tag = " *watch" if c in WATCH_CLASSES else ""
            acc_str = f"{acc*100:6.2f}%" if acc is not None else "   n/a"
            fh.write(f"{c.ljust(lang_w)}  {corr:>6,}/{n:<6,}  {acc_str}{tag}\n")
        fh.write("\n")

        fh.write("## Confusion summary (true -> top predicted, all classes)\n")
        for c in sorted(confusion):
            top = confusion[c].most_common(5)
            top_str = ", ".join(f"{pc}={n}" for pc, n in top)
            fh.write(f"{c.ljust(lang_w)}  {top_str}\n")
        fh.write("\n")

        if watch_present:
            fh.write(f"## Watch-class confusion block ({'/'.join(watch_present)})\n")
            header = "true".ljust(lang_w) + "".join(w.rjust(10) for w in watch_present) + "other".rjust(10)
            fh.write(header + "\n")
            for true_c in watch_present:
                row = true_c.ljust(lang_w)
                other = per_class_total[true_c]
                for pred_c in watch_present:
                    n = confusion[true_c].get(pred_c, 0)
                    other -= n
                    row += f"{n:,}".rjust(10)
                row += f"{other:,}".rjust(10)
                fh.write(row + "\n")
            fh.write("\n")

        if threshold_result is not None:
            fh.write(f"## Unknown-class threshold calibration (target FRR <= {args.target_frr:.2%})\n")
            fh.write(f"calibrated tau = {threshold_result['tau']:.4f}  "
                     f"(achieved val FRR = {threshold_result['achieved_frr']:.4f})\n\n")
            fh.write("## Accuracy-vs-coverage curve (test set, accuracy among accepted predictions)\n")
            fh.write("tau      coverage   accuracy\n")
            for tau, coverage, acc in threshold_result["curve"]:
                acc_str = f"{acc*100:6.2f}%" if acc is not None else "   n/a"
                marker = "  <- calibrated" if abs(tau - threshold_result["tau"]) < (1.0 / 100) else ""
                fh.write(f"{tau:.2f}     {coverage*100:6.2f}%    {acc_str}{marker}\n")
            fh.write("\n")

        fh.write("## Per-file predictions (true, predicted, confidence)\n")
        for true_code, pred_code, mp in test_results:
            fh.write(f"{true_code}\t{pred_code}\t{mp:.4f}\n")

    print(f"\nOverall: micro={micro_acc*100:.2f}%  macro={macro_acc*100:.2f}%")
    if watch_present:
        print("Watch classes: " + "  ".join(
            f"{c}={per_class_acc[c]*100:.2f}%" if per_class_acc[c] is not None else f"{c}=n/a"
            for c in watch_present
        ))
    if threshold_result is not None:
        print(f"Calibrated tau={threshold_result['tau']:.4f} "
              f"(val FRR={threshold_result['achieved_frr']:.4f})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
