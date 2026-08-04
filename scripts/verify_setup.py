#!/usr/bin/env python3
"""Phase 0 — Environment & Model Verification.

Run this FIRST, before any other script in the 19-class pipeline. It checks
every structural assumption the later scripts (train_19class.py,
evaluate_19class.py, diagnose_embeddings.py) rely on, and audits the real
per-language file counts in common/ so the yue-capping decision in
build_manifests_19class.py is based on measured numbers, not a guess.

Nothing here is destructive — this script only loads the pretrained model,
runs a dummy forward pass, and reads directory listings. It writes no data.

Usage:
    python scripts/verify_setup.py \\
        --data-root /home/users/ntu/birul001/scratch/data/common \\
        --lang-map configs/lang_map.csv \\
        --device cuda:0
"""

import argparse
import os
import sys

import torch

from lid_common import (
    DEFAULT_SOURCE,
    build_code_to_oldidx,
    is_audio,
    load_classifier,
    load_lang_map,
)

EXPECTED_TOTAL_PARAMS = 21_300_000  # embedding ~21.1M + classifier ~188K, derived from checkpoint sizes
PARAM_COUNT_TOLERANCE = 0.02  # 2% — these are derived estimates, not exact


def count_files(lang_dir, exts):
    if not os.path.isdir(lang_dir):
        return None
    n = 0
    for dirpath, dirnames, filenames in os.walk(lang_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        n += sum(1 for f in filenames if is_audio(f, exts))
    return n


def find_split_dirs(data_root):
    """common/ uses train/test/val (make_splits.py's naming) OR train/test/dev
    depending on which script produced it. Detect whichever exists."""
    candidates = {}
    for name in ("train", "val", "dev", "test"):
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            candidates[name] = path
    return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"HuggingFace Hub id or local model dir (default: {DEFAULT_SOURCE})")
    p.add_argument("--data-root", required=True,
                   help="dataset root containing train/val(or dev)/test split dirs")
    p.add_argument("--lang-map", required=True, help="path to lang_map.csv")
    p.add_argument("--yue-code", default="yue", help="Cantonese folder/label code")
    p.add_argument("--savedir", default=None,
                   help="local cache dir for the loaded model (default: ./pretrained_model_cache)")
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'cuda:0' etc.")
    args = p.parse_args()

    failures = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures.append(label)
        return condition

    # ---- 1. Load model ----
    print(f"Loading model from {args.source} ...")
    savedir = args.savedir or "./pretrained_model_cache"
    model = load_classifier(args.source, savedir, args.device)
    print("Model loaded.\n")

    # ---- 2. classifier.out.w shape ----
    out_linear = model.mods.classifier.out.w
    shape_ok = tuple(out_linear.weight.shape) == (107, 512)
    check("classifier.out.w.weight.shape == (107, 512)", shape_ok,
          f"got {tuple(out_linear.weight.shape)}")

    # ---- 3. Dummy forward pass: 107 log-probs ----
    torch.manual_seed(0)
    dummy = torch.randn(1, 16000 * 2)  # 2s of noise at 16kHz
    with torch.no_grad():
        emb = model.encode_batch(dummy)
        log_probs = model.mods.classifier(emb).squeeze(1)
    forward_ok = log_probs.shape[-1] == 107
    check("dummy forward pass produces 107-way output", forward_ok,
          f"got shape {tuple(log_probs.shape)}")
    if forward_ok:
        probs_sum = log_probs.exp().sum(dim=-1).item()
        check("output is a valid log-softmax (exp sums to ~1)",
              abs(probs_sum - 1.0) < 1e-3, f"sum={probs_sum:.6f}")

    # ---- 4. Total parameter count ----
    total_params = sum(param.numel() for param in model.parameters())
    rel_err = abs(total_params - EXPECTED_TOTAL_PARAMS) / EXPECTED_TOTAL_PARAMS
    check(f"total params within {PARAM_COUNT_TOLERANCE:.0%} of {EXPECTED_TOTAL_PARAMS:,}",
          rel_err <= PARAM_COUNT_TOLERANCE,
          f"got {total_params:,} ({rel_err:.2%} off)")

    # ---- 5. Module tree ----
    print("\n--- embedding_model / classifier module tree ---")
    conv_wrapper_ok = True
    for name, module in model.mods.embedding_model.named_modules():
        cls = type(module).__name__
        if cls == "Conv1d":
            print(f"  embedding_model.{name}  [{cls}]")
    for name, module in model.mods.classifier.named_modules():
        cls = type(module).__name__
        if cls in ("Linear",):
            print(f"  classifier.{name}  [{cls}]")

    # Spot-check the two documented gotchas: mfa/tdnn wrap at .conv.conv, fc is bare .conv
    has_mfa_inner = hasattr(model.mods.embedding_model.mfa, "conv") and \
        hasattr(model.mods.embedding_model.mfa.conv, "conv")
    has_fc_conv = hasattr(model.mods.embedding_model.fc, "conv") and \
        not hasattr(model.mods.embedding_model.fc.conv, "conv")
    check("embedding_model.mfa.conv.conv exists (wrapper nesting as expected)", has_mfa_inner)
    check("embedding_model.fc.conv is a bare Conv1d (no further .conv nesting)", has_fc_conv)

    # ---- 6. Label encoder / code mapping ----
    lang_map = load_lang_map(args.lang_map)
    lang_codes = [code for _, code in lang_map]
    print(f"\nlang_map.csv: {len(lang_codes)} languages: {lang_codes}")
    try:
        code_to_oldidx = build_code_to_oldidx(model)
        missing = [c for c in lang_codes if c not in code_to_oldidx]
        check("every lang_map.csv code found in the pretrained label_encoder",
              not missing, f"missing: {missing}" if missing else "")
        if args.yue_code in code_to_oldidx:
            print(f"[info] '{args.yue_code}' unexpectedly already has a pretrained index "
                  f"({code_to_oldidx[args.yue_code]}) — double check this is really new")
    except Exception as e:
        check("label_encoder readable via lab2ind", False, str(e))

    # ---- 7. Data audit ----
    print(f"\n--- Data audit: {args.data_root} ---")
    split_dirs = find_split_dirs(args.data_root)
    check("at least one split dir found (train/val/dev/test)", bool(split_dirs),
          f"found: {list(split_dirs)}")

    all_langs = lang_codes + [args.yue_code]
    counts = {split: {} for split in split_dirs}
    for split, split_dir in split_dirs.items():
        for lang in all_langs:
            counts[split][lang] = count_files(os.path.join(split_dir, lang), (".wav", ".flac",
                                                                               ".mp3", ".m4a",
                                                                               ".ogg", ".opus",
                                                                               ".sph", ".aac"))

    lang_w = max(len(l) for l in all_langs)
    header = "lang".ljust(lang_w) + "".join(s.rjust(12) for s in split_dirs) + "total".rjust(12)
    print(header)
    print("-" * len(header))
    totals = {}
    for lang in all_langs:
        row = lang.ljust(lang_w)
        total = 0
        for split in split_dirs:
            n = counts[split][lang]
            total += n or 0
            row += ("-" if n is None else f"{n:,}").rjust(12)
        row += f"{total:,}".rjust(12)
        totals[lang] = total
        print(row)

    for lang in all_langs:
        check(f"'{lang}' has audio in every found split", all(
            counts[split].get(lang, 0) for split in split_dirs
        ), f"counts: { {s: counts[s].get(lang) for s in split_dirs} }")

    yue_total = totals.get(args.yue_code, 0)
    other_totals = [totals[c] for c in lang_codes if totals.get(c)]
    if other_totals:
        median_other = sorted(other_totals)[len(other_totals) // 2]
        ratio = yue_total / median_other if median_other else float("inf")
        print(f"\nyue total: {yue_total:,}  |  median of other 17: {median_other:,}  |  "
              f"ratio: {ratio:.2f}x")
        if ratio > 2.0:
            print(f"[info] yue is >2x the median class size — build_manifests_19class.py "
                  f"should cap it (see Phase 1.2)")
        else:
            print(f"[info] yue is within 2x of the median class size — no capping needed")

    # ---- 8. CUDA ----
    cuda_ok = torch.cuda.is_available()
    status = "PASS" if cuda_ok else "INFO"
    print(f"\n[{status}] torch.cuda.is_available() = {cuda_ok}")
    if not cuda_ok:
        print("  (not a hard failure if you intend to run on CPU, but training will be slow. "
              "If this is unexpected on a GPU node, reinstall: "
              "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    if failures:
        print(f"GATE 0: FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("GATE 0: PASS — safe to proceed to build_manifests_19class.py")


if __name__ == "__main__":
    main()
