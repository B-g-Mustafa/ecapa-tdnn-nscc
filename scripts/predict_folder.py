#!/usr/bin/env python3
"""Run inference (no ground truth) with the fine-tuned ECAPA-TDNN and up to
two AmberNet models over every audio file in a folder, then merge all three
sets of predictions into one side-by-side comparison report.

Two-stage design, deliberately NOT one-shot. ECAPA-TDNN runs in the
'speechbrain' conda env; AmberNet runs in the separate 'ambernet' env (per
this project's actual setup). A single process can only import what its
current interpreter's env has installed, so:

  --stage predict --model ecapa      (run inside the speechbrain env, once)
  --stage predict --model ambernet   (run inside the ambernet env, once per
                                       model — pretrained and/or fine-tuned —
                                       each with a distinct --tag)
  --stage merge                      (run in either env — no model loading,
                                       just combines the JSON files below)

Each --stage predict run writes predictions_<tag>.json under --output-dir.
--stage merge reads every predictions_*.json there and writes the final
.txt comparison. Since there's no ground truth, no accuracy is reported —
only per-file predictions and inter-model agreement.

File paths are canonicalised with os.path.realpath so the three predict
runs merge correctly even if --audio-dir was passed slightly differently
(trailing slash, symlink) between them — but you should still point all
three runs at the same underlying folder.

Usage:
    # in the speechbrain env
    python scripts/predict_folder.py --stage predict --model ecapa \\
        --audio-dir /path/to/folder --ecapa-checkpoint ./runs/phase1_head_only/best.pt \\
        --output-dir ./predictions --device cuda:0

    # in the ambernet env, once per model you want to compare
    python scripts/predict_folder.py --stage predict --model ambernet \\
        --tag ambernet_pretrained --audio-dir /path/to/folder \\
        --output-dir ./predictions --device cuda:0

    python scripts/predict_folder.py --stage predict --model ambernet \\
        --tag ambernet_finetuned --ambernet-checkpoint /path/to/finetuned.nemo \\
        --audio-dir /path/to/folder --output-dir ./predictions --device cuda:0

    # merge (either env — pure JSON processing, no model loading)
    python scripts/predict_folder.py --stage merge \\
        --output-dir ./predictions --report ./predictions/comparison.txt
"""

import argparse
import json
import os
import sys
import time

from lid_common import is_audio

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")


def find_audio_files(audio_dir, exts, recursive):
    files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(audio_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith("._")]
            for f in sorted(filenames):
                if is_audio(f, exts):
                    files.append(os.path.realpath(os.path.join(dirpath, f)))
    else:
        for f in sorted(os.listdir(audio_dir)):
            full = os.path.join(audio_dir, f)
            if os.path.isfile(full) and is_audio(f, exts):
                files.append(os.path.realpath(full))
    return files


# --------------------------------------------------------------------------
# ECAPA-TDNN (fine-tuned) — speechbrain env only
# --------------------------------------------------------------------------

def predict_ecapa_file(model, idx_to_code, path, device):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.to(device)
    rel_lengths = torch.ones(1, device=device)

    with torch.no_grad():
        feats = model.mods.compute_features(wav)
        feats = model.mods.mean_var_norm(feats, rel_lengths)
        emb = model.mods.embedding_model(feats, rel_lengths)
        log_probs = model.mods.classifier(emb).squeeze(1)
        probs = log_probs.exp()

    conf, idx = probs.max(dim=-1)
    return idx_to_code[idx.item()], float(conf.item())


def predict_ecapa_folder(args):
    import torch
    from lid_common import load_finetuned_model

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    device = torch.device(args.device)
    savedir = args.savedir or "./pretrained_model_cache"
    print(f"Loading fine-tuned ECAPA-TDNN checkpoint from {args.ecapa_checkpoint} ...")
    model, idx_to_code = load_finetuned_model(args.ecapa_checkpoint, args.source, savedir, device)

    files = find_audio_files(args.audio_dir, DEFAULT_EXTS, args.recursive)
    if not files:
        sys.exit(f"error: no audio files found under {args.audio_dir}")
    print(f"Found {len(files):,} files. Running inference ...")

    results = {}
    start = time.time()
    for i, path in enumerate(files, 1):
        try:
            pred, conf = predict_ecapa_file(model, idx_to_code, path, device)
        except Exception as e:
            pred, conf = "ERROR", None
            print(f"[warn] could not classify {path}: {e}", file=sys.stderr)
        results[path] = {"pred": pred, "confidence": conf}
        if args.log_every and i % args.log_every == 0:
            print(f"  {i:,}/{len(files):,} ({time.time() - start:.0f}s elapsed)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "predictions_ecapa.json")
    with open(out_path, "w") as fh:
        json.dump({"tag": "ecapa", "source": args.ecapa_checkpoint,
                   "labels": idx_to_code, "results": results}, fh, indent=2)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------
# AmberNet (pretrained or fine-tuned) — ambernet env only
# --------------------------------------------------------------------------

def predict_ambernet_folder(args):
    import torch
    from evaluate_ambernet import load_ambernet, predict_one

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    model, model_labels = load_ambernet(args.ambernet_model_name, args.ambernet_checkpoint, args.device)

    files = find_audio_files(args.audio_dir, DEFAULT_EXTS, args.recursive)
    if not files:
        sys.exit(f"error: no audio files found under {args.audio_dir}")
    print(f"Found {len(files):,} files. Running inference ...")

    results = {}
    start = time.time()
    for i, path in enumerate(files, 1):
        try:
            pred, raw = predict_one(model, path, args.segment_duration, args.num_segments, args.seed)
        except Exception as e:
            pred, raw = "ERROR", str(e)
        results[path] = {"pred": pred, "raw_label": raw}
        if args.log_every and i % args.log_every == 0:
            print(f"  {i:,}/{len(files):,} ({time.time() - start:.0f}s elapsed)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"predictions_{args.tag}.json")
    source_desc = args.ambernet_checkpoint or args.ambernet_model_name
    with open(out_path, "w") as fh:
        json.dump({"tag": args.tag, "source": source_desc, "labels": model_labels,
                   "results": results}, fh, indent=2)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------
# Merge — no model loading, safe in either env
# --------------------------------------------------------------------------

def merge_predictions(args):
    pred_files = sorted(
        f for f in os.listdir(args.output_dir)
        if f.startswith("predictions_") and f.endswith(".json")
    )
    if not pred_files:
        sys.exit(f"error: no predictions_*.json files found in {args.output_dir} — "
                 f"run --stage predict first")

    models = []
    for fname in pred_files:
        with open(os.path.join(args.output_dir, fname)) as fh:
            data = json.load(fh)
        models.append(data)
        print(f"Loaded {fname}: tag={data['tag']}  source={data['source']}  "
              f"{len(data['results']):,} predictions")

    tags = [m["tag"] for m in models]
    if len(set(tags)) != len(tags):
        sys.exit(f"error: duplicate tags among predictions_*.json files: {tags} — "
                 f"re-run one of the --stage predict calls with a distinct --tag")

    all_paths = sorted(set().union(*(set(m["results"]) for m in models)))

    agree_all = 0
    disagreements = []

    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        fh.write("# Multi-model LID prediction comparison (no ground truth)\n")
        for m in models:
            fh.write(f"# {m['tag']}: {m['source']}\n")
        fh.write(f"# files: {len(all_paths):,}\n#\n")

        tag_w = max(12, max(len(t) for t in tags) + 2)
        header = "file".ljust(50) + "".join(t.rjust(tag_w) for t in tags)
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")

        for path in all_paths:
            preds = []
            for m in models:
                r = m["results"].get(path)
                preds.append(r["pred"] if r is not None else "n/a")
            short_name = os.path.basename(path)
            row = short_name[:48].ljust(50) + "".join(p.rjust(tag_w) for p in preds)
            fh.write(row + "\n")

            valid_preds = [p for p in preds if p not in ("n/a", "ERROR")]
            if valid_preds and len(set(valid_preds)) == 1 and len(valid_preds) == len(models):
                agree_all += 1
            elif len(set(preds)) > 1:
                disagreements.append((path, dict(zip(tags, preds))))

        fh.write("\n## Summary\n")
        fh.write(f"all models agree: {agree_all:,}/{len(all_paths):,}\n")
        fh.write(f"disagreements or missing predictions: {len(disagreements):,}\n\n")

        if disagreements:
            fh.write("## Disagreements (full paths)\n")
            for path, preds in disagreements:
                fh.write(f"{path}\n")
                for tag, pred in preds.items():
                    fh.write(f"    {tag}: {pred}\n")

    print(f"\nAll models agree on {agree_all:,}/{len(all_paths):,} files")
    print(f"Wrote {args.report}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["predict", "merge"])
    p.add_argument("--model", choices=["ecapa", "ambernet"],
                   help="required when --stage predict")
    p.add_argument("--audio-dir", help="folder of audio files (required for --stage predict)")
    p.add_argument("--recursive", action="store_true", help="search --audio-dir recursively")
    p.add_argument("--output-dir", required=True,
                   help="where predictions_*.json are written (predict) / read from (merge)")
    p.add_argument("--device", default=None,
                   help="'cpu', 'cuda:0', etc. Default: auto-detect.")
    p.add_argument("--savedir", default=None)
    p.add_argument("--log-every", type=int, default=50)

    # ecapa-specific
    p.add_argument("--ecapa-checkpoint", help="train_19class.py checkpoint (e.g. best.pt)")
    p.add_argument("--source", default="speechbrain/lang-id-voxlingua107-ecapa")

    # ambernet-specific
    p.add_argument("--tag", default=None,
                   help="label for this ambernet run in the merged report, e.g. "
                        "'ambernet_pretrained' / 'ambernet_finetuned' — must be distinct "
                        "across your two ambernet predict runs, or the second overwrites "
                        "the first's predictions_<tag>.json")
    p.add_argument("--ambernet-model-name", default="langid_ambernet")
    p.add_argument("--ambernet-checkpoint", default=None,
                   help=".nemo or .ckpt path; omit to use the pretrained --ambernet-model-name")
    p.add_argument("--segment-duration", type=float, default=None)
    p.add_argument("--num-segments", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)

    # merge-specific
    p.add_argument("--report", default=None,
                   help="output .txt path for --stage merge (default: <output-dir>/comparison.txt)")

    args = p.parse_args()

    if args.stage == "predict":
        if not args.model:
            sys.exit("error: --model {ecapa,ambernet} is required for --stage predict")
        if not args.audio_dir:
            sys.exit("error: --audio-dir is required for --stage predict")
        if args.model == "ecapa":
            if not args.ecapa_checkpoint:
                sys.exit("error: --ecapa-checkpoint is required for --model ecapa")
            predict_ecapa_folder(args)
        else:
            if not args.tag:
                sys.exit("error: --tag is required for --model ambernet (distinguishes "
                         "this run's predictions_<tag>.json from your other ambernet run)")
            predict_ambernet_folder(args)
    else:
        if args.report is None:
            args.report = os.path.join(args.output_dir, "comparison.txt")
        merge_predictions(args)


if __name__ == "__main__":
    main()
