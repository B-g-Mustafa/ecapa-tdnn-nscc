#!/usr/bin/env python3
"""Minimal LID inference: point at an audio folder or a manifest CSV, get a
predicted language per file. No accuracy/confusion reporting — that's what
evaluate_lid.py / evaluate_19class.py are for. This just predicts.

--checkpoint omitted -> pretrained speechbrain/lang-id-voxlingua107-ecapa (107-way)
--checkpoint <path>  -> fine-tuned checkpoint (18-way)

Usage:
    python scripts/identify.py --input /path/to/audio_folder
    python scripts/identify.py --input manifests/test.csv --checkpoint ...best.pt
"""

import argparse
import csv
import os
import sys

import torch

from lid_common import DEFAULT_SOURCE, is_audio, load_classifier, load_finetuned_model, read_manifest_csv, split_label

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")


def find_files(path, exts):
    if path.endswith(".csv"):
        return [row["wav"] for row in read_manifest_csv(path)]
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        for f in sorted(filenames):
            if is_audio(f, exts):
                files.append(os.path.join(dirpath, f))
    return files


def predict(model, path, idx_to_code):
    if idx_to_code is None:  # pretrained model: its own label_encoder already works
        _, score, _, text_lab = model.classify_file(path)
        raw = text_lab[0] if isinstance(text_lab, (list, tuple)) else text_lab
        code, _ = split_label(str(raw))
        conf = float(score[0]) if hasattr(score, "__len__") else float(score)
        return code, conf

    # fine-tuned model: classifier.out was replaced, so the original
    # label_encoder no longer applies — decode via idx_to_code instead.
    import torchaudio

    device = next(model.parameters()).device
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
        probs = model.mods.classifier(emb).squeeze(1).exp()

    conf, idx = probs.max(dim=-1)
    return idx_to_code[idx.item()], float(conf.item())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="audio folder, or a manifest CSV with a 'wav' column")
    p.add_argument("--checkpoint", default=None,
                   help="train_19class.py checkpoint; omit to use the pretrained HF model instead")
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--savedir", default="./pretrained_model_cache")
    p.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    p.add_argument("--output", default=None, help="optional CSV to write path,predicted_language,confidence")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if args.checkpoint:
        print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        model, idx_to_code = load_finetuned_model(args.checkpoint, args.source, args.savedir, device)
    else:
        print(f"Loading pretrained model: {args.source}")
        model = load_classifier(args.source, args.savedir, device)
        idx_to_code = None

    files = find_files(args.input, AUDIO_EXTS)
    if not files:
        sys.exit(f"error: no audio files found under {args.input}")

    rows = []
    for path in files:
        lang, conf = predict(model, path, idx_to_code)
        print(f"{path}\t{lang}\t{conf:.4f}")
        rows.append((path, lang, conf))

    if args.output:
        with open(args.output, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "predicted_language", "confidence"])
            w.writerows(rows)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
