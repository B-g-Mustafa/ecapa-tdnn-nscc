#!/usr/bin/env python3
"""Fold a staged batch of audio files into an existing train/test/val split
for one language, at the same 80/10/10 ratio used for the other languages.

Meant to run after sample_cantonese.py has staged N files into a folder.
This script shuffles that staged batch (seeded, reproducible), divides it
80/10/10, and adds the files into

    <split_root>/train/<lang>/
    <split_root>/test/<lang>/
    <split_root>/val/<lang>/

Files are SYMLINKED by default (no duplication) — pass --copy if the
filesystem needs real files. Also appends rows to the existing manifest
CSVs (<split_root>/manifests_80_10_10/{train,test,val}.csv) so the record
stays consistent with make_splits.py's output.

Usage:
    python scripts/add_sampled_to_split.py \\
        /Volumes/T7/speech-lab/prof-adam/ecapa-tdnn-nscc/staging/cantonese_sample \\
        /home/users/ntu/birul001/scratch/data/common \\
        --lang cantonese --dry-run
"""

import argparse
import csv
import os
import random
import sys

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")
SPLIT_NAMES = ("train", "test", "val")


def is_audio(name, exts):
    if name.startswith("._"):
        return False
    return name.lower().endswith(exts)


def find_audio(src_dir, exts):
    files = []
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        for f in filenames:
            if is_audio(f, exts):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def divide(files, ratios, seed):
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = round(total * ratios[0])
    n_test = round(total * ratios[1])
    n_val = total - n_train - n_test  # remainder, so it always sums to total

    return {
        "train": shuffled[:n_train],
        "test": shuffled[n_train:n_train + n_test],
        "val": shuffled[n_train + n_test:],
    }


def append_manifest(manifest_dir, split, rows, lang):
    path = os.path.join(manifest_dir, f"{split}.csv")
    is_new = not os.path.exists(path)
    os.makedirs(manifest_dir, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(["path", "lang", "split", "group"])
        for path_, _ in rows:
            w.writerow([path_, lang, split, "added_batch"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("staged_dir", help="folder of staged audio files (from sample_cantonese.py)")
    p.add_argument("split_root", help="dataset root containing train/test/val (e.g. .../common)")
    p.add_argument("--lang", required=True, help="language folder name to add these files under")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "TEST", "VAL"), help="split ratios")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="audio extensions to consider")
    p.add_argument("--manifest-dir", default=None,
                   help="manifest dir to append to (default: <split_root>/manifests_80_10_10)")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking (uses more disk)")
    p.add_argument("--dry-run", action="store_true",
                   help="print counts only, touch nothing on disk")
    args = p.parse_args()

    if not os.path.isdir(args.staged_dir):
        sys.exit(f"error: not a directory: {args.staged_dir}")
    if not os.path.isdir(args.split_root):
        sys.exit(f"error: not a directory: {args.split_root}")
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"error: ratios must sum to 1.0, got {sum(args.ratios)}")

    for split in SPLIT_NAMES:
        if not os.path.isdir(os.path.join(args.split_root, split)):
            sys.exit(f"error: expected split folder missing: {os.path.join(args.split_root, split)}")

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)
    manifest_dir = args.manifest_dir or os.path.join(args.split_root, "manifests_80_10_10")

    files = find_audio(args.staged_dir, exts)
    if not files:
        sys.exit(f"error: no audio files found under {args.staged_dir}")

    assigned = divide(files, args.ratios, args.seed)

    print(f"Staged files: {len(files):,}")
    for split in SPLIT_NAMES:
        print(f"  {split}: {len(assigned[split]):,}")
    print(f"\nlang: {args.lang}   seed: {args.seed}   "
          f"ratios: {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f}")
    print(f"target: {os.path.join(args.split_root, '{train,test,val}', args.lang)}")
    print(f"manifests: {manifest_dir}")

    if args.dry_run:
        print("\n--dry-run: nothing written, nothing linked/copied")
        return

    for split in SPLIT_NAMES:
        lang_dir = os.path.join(args.split_root, split, args.lang)
        os.makedirs(lang_dir, exist_ok=True)
        rows = []
        for path in assigned[split]:
            base = os.path.basename(path)
            dest = os.path.join(lang_dir, base)
            if os.path.lexists(dest):
                stem, ext = os.path.splitext(base)
                i = 2
                while os.path.lexists(dest):
                    dest = os.path.join(lang_dir, f"{stem}__{i}{ext}")
                    i += 1
            if args.copy:
                import shutil
                shutil.copy2(path, dest)
            else:
                os.symlink(os.path.abspath(path), dest)
            rows.append((dest, path))
        append_manifest(manifest_dir, split, rows, args.lang)
        print(f"{'Copied' if args.copy else 'Symlinked'} {len(rows):,} files into {lang_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
