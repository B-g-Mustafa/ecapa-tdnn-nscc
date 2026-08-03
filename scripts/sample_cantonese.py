#!/usr/bin/env python3
"""Randomly sample N audio files from a source folder into a staging folder.

Written for topping up the under-represented Cantonese language: pulls a
random subset out of the raw radio-scrape dump so it can later be folded
into the train/test/val split by add_sampled_to_split.py.

Files are COPIED (source is left untouched), and a manifest CSV of exactly
what was sampled is written alongside the staging folder so the operation
is reproducible / auditable.

Usage:
    python scripts/sample_cantonese.py \\
        /Volumes/T7/speech-lab/prof-adam/lang-id-cantonese/processed/audio/cantonese_radio \\
        /Volumes/T7/speech-lab/prof-adam/ecapa-tdnn-nscc/staging/cantonese_sample \\
        --n 12000 --seed 1337 --dry-run
"""

import argparse
import csv
import os
import random
import shutil
import sys

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")


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
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="folder to sample audio files from (searched recursively)")
    p.add_argument("dest", help="staging folder the sampled files are copied into")
    p.add_argument("--n", type=int, default=12000, help="number of files to sample")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed, for reproducibility")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="audio extensions to consider")
    p.add_argument("--manifest", default=None,
                   help="path to the sample manifest CSV (default: <dest>/sampled_files.csv)")
    p.add_argument("--dry-run", action="store_true",
                   help="only report how many files were found / would be sampled")
    args = p.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"error: not a directory: {args.source}")

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)

    candidates = find_audio(args.source, exts)
    print(f"Found {len(candidates):,} audio files under {args.source}")
    if len(candidates) < args.n:
        sys.exit(f"error: only {len(candidates):,} files available, cannot sample {args.n:,}")

    rng = random.Random(args.seed)
    sample = sorted(rng.sample(candidates, args.n))
    print(f"Sampled {len(sample):,} files (seed={args.seed})")

    manifest_path = args.manifest or os.path.join(args.dest, "sampled_files.csv")

    if args.dry_run:
        print(f"\n--dry-run: would copy {len(sample):,} files to {args.dest}")
        print(f"--dry-run: would write manifest to {manifest_path}")
        return

    os.makedirs(args.dest, exist_ok=True)
    rows = []
    for path in sample:
        base = os.path.basename(path)
        dest_path = os.path.join(args.dest, base)
        if os.path.lexists(dest_path):
            stem, ext = os.path.splitext(base)
            i = 2
            while os.path.lexists(dest_path):
                dest_path = os.path.join(args.dest, f"{stem}__{i}{ext}")
                i += 1
        shutil.copy2(path, dest_path)
        rows.append([path, dest_path])

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_path", "staged_path"])
        w.writerows(rows)

    print(f"Copied {len(rows):,} files to {args.dest}")
    print(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
