#!/usr/bin/env python3
"""Convert an existing make_splits.py-style manifest (path,lang,split,group)
into the ID,duration,wav,label format train_19class.py's ManifestDataset
expects.

Use this instead of build_manifests_19class.py when a real 80/10/10 split
already exists (e.g. server-data/manifests_80_10_10/{train,test,val}.csv) —
it's a straight schema conversion, not a re-derivation, so it preserves
whatever real class balance is already on disk. No yue-capping logic here:
verify the real per-class counts first (see verify_setup.py or just count
the source CSV) before assuming a cap is needed.

Only the 'path' column is trusted for file location. The 'group' column in
make_splits.py output can go stale after an in-place backup-and-symlink
step (path gets relocated, group doesn't) — it is read but never used to
locate audio.

Usage:
    python scripts/convert_split_manifest.py \\
        --split-manifest-dir server-data/manifests_80_10_10 \\
        --lang-map configs/lang_map.csv \\
        --yue-code yue \\
        --out-dir /home/users/ntu/birul001/scratch/data/common/manifests_19class
"""

import argparse
import csv
import os
import sys

import soundfile as sf

from lid_common import load_lang_map, write_manifest_csv

SPLIT_FILES = {"train": "train.csv", "test": "test.csv", "val": "val.csv"}


def read_split_manifest(path):
    """Reads a path,lang,split,group CSV (make_splits.py's format)."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing_cols = {"path", "lang"} - set(reader.fieldnames or [])
        if missing_cols:
            sys.exit(f"error: {path} is missing expected column(s) {missing_cols}. "
                     f"Found columns: {reader.fieldnames}. This doesn't look like a "
                     f"make_splits.py manifest.")
        for row in reader:
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-manifest-dir", required=True,
                   help="dir containing train.csv/test.csv/val.csv in path,lang,split,group format")
    p.add_argument("--lang-map", required=True)
    p.add_argument("--yue-code", default="yue")
    p.add_argument("--out-dir", required=True,
                   help="where to write the converted ID,duration,wav,label manifests")
    p.add_argument("--check-exists", action="store_true", default=True,
                   help="verify every referenced audio file actually exists before writing "
                        "(catches a stale _pre_split_backup/ early, rather than failing "
                        "file-by-file during training)")
    p.add_argument("--no-check-exists", dest="check_exists", action="store_false")
    p.add_argument("--skip-duration", action="store_true",
                   help="write duration=0.0 instead of reading every file's header. "
                        "Faster, but train_19class.py doesn't use duration for anything "
                        "besides the manifest format, so this is safe to use.")
    args = p.parse_args()

    lang_map = load_lang_map(args.lang_map)
    lang_codes = [code for _, code in lang_map]
    expected_langs = set(lang_codes) | {args.yue_code}

    os.makedirs(args.out_dir, exist_ok=True)

    for split, filename in SPLIT_FILES.items():
        src_path = os.path.join(args.split_manifest_dir, filename)
        if not os.path.exists(src_path):
            print(f"[warn] {src_path} not found, skipping '{split}'", file=sys.stderr)
            continue

        src_rows = read_split_manifest(src_path)
        print(f"{filename}: {len(src_rows):,} rows")

        unexpected_langs = {r["lang"] for r in src_rows} - expected_langs
        if unexpected_langs:
            print(f"  [info] {len(unexpected_langs)} language(s) in the source manifest are "
                  f"not in lang_map.csv/--yue-code, will be excluded: {sorted(unexpected_langs)}")

        out_rows = []
        missing_files = []
        skipped_unreadable = 0
        per_lang_count = {}

        for i, row in enumerate(src_rows):
            lang = row["lang"]
            if lang not in expected_langs:
                continue
            path = row["path"]

            if args.check_exists and not os.path.exists(path):
                missing_files.append(path)
                continue

            if args.skip_duration:
                duration = 0.0
            else:
                try:
                    info = sf.info(path)
                    duration = info.frames / float(info.samplerate)
                except Exception as e:
                    print(f"[warn] could not read {path}: {e}", file=sys.stderr)
                    skipped_unreadable += 1
                    continue

            per_lang_count[lang] = per_lang_count.get(lang, 0) + 1
            out_rows.append({
                "ID": f"{split}_{lang}_{i:07d}",
                "duration": f"{duration:.4f}",
                "wav": path,
                "label": lang,
            })

        if missing_files:
            print(f"  [warn] {len(missing_files)} file(s) referenced in {filename} do not "
                  f"exist on disk (first 3: {missing_files[:3]}). Excluded from output.",
                  file=sys.stderr)
        if skipped_unreadable:
            print(f"  [warn] {skipped_unreadable} file(s) existed but could not be read "
                  f"as audio. Excluded from output.", file=sys.stderr)

        out_path = os.path.join(args.out_dir, f"{split}.csv")
        write_manifest_csv(out_path, out_rows)
        print(f"  -> wrote {out_path} ({len(out_rows):,} rows)")
        for lang in sorted(per_lang_count):
            print(f"       {lang}: {per_lang_count[lang]:,}")

    print(f"\nDone. Point train_19class.py / evaluate_19class.py --manifest-dir at {args.out_dir}")


if __name__ == "__main__":
    main()
