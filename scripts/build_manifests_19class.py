#!/usr/bin/env python3
"""Phase 1.2 — Build SpeechBrain-style CSV manifests for the 18-way head
(17 lang_map.csv languages + yue).

Reads audio directly from common/{train,val or dev,test}/<lang>/*, restricted
to the languages named in lang_map.csv plus --yue-code. Writes one CSV per
split with columns: ID, duration, wav, label.

Class balancing: per Phase 0's data audit, only caps yue if it is actually
oversized relative to the other 17 classes (>--cap-ratio times the median),
rather than capping on principle. Run verify_setup.py first so this decision
is based on measured counts, not a guess.

Usage:
    python scripts/build_manifests_19class.py \\
        --data-root /home/users/ntu/birul001/scratch/data/common \\
        --lang-map configs/lang_map.csv \\
        --out-dir /home/users/ntu/birul001/scratch/data/common/manifests_19class
"""

import argparse
import os
import random
import sys

import soundfile as sf

from lid_common import find_audio_by_lang, load_lang_map, write_manifest_csv

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")


def find_split_dirs(data_root, requested):
    dirs = {}
    for name in requested:
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            dirs[name] = path
        else:
            print(f"[warn] split dir not found, skipping: {path}", file=sys.stderr)
    return dirs


def build_rows(files, lang, id_prefix):
    rows = []
    skipped = 0
    for i, path in enumerate(files):
        try:
            info = sf.info(path)
            duration = info.frames / float(info.samplerate)
        except Exception as e:
            print(f"[warn] could not read {path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        rows.append({
            "ID": f"{id_prefix}_{lang}_{i:06d}",
            "duration": f"{duration:.4f}",
            "wav": path,
            "label": lang,
        })
    return rows, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="dataset root containing train/val(or dev)/test split dirs")
    p.add_argument("--lang-map", required=True, help="path to lang_map.csv")
    p.add_argument("--yue-code", default="yue", help="Cantonese folder/label code")
    p.add_argument("--out-dir", default=None,
                   help="manifest output dir (default: <data-root>/manifests_19class)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "dev", "test"],
                   help="candidate split dir names to look for (only existing ones are used)")
    p.add_argument("--cap-ratio", type=float, default=2.0,
                   help="cap yue if its total exceeds this multiple of the median "
                        "of the other 17 classes' totals (measure first via verify_setup.py)")
    p.add_argument("--cap-split", default="train",
                   help="which split's class sizes to use when deciding the yue cap "
                        "(capping only applies within that split; val/test are left as-is "
                        "so evaluation reflects the true class balance)")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for yue subsampling")
    args = p.parse_args()

    if not os.path.isdir(args.data_root):
        sys.exit(f"error: not a directory: {args.data_root}")

    lang_map = load_lang_map(args.lang_map)
    lang_codes = [code for _, code in lang_map]
    all_langs = lang_codes + [args.yue_code]

    split_dirs = find_split_dirs(args.data_root, args.splits)
    if not split_dirs:
        sys.exit("error: no split directories found")

    out_dir = args.out_dir or os.path.join(args.data_root, "manifests_19class")

    # Discover audio per split.
    per_split_files = {}  # {split: {lang: [paths]}}
    for split, split_dir in split_dirs.items():
        per_split_files[split] = find_audio_by_lang(split_dir, all_langs, AUDIO_EXTS)

    # Decide whether to cap yue, based on --cap-split's counts (per Phase 0 audit logic).
    cap_n = None
    if args.cap_split in per_split_files:
        counts = {lang: len(files) for lang, files in per_split_files[args.cap_split].items()}
        yue_n = counts.get(args.yue_code, 0)
        other_counts = [counts[c] for c in lang_codes if counts.get(c)]
        if other_counts and yue_n:
            other_counts.sort()
            median_other = other_counts[len(other_counts) // 2]
            ratio = yue_n / median_other if median_other else float("inf")
            print(f"[{args.cap_split}] yue={yue_n:,}  median-of-17={median_other:,}  "
                  f"ratio={ratio:.2f}x  (cap threshold={args.cap_ratio}x)")
            if ratio > args.cap_ratio:
                cap_n = int(round(median_other * args.cap_ratio))
                print(f"  -> capping yue at {cap_n:,} in '{args.cap_split}' "
                      f"(other splits left uncapped)")
            else:
                print(f"  -> no cap needed")

    rng = random.Random(args.seed)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for split, files_by_lang in per_split_files.items():
        rows = []
        skipped_total = 0
        for lang in all_langs:
            files = files_by_lang.get(lang, [])
            if not files:
                print(f"[warn] no audio found for '{lang}' in split '{split}'", file=sys.stderr)
                continue
            if lang == args.yue_code and split == args.cap_split and cap_n is not None and len(files) > cap_n:
                files = sorted(rng.sample(files, cap_n))
            lang_rows, skipped = build_rows(files, lang, id_prefix=split)
            rows.extend(lang_rows)
            skipped_total += skipped
            summary.setdefault(split, {})[lang] = len(lang_rows)

        rng.shuffle(rows)
        out_path = os.path.join(out_dir, f"{split}.csv")
        write_manifest_csv(out_path, rows)
        print(f"Wrote {out_path}  ({len(rows):,} rows"
              + (f", {skipped_total} skipped (unreadable)" if skipped_total else "") + ")")

    # Summary table.
    print("\n--- Manifest class counts ---")
    lang_w = max(len(l) for l in all_langs)
    splits_present = list(per_split_files)
    header = "lang".ljust(lang_w) + "".join(s.rjust(12) for s in splits_present)
    print(header)
    print("-" * len(header))
    for lang in all_langs:
        row = lang.ljust(lang_w)
        for split in splits_present:
            n = summary.get(split, {}).get(lang, 0)
            row += f"{n:,}".rjust(12)
        print(row)


if __name__ == "__main__":
    main()
