#!/usr/bin/env python3
"""Pool the audio of each language and re-split it 80/10/10 (train/test/val).

Reads the existing split directories under the dataset root (default: the
`train` and `dev` folders of `common/`), collects every audio file per
language, then writes a fresh, reproducible 80/10/10 partition.

Output (default `<root>/splits_80_10_10/`):

    manifests/train.csv, test.csv, val.csv   # path,lang,split,group
    manifests/summary.csv                    # per-language counts
    audio/train/<lang>/...                   # symlinks, only with --link

Nothing is copied or moved by default: the manifests reference the original
files in place. Pass --link to additionally build a symlink tree, or
--copy to physically copy the files (slow, doubles disk usage).

Usage:
    python scripts/make_splits.py /path/to/common
    python scripts/make_splits.py /path/to/common --sources train dev --seed 1337
    python scripts/make_splits.py /path/to/common --group-by parent --link
    python scripts/make_splits.py /path/to/common --min-per-split 2 --dry-run
"""

import argparse
import csv
import os
import random
import re
import sys

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")
SPLIT_NAMES = ("train", "test", "val")


def is_audio(name, exts):
    if name.startswith("._"):  # macOS AppleDouble sidecar files
        return False
    return name.lower().endswith(exts)


def collect(root, sources, exts):
    """Return {lang: [abs_path, ...]} pooled over the source split dirs."""
    per_lang = {}
    for src in sources:
        src_dir = os.path.join(root, src)
        if not os.path.isdir(src_dir):
            print(f"[warn] source split not found, skipping: {src_dir}", file=sys.stderr)
            continue
        langs = sorted(
            d for d in os.listdir(src_dir)
            if not d.startswith(".") and os.path.isdir(os.path.join(src_dir, d))
        )
        for lang in langs:
            bucket = per_lang.setdefault(lang, [])
            for dirpath, dirnames, filenames in os.walk(os.path.join(src_dir, lang)):
                dirnames[:] = [d for d in dirnames if not d.startswith("._")]
                for f in sorted(filenames):
                    if is_audio(f, exts):
                        bucket.append(os.path.join(dirpath, f))
    return per_lang


def group_key(path, lang_root, mode, pattern):
    """Key that must not be split across train/test/val (speaker leakage guard)."""
    if mode == "file":
        return path
    if mode == "parent":
        rel = os.path.relpath(path, lang_root)
        parts = rel.split(os.sep)
        return parts[0] if len(parts) > 1 else path
    if mode == "regex":
        m = pattern.search(os.path.basename(path))
        if not m:
            return path
        return m.group(1) if m.groups() else m.group(0)
    raise ValueError(mode)


def split_groups(groups, ratios, seed, lang, min_per_split):
    """Assign whole groups to train/test/val, sizing by file count.

    groups: {key: [paths]}. Returns {split: [paths]}.
    """
    rng = random.Random(f"{seed}:{lang}")
    keys = sorted(groups)
    rng.shuffle(keys)

    total = sum(len(groups[k]) for k in keys)
    targets = {
        "train": total * ratios[0],
        "test": total * ratios[1],
        "val": total * ratios[2],
    }
    out = {s: [] for s in SPLIT_NAMES}
    counts = {s: 0 for s in SPLIT_NAMES}

    # Seed the small splits first so they are never starved on tiny languages.
    order = ["val", "test", "train"]
    for key in keys:
        n = len(groups[key])
        # pick the split furthest below its target, in files
        pick = max(order, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9))
        out[pick].extend(groups[key])
        counts[pick] += n

    for s in SPLIT_NAMES:
        if min_per_split and counts[s] < min_per_split:
            print(f"[warn] {lang}: only {counts[s]} file(s) in '{s}' "
                  f"(min-per-split={min_per_split}, total={total})", file=sys.stderr)
    return out


def materialize(rows, out_audio, root, mode):
    """Create a symlink ('link') or copy ('copy') tree mirroring the splits."""
    import shutil
    made = 0
    for path, lang, split, _ in rows:
        rel = os.path.relpath(path, root).replace(os.sep, "__")
        dest_dir = os.path.join(out_audio, split, lang)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, rel)
        if os.path.lexists(dest):
            continue
        if mode == "link":
            os.symlink(os.path.abspath(path), dest)
        else:
            shutil.copy2(path, dest)
        made += 1
    print(f"{'Linked' if mode == 'link' else 'Copied'} {made:,} files into {out_audio}")


def write_manifests(rows_by_split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split, rows in rows_by_split.items():
        path = os.path.join(out_dir, f"{split}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "lang", "split", "group"])
            w.writerows(rows)
        print(f"Wrote {path}  ({len(rows):,} rows)")


def write_summary(per_lang_counts, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "train", "test", "val", "total"])
        for lang in sorted(per_lang_counts):
            c = per_lang_counts[lang]
            w.writerow([lang, c["train"], c["test"], c["val"], sum(c.values())])
    print(f"Wrote {path}")


def print_table(per_lang_counts):
    langs = sorted(per_lang_counts)
    if not langs:
        print("No audio found.")
        return
    lang_w = max(max(len(l) for l in langs), 4)
    header = ("lang".ljust(lang_w) + "train".rjust(12) + "test".rjust(10)
              + "val".rjust(10) + "total".rjust(12) + "  (train%)")
    print(header)
    print("-" * len(header))
    tot = {s: 0 for s in SPLIT_NAMES}
    for lang in langs:
        c = per_lang_counts[lang]
        n = sum(c.values())
        for s in SPLIT_NAMES:
            tot[s] += c[s]
        pct = 100.0 * c["train"] / n if n else 0.0
        print(lang.ljust(lang_w) + f"{c['train']:,}".rjust(12)
              + f"{c['test']:,}".rjust(10) + f"{c['val']:,}".rjust(10)
              + f"{n:,}".rjust(12) + f"   {pct:5.1f}%")
    print("-" * len(header))
    grand = sum(tot.values())
    print("TOTAL".ljust(lang_w) + f"{tot['train']:,}".rjust(12)
          + f"{tot['test']:,}".rjust(10) + f"{tot['val']:,}".rjust(10)
          + f"{grand:,}".rjust(12))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="dataset root (e.g. .../common)")
    p.add_argument("--sources", nargs="+", default=["train", "dev"],
                   help="existing split dirs to pool from (default: train dev)")
    p.add_argument("--out", default="splits_80_10_10",
                   help="output dir, relative to root unless absolute")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "TEST", "VAL"), help="split ratios")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed (per-language)")
    p.add_argument("--group-by", choices=["file", "parent", "regex"], default="file",
                   help="unit kept intact across splits: each file (default), the "
                        "first subdirectory under the language dir, or a regex on "
                        "the filename (use for speaker-disjoint splits)")
    p.add_argument("--group-regex", default=r"^([^_-]+)",
                   help="with --group-by regex: group id = group(1) of this pattern")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="audio extensions to include")
    p.add_argument("--min-per-split", type=int, default=1,
                   help="warn if a language ends up with fewer files in a split")
    p.add_argument("--link", action="store_true", help="also build a symlink tree")
    p.add_argument("--copy", action="store_true", help="also copy the audio files")
    p.add_argument("--dry-run", action="store_true",
                   help="print the table only, write nothing")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"error: not a directory: {args.root}")
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"error: ratios must sum to 1.0, got {sum(args.ratios)}")
    if args.link and args.copy:
        sys.exit("error: use either --link or --copy, not both")

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)
    pattern = re.compile(args.group_regex)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(args.root, args.out)

    per_lang = collect(args.root, args.sources, exts)
    per_lang = {k: v for k, v in per_lang.items() if v}
    if not per_lang:
        sys.exit("error: no audio files found under the given sources")

    rows_by_split = {s: [] for s in SPLIT_NAMES}
    counts = {}
    for lang, paths in sorted(per_lang.items()):
        lang_roots = [os.path.join(args.root, s, lang) for s in args.sources]
        groups = {}
        for path in paths:
            lang_root = next((r for r in lang_roots if path.startswith(r + os.sep)),
                             os.path.dirname(path))
            key = group_key(path, lang_root, args.group_by, pattern)
            groups.setdefault(key, []).append(path)

        assigned = split_groups(groups, args.ratios, args.seed, lang,
                                args.min_per_split)
        counts[lang] = {s: len(assigned[s]) for s in SPLIT_NAMES}

        inv = {p: k for k, ps in groups.items() for p in ps}
        for split in SPLIT_NAMES:
            for path in sorted(assigned[split]):
                rows_by_split[split].append([path, lang, split, inv[path]])

    print_table(counts)
    print(f"\ngrouping: {args.group_by}   seed: {args.seed}   "
          f"ratios: {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    manifest_dir = os.path.join(out_dir, "manifests")
    write_manifests(rows_by_split, manifest_dir)
    write_summary(counts, manifest_dir)

    if args.link or args.copy:
        all_rows = [r for s in SPLIT_NAMES for r in rows_by_split[s]]
        materialize(all_rows, os.path.join(out_dir, "audio"), args.root,
                    "link" if args.link else "copy")


if __name__ == "__main__":
    main()
