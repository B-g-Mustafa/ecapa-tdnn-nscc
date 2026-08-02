#!/usr/bin/env python3
"""Count audio files per language for each split of the dataset.

Expected layout (as in the `common/` root):

    <root>/
        train/
            en/ ... audio files (may be nested further)
            de/ ...
        dev/
            en/ ...
            ...

Usage:
    python scripts/dataset_stats.py /path/to/common
    python scripts/dataset_stats.py /path/to/common --splits train dev test
    python scripts/dataset_stats.py /path/to/common --csv stats.csv
"""

import argparse
import csv
import os
import sys

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")


def is_audio(name, exts):
    if name.startswith("._"):  # macOS AppleDouble sidecar files
        return False
    return name.lower().endswith(exts)


def count_language(lang_dir, exts):
    """Recursively count audio files under one language directory."""
    total = 0
    for _, dirnames, filenames in os.walk(lang_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        total += sum(1 for f in filenames if is_audio(f, exts))
    return total


def list_languages(split_dir):
    return sorted(
        d
        for d in os.listdir(split_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(split_dir, d))
    )


def scan(root, splits, exts):
    """Return {split: {lang: count}} for the splits that exist under root."""
    stats = {}
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"[warn] split directory not found, skipping: {split_dir}",
                  file=sys.stderr)
            continue
        stats[split] = {
            lang: count_language(os.path.join(split_dir, lang), exts)
            for lang in list_languages(split_dir)
        }
    return stats


def print_table(stats):
    splits = list(stats)
    langs = sorted({lang for s in stats.values() for lang in s})
    if not langs:
        print("No language directories found.")
        return

    lang_w = max(max(len(l) for l in langs), len("lang"))
    col_w = max(12, max(len(s) for s in splits) + 2)

    header = "lang".ljust(lang_w) + "".join(s.rjust(col_w) for s in splits)
    header += "total".rjust(col_w)
    print(header)
    print("-" * len(header))

    for lang in langs:
        row = lang.ljust(lang_w)
        total = 0
        for split in splits:
            n = stats[split].get(lang)
            total += n or 0
            row += ("-" if n is None else f"{n:,}").rjust(col_w)
        row += f"{total:,}".rjust(col_w)
        print(row)

    print("-" * len(header))
    grand = 0
    row = "TOTAL".ljust(lang_w)
    for split in splits:
        n = sum(stats[split].values())
        grand += n
        row += f"{n:,}".rjust(col_w)
    row += f"{grand:,}".rjust(col_w)
    print(row)

    print()
    for split in splits:
        n_lang = len(stats[split])
        empty = [l for l, n in stats[split].items() if n == 0]
        print(f"{split}: {n_lang} languages, {sum(stats[split].values()):,} files"
              + (f"  (empty: {', '.join(sorted(empty))})" if empty else ""))


def print_coverage(stats, ref="train", other="dev"):
    if ref not in stats or other not in stats:
        return
    ref_langs = set(stats[ref])
    other_langs = set(stats[other])

    missing = sorted(ref_langs - other_langs)
    extra = sorted(other_langs - ref_langs)

    print()
    print(f"Languages in '{ref}' but missing from '{other}' ({len(missing)}):")
    print("  " + (", ".join(missing) if missing else "(none)"))
    if extra:
        print(f"Languages in '{other}' but not in '{ref}' ({len(extra)}):")
        print("  " + ", ".join(extra))


def write_csv(stats, path):
    splits = list(stats)
    langs = sorted({lang for s in stats.values() for lang in s})
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lang"] + splits + ["total"])
        for lang in langs:
            counts = [stats[s].get(lang, "") for s in splits]
            total = sum(c for c in counts if c != "")
            w.writerow([lang] + counts + [total])
    print(f"\nWrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="dataset root containing the split directories")
    p.add_argument("--splits", nargs="+", default=["train", "dev"],
                   help="split directory names (default: train dev)")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="audio file extensions to count")
    p.add_argument("--csv", help="also write the table to this CSV file")
    args = p.parse_args()

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)

    if not os.path.isdir(args.root):
        sys.exit(f"error: not a directory: {args.root}")

    stats = scan(args.root, args.splits, exts)
    if not stats:
        sys.exit("error: none of the requested splits were found")

    print_table(stats)
    print_coverage(stats, ref=args.splits[0],
                   other=args.splits[1] if len(args.splits) > 1 else args.splits[0])
    if args.csv:
        write_csv(stats, args.csv)


if __name__ == "__main__":
    main()
