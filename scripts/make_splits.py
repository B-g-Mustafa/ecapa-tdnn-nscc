#!/usr/bin/env python3
"""Pool the audio of each language and re-split it 80/10/10 (train/test/val)
IN PLACE inside the dataset root, using symlinks (no file duplication).

This REPLACES the existing split folders. Concretely, given a root like
`common/` with existing `train/`, `dev/`, `test/` folders full of real audio:

  1. The existing source folders (--sources, default: train dev test) are
     moved aside to a backup dir (default: <root>/_pre_split_backup/) so no
     data is lost or duplicated.
  2. Every audio file is pooled per language across those backed-up folders.
  3. Each language is independently re-split 80/10/10.
  4. New folders <root>/train, <root>/test, <root>/val are created, each
     containing SYMLINKS (not copies) into the backed-up original files.
  5. Manifests (path,lang,split,group) are written to
     <root>/manifests_80_10_10/{train,test,val}.csv and summary.csv.

Nothing on disk is touched unless you drop --dry-run. Re-running after a
successful run is refused (the tool detects it via _pre_split_backup) —
pass --force only if you understand what a second pass would do.

Usage:
    python scripts/make_splits.py /path/to/common --dry-run
    python scripts/make_splits.py /path/to/common
    python scripts/make_splits.py /path/to/common --sources train dev --seed 1337
    python scripts/make_splits.py /path/to/common --group-by parent
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")
SPLIT_NAMES = ("train", "test", "val")


def is_audio(name, exts):
    if name.startswith("._"):  # macOS AppleDouble sidecar files
        return False
    return name.lower().endswith(exts)


def collect(source_dirs, exts):
    """source_dirs: {source_name: path}. Returns {lang: [abs_path, ...]}."""
    per_lang = {}
    for name, src_dir in source_dirs.items():
        if not os.path.isdir(src_dir):
            print(f"[warn] source directory not found, skipping: {src_dir}", file=sys.stderr)
            continue
        langs = sorted(
            d for d in os.listdir(src_dir)
            if not d.startswith(".") and os.path.isdir(os.path.join(src_dir, d))
        )
        for lang in langs:
            bucket = per_lang.setdefault(lang, [])
            lang_dir = os.path.join(src_dir, lang)
            for dirpath, dirnames, filenames in os.walk(lang_dir):
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
    """Assign whole groups to train/test/val, sizing by file count."""
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

    for key in keys:
        n = len(groups[key])
        pick = max(SPLIT_NAMES, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9))
        out[pick].extend(groups[key])
        counts[pick] += n

    for s in SPLIT_NAMES:
        if min_per_split and counts[s] < min_per_split:
            print(f"[warn] {lang}: only {counts[s]} file(s) in '{s}' "
                  f"(min-per-split={min_per_split}, total={total})", file=sys.stderr)
    return out


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="dataset root (e.g. .../common)")
    p.add_argument("--sources", nargs="+", default=["train", "dev", "test"],
                   help="existing folders under root to pool audio from and "
                        "replace (default: train dev test)")
    p.add_argument("--backup-dir", default=None,
                   help="where the original source folders are moved before "
                        "the new split is written (default: <root>/_pre_split_backup)")
    p.add_argument("--manifest-dir", default=None,
                   help="where manifests are written (default: <root>/manifests_80_10_10)")
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
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking (uses more disk)")
    p.add_argument("--force", action="store_true",
                   help="proceed even if backup dir or target split dirs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="print the table only, touch nothing on disk")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"error: not a directory: {args.root}")
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"error: ratios must sum to 1.0, got {sum(args.ratios)}")

    root = os.path.abspath(args.root)
    backup_dir = os.path.abspath(args.backup_dir) if args.backup_dir else os.path.join(root, "_pre_split_backup")
    manifest_dir = os.path.abspath(args.manifest_dir) if args.manifest_dir else os.path.join(root, "manifests_80_10_10")
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)
    pattern = re.compile(args.group_regex)

    source_dirs = {name: os.path.join(root, name) for name in args.sources}

    # Safety checks before doing anything destructive.
    if not args.dry_run:
        if os.path.exists(backup_dir) and not args.force:
            sys.exit(f"error: backup dir already exists ({backup_dir}) — looks like this "
                      f"was already run. Pass --force to proceed anyway.")
        for split in SPLIT_NAMES:
            target = os.path.join(root, split)
            if os.path.islink(target) and not args.force:
                sys.exit(f"error: {target} is already a symlink dir (previous run's output). "
                          f"Pass --force to overwrite.")
        for split in SPLIT_NAMES:
            if split in source_dirs and os.path.islink(source_dirs[split]):
                sys.exit(f"error: source {source_dirs[split]} is a symlink, not the original "
                          f"data — refusing to pool from it.")

    per_lang = collect(source_dirs, exts)
    per_lang = {k: v for k, v in per_lang.items() if v}
    if not per_lang:
        sys.exit("error: no audio files found under the given sources")

    rows_by_split = {s: [] for s in SPLIT_NAMES}
    counts = {}
    for lang, paths in sorted(per_lang.items()):
        lang_roots = [os.path.join(d, lang) for d in source_dirs.values()]
        groups = {}
        for path in paths:
            lang_root = next((r for r in lang_roots if path.startswith(r + os.sep)),
                             os.path.dirname(path))
            key = group_key(path, lang_root, args.group_by, pattern)
            groups.setdefault(key, []).append(path)

        assigned = split_groups(groups, args.ratios, args.seed, lang, args.min_per_split)
        counts[lang] = {s: len(assigned[s]) for s in SPLIT_NAMES}

        inv = {p: k for k, ps in groups.items() for p in ps}
        for split in SPLIT_NAMES:
            for path in sorted(assigned[split]):
                rows_by_split[split].append([path, lang, split, inv[path]])

    print_table(counts)
    print(f"\ngrouping: {args.group_by}   seed: {args.seed}   "
          f"ratios: {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f}")
    print(f"sources: {', '.join(source_dirs.values())}")
    print(f"backup:  {backup_dir}")
    print(f"targets: {', '.join(os.path.join(root, s) for s in SPLIT_NAMES)}")

    if args.dry_run:
        print("\n--dry-run: nothing written, nothing moved")
        return

    # 1. Move original source folders aside so nothing is lost or duplicated.
    os.makedirs(backup_dir, exist_ok=True)
    moved_from = {}
    for name, src in source_dirs.items():
        if not os.path.isdir(src):
            continue
        dest = os.path.join(backup_dir, name)
        if os.path.exists(dest):
            sys.exit(f"error: backup target already exists: {dest}")
        shutil.move(src, dest)
        moved_from[name] = dest
        print(f"Moved {src} -> {dest}")

    # 2. Rewrite collected paths to point at the backed-up originals.
    def relocate(path):
        for name, src in source_dirs.items():
            if name in moved_from and path.startswith(src + os.sep):
                return moved_from[name] + path[len(src):]
        return path

    for split in SPLIT_NAMES:
        for row in rows_by_split[split]:
            row[0] = relocate(row[0])

    # 3. Build the new train/test/val symlink (or copy) trees.
    for split in SPLIT_NAMES:
        split_dir = os.path.join(root, split)
        if os.path.exists(split_dir) or os.path.islink(split_dir):
            if os.path.islink(split_dir) or not os.listdir(split_dir):
                if os.path.islink(split_dir):
                    os.unlink(split_dir)
                else:
                    os.rmdir(split_dir)
            elif not args.force:
                sys.exit(f"error: {split_dir} already exists and is non-empty")
        os.makedirs(split_dir, exist_ok=True)

    made = 0
    for split in SPLIT_NAMES:
        for path, lang, _, _ in rows_by_split[split]:
            lang_dir = os.path.join(root, split, lang)
            os.makedirs(lang_dir, exist_ok=True)
            dest = os.path.join(lang_dir, os.path.basename(path))
            if os.path.lexists(dest):
                # avoid collisions between files that share a basename
                stem, ext = os.path.splitext(os.path.basename(path))
                i = 2
                while os.path.lexists(dest):
                    dest = os.path.join(lang_dir, f"{stem}__{i}{ext}")
                    i += 1
            if args.copy:
                shutil.copy2(path, dest)
            else:
                os.symlink(path, dest)
            made += 1
    print(f"\n{'Copied' if args.copy else 'Symlinked'} {made:,} files into "
          f"{', '.join(os.path.join(root, s) for s in SPLIT_NAMES)}")

    # 4. Manifests.
    write_manifests(rows_by_split, manifest_dir)
    write_summary(counts, manifest_dir)


if __name__ == "__main__":
    main()
