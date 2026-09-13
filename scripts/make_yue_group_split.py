#!/usr/bin/env python3
"""Rebuild the yue train/val/test assignment so that source GROUPS are disjoint
across splits. Rows for every other language are copied through untouched.

Why: yue clips come from two pools, and clip-level shuffling put the same
source in every split —

    RTHK radio : <programme/date>_<seg>_<seg>.wav  -> 458 episodes, 99.4% of
                 test episodes also in train
    corpus     : G<speaker>_S<session>.wav        -> 10 speakers, 100% overlap

so val yue accuracy measures memorisation of those episodes/speakers, not
Cantonese. That makes it useless for early stopping — which matters as soon
as the encoder is unfrozen and can actually overfit them.

This assigns whole episodes / whole speakers to exactly one split, keeping the
two pools proportionally represented in each split.

Usage:
    python scripts/make_yue_group_split.py \\
        --manifest-dir /path/to/manifests_train_19class \\
        --out-dir      /path/to/manifests_yue_grouped
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict

from lid_common import read_manifest_csv, write_manifest_csv

SPLITS = ("train", "val", "test")


def group_key(filename):
    """('corpus', 'G0051') for the structured corpus, ('radio', '<episode>')
    for RTHK clips. Episode = filename minus the trailing _<seg>_<seg>."""
    base = filename[:-4] if filename.lower().endswith(".wav") else filename
    if re.match(r"^G\d+_S\d+$", base):
        return ("corpus", base.split("_")[0])
    if re.match(r"^A\d+_S\d+", base):                      # rare singletons
        return ("corpus", "_".join(t for t in base.split("_") if t.startswith("G")) or base)
    return ("radio", re.sub(r"_\d+_\d+$", "", base))


def assign(groups, ratios, seed):
    """Distribute whole groups across splits, sized by clip count."""
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)

    total = sum(len(groups[k]) for k in keys)
    target = dict(zip(SPLITS, (total * r for r in ratios)))
    out = {s: [] for s in SPLITS}
    count = {s: 0 for s in SPLITS}

    # Largest groups first so one big episode can't blow past a small split's
    # budget; each group then goes to whichever split is furthest below target.
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        pick = max(SPLITS, key=lambda s: (target[s] - count[s]) / max(target[s], 1e-9))
        out[pick].extend(groups[k])
        count[pick] += len(groups[k])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--yue-code", default="yue")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"error: ratios must sum to 1.0, got {sum(args.ratios)}")

    rows = {}
    for s in SPLITS:
        path = os.path.join(args.manifest_dir, f"{s}.csv")
        if not os.path.exists(path):
            sys.exit(f"error: missing {path}")
        rows[s] = read_manifest_csv(path)

    # Pool yue, keep everything else where it is.
    yue, kept = [], {s: [] for s in SPLITS}
    for s in SPLITS:
        for r in rows[s]:
            (yue if r["label"] == args.yue_code else kept[s]).append(r)
    if not yue:
        sys.exit(f"error: no '{args.yue_code}' rows found in {args.manifest_dir}")

    # Group separately per pool so both are represented in every split.
    pools = defaultdict(lambda: defaultdict(list))
    for r in yue:
        kind, key = group_key(os.path.basename(r["wav"]))
        pools[kind][key].append(r)

    assigned = {s: [] for s in SPLITS}
    for kind, groups in sorted(pools.items()):
        part = assign(groups, args.ratios, args.seed)
        for s in SPLITS:
            assigned[s].extend(part[s])
        sizes = {s: len(part[s]) for s in SPLITS}
        print(f"{kind:7}: {len(groups):>4} groups, {sum(sizes.values()):>6,} clips -> "
              + "  ".join(f"{s}={sizes[s]:,}" for s in SPLITS))

    # Verify disjointness before writing — this is the whole point of the script.
    seen = {s: {group_key(os.path.basename(r["wav"])) for r in assigned[s]} for s in SPLITS}
    bad = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = seen[a] & seen[b]
        if overlap:
            bad = True
            print(f"[FAIL] {a}/{b} share {len(overlap)} groups: {sorted(overlap)[:5]}")
    if bad:
        sys.exit("error: groups leaked across splits — not writing manifests")
    print("group-disjointness verified across train/val/test")

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    for s in SPLITS:
        out = kept[s] + assigned[s]
        rng.shuffle(out)
        write_manifest_csv(os.path.join(args.out_dir, f"{s}.csv"), out)
        n_yue = len(assigned[s])
        print(f"wrote {s}.csv: {len(out):,} rows ({n_yue:,} yue, {len(kept[s]):,} other)")


if __name__ == "__main__":
    main()
