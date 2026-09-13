#!/usr/bin/env python3
"""Check every audio path in a manifest (.csv or .json) actually opens,
without decoding full files or touching a GPU. Run this FIRST on any new
manifest — especially one from someone else's pipeline — before a training
job spends walltime discovering the same thing one file at a time.

Single-process on purpose: DataLoader workers swallow which file failed
behind an opaque multiprocessing traceback (exactly what happened when this
was first hit inside train_19class.py — "soundfile.LibsndfileError:
<exception str() failed>" with no path attached). Here every failure is
attributed to its exact path and reported with repr(), not str(), since
some libsndfile errors make str(e) itself raise.

Usage:
    python scripts/validate_manifest_audio.py --manifest /path/to/lid_eval.json
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from lid_common import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help=".csv or .json manifest")
    p.add_argument("--audio-root", default=None,
                   help="joined onto any audio path in the manifest that isn't already "
                        "absolute. Needed when a manifest stores paths relative to "
                        "wherever it expects to be run from (e.g. 'data/voxlingua/...').")
    p.add_argument("--limit", type=int, default=None, help="check only the first N rows")
    p.add_argument("--show-failures", type=int, default=20,
                   help="print this many failing paths with their error (default 20)")
    p.add_argument("--fail-threshold", type=float, default=0.0,
                   help="exit 1 only if the failure RATE exceeds this fraction (default "
                        "0.0: any failure exits 1, for ad-hoc manual checks). Raise this "
                        "when calling from a job script that should tolerate a handful of "
                        "genuinely-bad files (the training pipeline skips those on its "
                        "own) but still abort loudly if e.g. --audio-root is still wrong "
                        "and most of the manifest is unreachable.")
    args = p.parse_args()

    import soundfile as sf

    rows = read_manifest(args.manifest, audio_root=args.audio_root)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Checking {len(rows):,} rows from {args.manifest} ...\n")

    ok = 0
    fail_by_ext = Counter()
    fail_by_error = Counter()
    failures = []
    missing = 0

    for i, row in enumerate(rows, 1):
        path = row["wav"]
        ext = Path(path).suffix.lower() or "(none)"
        if not Path(path).exists():
            missing += 1
            fail_by_ext[ext] += 1
            fail_by_error["FileNotFound"] += 1
            failures.append((path, "FileNotFound", "path does not exist"))
            continue
        try:
            info = sf.info(path)  # header only — no decode, cheap
            if info.frames <= 0:
                raise ValueError(f"zero-length audio (frames={info.frames})")
            ok += 1
        except Exception as e:
            fail_by_ext[ext] += 1
            fail_by_error[type(e).__name__] += 1
            failures.append((path, type(e).__name__, repr(e)))

        if i % 5000 == 0:
            print(f"  {i:,}/{len(rows):,} checked ({len(failures):,} failures so far)",
                  file=sys.stderr)

    n = len(rows)
    print(f"\n=== Summary ===")
    print(f"total:   {n:,}")
    print(f"ok:      {ok:,} ({100*ok/max(n,1):.1f}%)")
    print(f"failed:  {len(failures):,} ({100*len(failures)/max(n,1):.1f}%)")
    if missing:
        print(f"  of which missing (path doesn't exist): {missing:,}")

    if fail_by_ext:
        print(f"\nfailures by extension: {dict(fail_by_ext.most_common())}")
    if fail_by_error:
        print(f"failures by error type: {dict(fail_by_error.most_common())}")

    if failures and args.show_failures:
        print(f"\n=== First {min(args.show_failures, len(failures))} failures ===")
        for path, etype, err in failures[:args.show_failures]:
            print(f"  [{etype}] {path}\n      {err}")

    fail_rate = len(failures) / max(n, 1)
    if fail_rate > args.fail_threshold:
        print(f"\nfailure rate {fail_rate:.1%} exceeds --fail-threshold {args.fail_threshold:.1%}")
        sys.exit(1)  # non-zero exit so a calling script can gate on this


if __name__ == "__main__":
    main()
