#!/usr/bin/env python3
"""Run language identification inference with the pretrained ECAPA-TDNN
VoxLingua107 model (speechbrain/lang-id-voxlingua107-ecapa) over a test
split and write predictions + accuracy metrics to a .txt file.

Expects the test folder to be laid out as produced by make_splits.py:

    <test_root>/<lang>/*.wav   (or other audio extensions)

where <lang> is the folder-name ground-truth label (e.g. "en", "cantonese").

Model loading follows the usage shown on the model's HuggingFace card
(https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa):

    from speechbrain.inference.classifiers import EncoderClassifier
    language_id = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp"
    )

This script calls classify_file() per audio file, which is the file-path
equivalent of the card's load_audio() + classify_batch() (verified against
the speechbrain source: classify_file loads the waveform, unsqueezes it to
a batch of 1, and calls encode_batch with rel_length=[1.0] — the same math
classify_batch does). --source defaults to the HF Hub id above, but also
accepts a local dir with hyperparams.yaml + checkpoint for a custom-trained
model.

IMPORTANT label format: the model's predicted text_lab is "code: Full Name"
(e.g. "en: English", "yue: Cantonese"), not a bare code. This script parses
out the code before the colon for comparison. Since VoxLingua107's codes
don't always match your folder names (e.g. a "cantonese" folder needs to
match against the model's "yue"), pass --lang-map to remap folder names to
the expected VoxLingua107 code:

    --lang-map lang_map.csv     # CSV with header: folder,code
                                 # e.g.  cantonese,yue

Any folder with no --lang-map entry is compared against the model code
as-is (so "en" folder vs "en" model code works with no mapping needed).

Output .txt contains an accuracy summary (overall + per-language), a
confusion breakdown, and a full per-file prediction log.

Usage:
    python scripts/evaluate_lid.py \\
        --test-root /home/users/ntu/birul001/scratch/data/common/test \\
        --output results/lid_test_results.txt \\
        --lang-map lang_map.csv \\
        --device cuda
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")
DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"


def load_lang_map(path):
    """CSV with header 'folder,code' -> {folder_name: expected_model_code}."""
    mapping = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["folder"].strip()] = row["code"].strip()
    return mapping


def split_label(text_lab):
    """'en: English' -> ('en', 'English'); falls back gracefully if no colon."""
    if ":" in text_lab:
        code, _, name = text_lab.partition(":")
        return code.strip(), name.strip()
    return text_lab.strip(), ""


def is_audio(name, exts):
    if name.startswith("._"):
        return False
    return name.lower().endswith(exts)


def find_audio_by_lang(test_root, exts):
    """Return {lang: [abs_path, ...]} for the language subfolders of test_root."""
    per_lang = {}
    langs = sorted(
        d for d in os.listdir(test_root)
        if not d.startswith(".") and os.path.isdir(os.path.join(test_root, d))
    )
    for lang in langs:
        lang_dir = os.path.join(test_root, lang)
        files = []
        for dirpath, dirnames, filenames in os.walk(lang_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith("._")]
            for f in sorted(filenames):
                if is_audio(f, exts):
                    files.append(os.path.join(dirpath, f))
        per_lang[lang] = files
    return per_lang


def load_classifier(source, savedir, device):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # older speechbrain versions

    run_opts = {"device": device} if device else None
    return EncoderClassifier.from_hparams(
        source=source, savedir=savedir, run_opts=run_opts
    )


def classify_one(classifier, path):
    """Returns (raw_label, predicted_code, confidence) for one audio file.

    raw_label is exactly what the model returns, e.g. "en: English".
    predicted_code is the part before the colon, e.g. "en".
    """
    out_prob, score, index, text_lab = classifier.classify_file(path)
    raw = text_lab[0] if isinstance(text_lab, (list, tuple)) else text_lab
    code, _ = split_label(raw)
    conf = float(score[0]) if hasattr(score, "__len__") else float(score)
    return raw, code, conf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"HuggingFace Hub id or local model dir for EncoderClassifier "
                        f"(default: {DEFAULT_SOURCE})")
    p.add_argument("--test-root", required=True,
                   help="folder containing <lang>/*.wav test subfolders")
    p.add_argument("--output", required=True, help="path to write the results .txt file")
    p.add_argument("--savedir", default=None,
                   help="local cache dir for the loaded model (default: <output_dir>/pretrained_model_cache)")
    p.add_argument("--lang-map", default=None,
                   help="CSV with header 'folder,code' mapping test folder names to the "
                        "VoxLingua107 code they should match (e.g. cantonese,yue). "
                        "Folders not listed are compared against the model code as-is.")
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'cuda:0' etc.")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS),
                   help="audio extensions to evaluate")
    p.add_argument("--limit-per-lang", type=int, default=None,
                   help="cap files per language, for a quick smoke test")
    p.add_argument("--log-every", type=int, default=200,
                   help="print progress to stderr every N files")
    args = p.parse_args()

    if not os.path.isdir(args.test_root):
        sys.exit(f"error: not a directory: {args.test_root}")

    lang_map = load_lang_map(args.lang_map) if args.lang_map else {}

    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    savedir = args.savedir or os.path.join(out_dir, "pretrained_model_cache")

    per_lang = find_audio_by_lang(args.test_root, exts)
    per_lang = {lang: files for lang, files in per_lang.items() if files}
    if not per_lang:
        sys.exit(f"error: no audio files found under {args.test_root}")

    if args.limit_per_lang:
        per_lang = {lang: files[:args.limit_per_lang] for lang, files in per_lang.items()}

    total_files = sum(len(v) for v in per_lang.values())
    print(f"Loading model from {args.source} ...", file=sys.stderr)
    classifier = load_classifier(args.source, savedir, args.device)
    print(f"Model loaded. Evaluating {total_files:,} files across "
          f"{len(per_lang)} languages ...", file=sys.stderr)

    unmapped_langs = sorted(l for l in per_lang if l not in lang_map)
    if lang_map:
        print(f"Loaded {len(lang_map)} folder->code mapping(s) from {args.lang_map}",
              file=sys.stderr)
        if unmapped_langs:
            print(f"[info] no --lang-map entry for: {', '.join(unmapped_langs)} "
                  f"— comparing folder name to model code as-is", file=sys.stderr)

    per_lang_correct = Counter()
    per_lang_total = Counter()
    confusion = defaultdict(Counter)  # confusion[true_lang][pred_code] += 1
    detail_lines = []
    failures = []

    start = time.time()
    done = 0
    for lang in sorted(per_lang):
        expected_code = lang_map.get(lang, lang)
        for path in per_lang[lang]:
            per_lang_total[lang] += 1
            try:
                raw_label, pred_code, conf = classify_one(classifier, path)
            except Exception as e:
                failures.append((path, lang, str(e)))
                raw_label, pred_code, conf = "ERROR", "ERROR", 0.0
            else:
                confusion[lang][pred_code] += 1
                if pred_code == expected_code:
                    per_lang_correct[lang] += 1
            detail_lines.append(
                f"{lang}\t{expected_code}\t{pred_code}\t{conf:.4f}\t{raw_label}\t{path}"
            )

            done += 1
            if args.log_every and done % args.log_every == 0:
                elapsed = time.time() - start
                print(f"  {done:,}/{total_files:,} files "
                      f"({elapsed:.0f}s elapsed)", file=sys.stderr)

    elapsed = time.time() - start
    overall_total = sum(per_lang_total.values())
    overall_correct = sum(per_lang_correct.values())
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    with open(args.output, "w") as fh:
        fh.write("# Language identification results\n")
        fh.write(f"# model source: {args.source}\n")
        fh.write(f"# test root:    {args.test_root}\n")
        if args.lang_map:
            fh.write(f"# lang map:     {args.lang_map}\n")
        if unmapped_langs:
            fh.write(f"# no lang-map entry (compared as-is): {', '.join(unmapped_langs)}\n")
        fh.write(f"# files evaluated: {overall_total:,}   elapsed: {elapsed:.1f}s\n")
        fh.write(f"# overall accuracy: {overall_correct:,}/{overall_total:,} "
                  f"= {overall_acc * 100:.2f}%\n")
        if failures:
            fh.write(f"# failures (could not classify): {len(failures):,}\n")
        fh.write("#\n")

        fh.write("## Per-language accuracy\n")
        lang_w = max((len(l) for l in per_lang_total), default=4)
        for lang in sorted(per_lang_total):
            n = per_lang_total[lang]
            c = per_lang_correct[lang]
            acc = c / n if n else 0.0
            fh.write(f"{lang.ljust(lang_w)}  {c:>6,}/{n:<6,}  {acc * 100:6.2f}%\n")
        fh.write("\n")

        fh.write("## Confusion summary (true_lang -> top predicted codes)\n")
        for lang in sorted(confusion):
            top = confusion[lang].most_common(5)
            top_str = ", ".join(f"{pc}={n}" for pc, n in top)
            fh.write(f"{lang.ljust(lang_w)}  {top_str}\n")
        fh.write("\n")

        if failures:
            fh.write("## Failures\n")
            for path, lang, err in failures:
                fh.write(f"{lang}\t{path}\t{err}\n")
            fh.write("\n")

        fh.write("## Per-file predictions "
                  "(true_lang, expected_code, predicted_code, confidence, raw_model_label, path)\n")
        for line in detail_lines:
            fh.write(line + "\n")

    print(f"\nOverall accuracy: {overall_correct:,}/{overall_total:,} "
          f"= {overall_acc * 100:.2f}%", file=sys.stderr)
    print(f"Wrote results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
