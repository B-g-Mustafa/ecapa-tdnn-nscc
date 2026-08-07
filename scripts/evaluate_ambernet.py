#!/usr/bin/env python3
"""Evaluate NVIDIA NeMo's AmberNet LID model — pretrained or your own
fine-tuned checkpoint — on the same test set used for the ECAPA-TDNN
baseline (lid_test_results.txt) and the fine-tuned 19-class model
(evaluate_19class.py), for a like-for-like comparison.

Two loading modes:
  --checkpoint path/to/model.nemo   -> ModelClass.restore_from(...)
  --checkpoint path/to/model.ckpt   -> ModelClass.load_from_checkpoint(...)
  (neither given)                   -> from_pretrained(model_name=...), the
                                        original 107-language VoxLingua107
                                        pretrained AmberNet

INTERPRETIVE CAVEAT (only applies to the pretrained, no-checkpoint path):
the stock 'langid_ambernet' model is trained on the same VoxLingua107
107-language set as the *original* pretrained ECAPA-TDNN — it has no 'yue'
class either, so it will show 0% yue recall, exactly like the zero-shot
ECAPA-TDNN baseline in lid_test_results.txt did. That is the correct
comparison for the pretrained path: AmberNet (zero-shot, 107-way) vs. the
original ECAPA-TDNN (zero-shot, 107-way) — not evaluate_19class.py's
fine-tuned checkpoint, which was specifically trained to add yue.
Once you pass --checkpoint for your own fine-tuned AmberNet, this script
reads its actual label set from the loaded model and only prints this
caveat if yue is genuinely still absent from it.

Loads the model via NeMo's own restore_from/load_from_checkpoint/
from_pretrained + get_label() (verified against NeMo's
nemo/collections/asr/models/label_models.py source):

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="langid_ambernet"
    )
    label = model.get_label(path2audio_file)   # full-utterance, single pass by default

get_label() reads and resamples the audio itself (via soundfile + librosa),
so this script does no manual audio loading — matching evaluate_lid.py's
file-by-file pattern, for the same reason: it is the best-verified, most
robust path when the exact internals can't be executed and checked here.

Usage:
    # pretrained, zero-shot
    python scripts/evaluate_ambernet.py \\
        --test-root /home/users/ntu/birul001/scratch/data/common/test \\
        --lang-map configs/lang_map.csv \\
        --output results/ambernet_eval.txt \\
        --device cuda:0

    # your fine-tuned checkpoint
    python scripts/evaluate_ambernet.py \\
        --checkpoint /path/to/ambernet_finetuned.nemo \\
        --test-root /home/users/ntu/birul001/scratch/data/common/test \\
        --lang-map configs/lang_map.csv \\
        --output results/ambernet_finetuned_eval.txt \\
        --device cuda:0
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

from lid_common import find_audio_by_lang, load_lang_map, split_label

WATCH_CLASSES = ("zh", "yue", "vi", "th")


def load_ambernet(model_name, checkpoint, device):
    import nemo.collections.asr as nemo_asr

    ModelClass = nemo_asr.models.EncDecSpeakerLabelModel

    if checkpoint:
        ext = os.path.splitext(checkpoint)[1].lower()
        print(f"Loading fine-tuned AmberNet checkpoint from {checkpoint} ...")
        if ext == ".nemo":
            model = ModelClass.restore_from(restore_path=checkpoint)
        elif ext in (".ckpt", ".pt", ".pth"):
            model = ModelClass.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            raise ValueError(
                f"unrecognized checkpoint extension '{ext}' for {checkpoint} — "
                f"expected .nemo (restore_from) or .ckpt/.pt/.pth (load_from_checkpoint)"
            )
    else:
        print(f"Loading pretrained NeMo model '{model_name}' ...")
        model = ModelClass.from_pretrained(model_name=model_name)

    model = model.to(device)
    model.eval()

    labels = None
    try:
        labels = list(model._cfg["train_ds"].get("labels", None) or [])
        if labels:
            print(f"Model label set ({len(labels)}): {labels}")
        else:
            labels = None
    except Exception as e:
        print(f"[warn] could not read the model's label set from cfg.train_ds.labels: {e}")
    return model, labels


def predict_one(model, path, segment_duration, num_segments, seed):
    import numpy as np

    raw_label = model.get_label(
        path,
        segment_duration=segment_duration if segment_duration is not None else np.inf,
        num_segments=num_segments,
        random_seed=seed,
    )
    code, _ = split_label(str(raw_label))
    return code, str(raw_label)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="langid_ambernet",
                   help="NGC/NeMo pretrained model name (ignored if --checkpoint is given)")
    p.add_argument("--checkpoint", default=None,
                   help="path to your own fine-tuned AmberNet checkpoint: a .nemo file "
                        "(loaded via restore_from) or a .ckpt/.pt/.pth file (loaded via "
                        "load_from_checkpoint). If omitted, loads the pretrained "
                        "--model-name from NGC instead.")
    p.add_argument("--test-root", required=True,
                   help="folder containing <lang>/*.wav test subfolders "
                        "(same layout as common/test)")
    p.add_argument("--lang-map", required=True, help="path to lang_map.csv (the 17 languages)")
    p.add_argument("--yue-code", default="yue", help="Cantonese folder/label code")
    p.add_argument("--device", default=None,
                   help="'cpu', 'cuda', 'cuda:0', etc. Default: auto-detect — "
                        "cuda:0 if available, else cpu.")
    p.add_argument("--segment-duration", type=float, default=None,
                   help="seconds per segment for get_label(); default (None) uses the "
                        "whole utterance in one pass, matching evaluate_lid.py/"
                        "evaluate_19class.py's file-by-file protocol")
    p.add_argument("--num-segments", type=int, default=1,
                   help="number of random segments to majority-vote over per file "
                        "(only matters if --segment-duration is set below the "
                        "utterance length)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--limit-per-lang", type=int, default=None,
                   help="cap files per language, for a quick smoke test")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--output", required=True, help="path to write the results .txt file")
    args = p.parse_args()

    if not os.path.isdir(args.test_root):
        sys.exit(f"error: not a directory: {args.test_root}")

    import torch
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    lang_map = load_lang_map(args.lang_map)
    lang_codes = [code for _, code in lang_map]
    all_langs = lang_codes + [args.yue_code]

    per_lang = find_audio_by_lang(args.test_root, all_langs)
    per_lang = {lang: files for lang, files in per_lang.items() if files}
    if not per_lang:
        sys.exit(f"error: no audio files found under {args.test_root} for {all_langs}")

    if args.limit_per_lang:
        per_lang = {lang: files[:args.limit_per_lang] for lang, files in per_lang.items()}

    total_files = sum(len(v) for v in per_lang.values())
    model, model_labels = load_ambernet(args.model_name, args.checkpoint, args.device)
    print(f"Model loaded. Evaluating {total_files:,} files across {len(per_lang)} languages ...")

    yue_in_model = model_labels is not None and args.yue_code in model_labels

    per_class_correct = Counter()
    per_class_total = Counter()
    confusion = defaultdict(Counter)
    detail_lines = []
    failures = []

    start = time.time()
    done = 0
    for lang in sorted(per_lang):
        for path in per_lang[lang]:
            per_class_total[lang] += 1
            try:
                pred_code, raw_label = predict_one(
                    model, path, args.segment_duration, args.num_segments, args.seed
                )
            except Exception as e:
                failures.append((path, lang, str(e)))
                pred_code, raw_label = "ERROR", "ERROR"
            else:
                confusion[lang][pred_code] += 1
                if pred_code == lang:
                    per_class_correct[lang] += 1
            detail_lines.append(f"{lang}\t{pred_code}\t{raw_label}\t{path}")

            done += 1
            if args.log_every and done % args.log_every == 0:
                elapsed = time.time() - start
                print(f"  {done:,}/{total_files:,} files ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - start
    overall_total = sum(per_class_total.values())
    overall_correct = sum(per_class_correct.values())
    micro_acc = overall_correct / overall_total if overall_total else 0.0
    per_class_acc = {c: (per_class_correct[c] / per_class_total[c] if per_class_total[c] else None)
                      for c in all_langs}
    valid_accs = [a for a in per_class_acc.values() if a is not None]
    macro_acc = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("# AmberNet (NeMo) evaluation\n")
        if args.checkpoint:
            fh.write(f"# model: fine-tuned checkpoint {args.checkpoint}\n")
        else:
            fh.write(f"# model: pretrained {args.model_name}\n")
        fh.write(f"# test root: {args.test_root}\n")
        if model_labels is not None:
            fh.write(f"# model label set ({len(model_labels)}): {model_labels}\n")
        if yue_in_model:
            fh.write(f"# yue IS in this model's label set — this is a like-for-like\n")
            fh.write(f"#   comparison against evaluate_19class.py's fine-tuned checkpoint.\n")
        else:
            fh.write(f"# NOTE: this model's label set has no yue class (either the stock\n")
            fh.write(f"#   107-language VoxLingua107 pretrained AmberNet, or a checkpoint\n")
            fh.write(f"#   that wasn't fine-tuned to add it). It will show 0% yue recall by\n")
            fh.write(f"#   construction — compare against lid_test_results.txt (zero-shot\n")
            fh.write(f"#   ECAPA-TDNN), not against evaluate_19class.py's fine-tuned\n")
            fh.write(f"#   checkpoint, for yue specifically.\n")
        fh.write(f"# files evaluated: {overall_total:,}   elapsed: {elapsed:.1f}s\n")
        fh.write(f"# overall accuracy: micro={micro_acc*100:.2f}%  macro={macro_acc*100:.2f}%\n")
        if failures:
            fh.write(f"# failures (could not classify): {len(failures):,}\n")
        fh.write("#\n")

        fh.write("## Per-language accuracy\n")
        lang_w = max(len(l) for l in all_langs)
        for lang in all_langs:
            n = per_class_total[lang]
            c = per_class_correct[lang]
            acc = per_class_acc[lang]
            tag = " *watch" if lang in WATCH_CLASSES else ""
            acc_str = f"{acc*100:6.2f}%" if acc is not None else "   n/a"
            fh.write(f"{lang.ljust(lang_w)}  {c:>6,}/{n:<6,}  {acc_str}{tag}\n")
        fh.write("\n")

        fh.write("## Confusion summary (true_lang -> top predicted codes)\n")
        for lang in sorted(confusion):
            top = confusion[lang].most_common(5)
            top_str = ", ".join(f"{pc}={n}" for pc, n in top)
            fh.write(f"{lang.ljust(lang_w)}  {top_str}\n")
        fh.write("\n")

        watch_present = [c for c in WATCH_CLASSES if c in all_langs]
        if watch_present:
            fh.write(f"## Watch-class confusion block ({'/'.join(watch_present)})\n")
            header = "true".ljust(lang_w) + "".join(w.rjust(10) for w in watch_present) + "other".rjust(10)
            fh.write(header + "\n")
            for true_c in watch_present:
                row = true_c.ljust(lang_w)
                other = per_class_total[true_c]
                for pred_c in watch_present:
                    n = confusion[true_c].get(pred_c, 0)
                    other -= n
                    row += f"{n:,}".rjust(10)
                row += f"{other:,}".rjust(10)
                fh.write(row + "\n")
            fh.write("\n")

        if failures:
            fh.write("## Failures\n")
            for path, lang, err in failures:
                fh.write(f"{lang}\t{path}\t{err}\n")
            fh.write("\n")

        fh.write("## Per-file predictions (true_lang, predicted_code, raw_label, path)\n")
        for line in detail_lines:
            fh.write(line + "\n")

    print(f"\nOverall: micro={micro_acc*100:.2f}%  macro={macro_acc*100:.2f}%")
    print("Watch classes: " + "  ".join(
        f"{c}={per_class_acc[c]*100:.2f}%" if per_class_acc.get(c) is not None else f"{c}=n/a"
        for c in WATCH_CLASSES if c in all_langs
    ))
    if not yue_in_model:
        print(f"[note] this model's label set has no '{args.yue_code}' class — its yue "
              f"recall is 0% by construction. Compare against lid_test_results.txt "
              f"(zero-shot ECAPA-TDNN), not evaluate_19class.py's fine-tuned checkpoint.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
