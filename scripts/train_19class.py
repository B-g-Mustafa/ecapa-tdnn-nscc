#!/usr/bin/env python3
"""Phase 1.1 / 1.3 (and Phase 3 via --unfreeze) — train the 18-way head.

Model surgery: classifier.out.w is replaced 107-way -> 18-way (17 lang_map.csv
languages, in file order, + yue). Rows for the 17 languages are copied
verbatim from the pretrained weights; the yue row is freshly initialised.
The encoder is frozen by default (Phase 1). Pass --unfreeze with one or more
module-path prefixes (e.g. --unfreeze embedding_model.blocks.3) to unfreeze
specific encoder parts for Phase 3 — never pass blocks.0/1/2, per the plan.

BatchNorm running stats inside embedding_model are ALWAYS kept in eval mode,
regardless of --unfreeze — only a block's conv/affine weights become
trainable, never its running statistics. This must be re-applied every epoch
because model.train() cascades .train() to every child module, undoing it.

Per-epoch validation prints per-class recall, with zh/yue/vi/th called out
explicitly (the two primary and two secondary confusion risks, per the
measured zero-shot baseline in lid_test_results.txt). Checkpoints the best
epoch by validation loss, not by yue accuracy alone.

Usage:
    python scripts/train_19class.py \\
        --manifest-dir /home/users/ntu/birul001/scratch/data/common/manifests_19class \\
        --lang-map configs/lang_map.csv \\
        --output-dir ./runs/phase1_head_only \\
        --device cuda:0

    # Phase 3a example:
    python scripts/train_19class.py \\
        --manifest-dir ... --lang-map ... --output-dir ./runs/phase3a \\
        --init-checkpoint ./runs/phase1_head_only/best.pt \\
        --unfreeze embedding_model.blocks.3 \\
        --lr-encoder 1e-4 --epochs 8 --device cuda:0
"""

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lid_common import (
    DEFAULT_SOURCE,
    expand_classifier_head,
    load_classifier,
    load_lang_map,
    read_manifest_csv,
    set_bn_eval,
    set_unfreeze_patterns,
)

SAMPLE_RATE = 16000
WATCH_CLASSES = ("zh", "yue", "vi", "th")  # per the measured zero-shot confusion data


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class ManifestDataset(Dataset):
    def __init__(self, rows, code_to_idx, chunk_seconds=None, sample_rate=SAMPLE_RATE):
        self.rows = rows
        self.code_to_idx = code_to_idx
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torchaudio

        row = self.rows[i]
        wav, sr = torchaudio.load(row["wav"])
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)

        if self.chunk_seconds is not None:
            chunk_len = int(self.chunk_seconds * self.sample_rate)
            if wav.shape[0] > chunk_len:
                start = random.randint(0, wav.shape[0] - chunk_len)
                wav = wav[start:start + chunk_len]

        label_idx = self.code_to_idx[row["label"]]
        return wav, label_idx


def collate_fn(batch):
    wavs, labels = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in wavs])
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        padded[i, :w.shape[0]] = w
    rel_lengths = lengths.float() / max_len
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, rel_lengths, labels


def build_sampler(rows, code_to_idx):
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    weights = [1.0 / counts[row["label"]] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


# --------------------------------------------------------------------------
# Augmentation — prefer SpeechBrain's implementations, fall back to a minimal
# manual version if the import paths differ across speechbrain versions.
# --------------------------------------------------------------------------

def build_speed_perturb(device):
    try:
        from speechbrain.augment.time_domain import SpeedPerturb
        return SpeedPerturb(orig_freq=SAMPLE_RATE, speeds=[95, 100, 105]).to(device)
    except ImportError:
        pass
    try:
        from speechbrain.lobes.augment import TimeDomainSpecAugment  # older versions bundle speed here
        return TimeDomainSpecAugment(sample_rate=SAMPLE_RATE, speeds=[95, 100, 105]).to(device)
    except ImportError:
        pass

    print("[warn] SpeechBrain SpeedPerturb not found under either known import path; "
          "using a minimal manual resample-based fallback.", file=sys.stderr)

    def manual_speed_perturb(wavs, lengths):
        import torchaudio
        speed = random.choice([0.95, 1.0, 1.05])
        if speed == 1.0:
            return wavs, lengths
        new_sr = int(SAMPLE_RATE * speed)
        out = torchaudio.functional.resample(wavs, SAMPLE_RATE, new_sr)
        out = torchaudio.functional.resample(out, new_sr, SAMPLE_RATE)
        # length changes negligibly after resample-round-trip; keep original rel lengths
        min_len = min(out.shape[1], wavs.shape[1])
        return out[:, :min_len], lengths

    return manual_speed_perturb


def build_specaugment(device):
    try:
        from speechbrain.augment.freq_domain import SpecAugment
        return SpecAugment(time_warp=True, freq_mask=True, time_mask=True).to(device)
    except ImportError:
        pass
    try:
        from speechbrain.lobes.augment import SpecAugment
        return SpecAugment(time_warp=True, freq_mask=True, time_mask=True).to(device)
    except ImportError:
        pass

    print("[warn] SpeechBrain SpecAugment not found under either known import path; "
          "using a minimal manual freq/time masking fallback.", file=sys.stderr)

    def manual_specaugment(feats):
        # feats: (batch, time, n_mels)
        feats = feats.clone()
        b, t, f = feats.shape
        for i in range(b):
            f_width = random.randint(0, max(1, f // 5))
            f_start = random.randint(0, max(0, f - f_width))
            feats[i, :, f_start:f_start + f_width] = 0.0
            t_width = random.randint(0, max(1, t // 10))
            t_start = random.randint(0, max(0, t - t_width))
            feats[i, t_start:t_start + t_width, :] = 0.0
        return feats

    return manual_specaugment


def apply_speed_perturb(fn, wavs, rel_lengths):
    try:
        return fn(wavs, rel_lengths), rel_lengths
    except TypeError:
        out, lengths = fn(wavs, rel_lengths)
        return out, lengths


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def set_train_mode(model, has_unfrozen_encoder):
    model.mods.embedding_model.train(has_unfrozen_encoder)
    set_bn_eval(model.mods.embedding_model)  # must come AFTER .train(), it cascades and undoes this
    model.mods.classifier.train()


def set_eval_mode(model):
    model.mods.embedding_model.eval()
    model.mods.classifier.eval()


def encode(model, wavs, rel_lengths, has_unfrozen_encoder, specaugment_fn=None, training=False):
    """Replicates EncoderClassifier.encode_batch's compute_features ->
    mean_var_norm -> embedding_model pipeline manually (rather than calling
    encode_batch as a black box) so SpecAugment can be inserted on the
    frame-level fbank features, where it belongs — not on the pooled 256-d
    embedding, which has no time/frequency axes left to mask.
    """
    def _forward():
        feats = model.mods.compute_features(wavs)
        feats = model.mods.mean_var_norm(feats, rel_lengths)
        if training and specaugment_fn is not None:
            feats = specaugment_fn(feats)
        return model.mods.embedding_model(feats, rel_lengths)

    if has_unfrozen_encoder:
        return _forward()
    with torch.no_grad():
        return _forward()


def forward_batch(model, wavs, rel_lengths, has_unfrozen_encoder,
                   specaugment_fn=None, training=False):
    emb = encode(model, wavs, rel_lengths, has_unfrozen_encoder,
                 specaugment_fn=specaugment_fn, training=training)
    log_probs = model.mods.classifier(emb).squeeze(1)
    return log_probs


@torch.no_grad()
def evaluate(model, loader, idx_to_code, device, has_unfrozen_encoder):
    set_eval_mode(model)
    total_loss, n_batches = 0.0, 0
    correct = {c: 0 for c in idx_to_code}
    total = {c: 0 for c in idx_to_code}
    overall_correct, overall_total = 0, 0

    for wavs, rel_lengths, labels in loader:
        wavs, rel_lengths, labels = wavs.to(device), rel_lengths.to(device), labels.to(device)
        log_probs = forward_batch(model, wavs, rel_lengths, has_unfrozen_encoder)
        loss = F.nll_loss(log_probs, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = log_probs.argmax(dim=-1)
        for pred, label in zip(preds.tolist(), labels.tolist()):
            code = idx_to_code[label]
            total[code] += 1
            overall_total += 1
            if pred == label:
                correct[code] += 1
                overall_correct += 1

    per_class_recall = {c: (correct[c] / total[c] if total[c] else None) for c in idx_to_code}
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "overall_acc": overall_correct / max(overall_total, 1),
        "per_class_recall": per_class_recall,
        "correct": correct,
        "total": total,
    }


def print_epoch_report(epoch, train_loss, eval_result, idx_to_code):
    print(f"\n--- Epoch {epoch} ---")
    print(f"train_loss={train_loss:.4f}  val_loss={eval_result['val_loss']:.4f}  "
          f"val_acc={eval_result['overall_acc']:.4f}")
    watch = [c for c in WATCH_CLASSES if c in idx_to_code]
    if watch:
        parts = []
        for c in watch:
            r = eval_result["per_class_recall"][c]
            parts.append(f"{c}={r:.4f}" if r is not None else f"{c}=n/a")
        print("  watch: " + "  ".join(parts))
    for c in idx_to_code:
        r = eval_result["per_class_recall"][c]
        tag = " *" if c in WATCH_CLASSES else ""
        r_str = f"{r:.4f}" if r is not None else "n/a"
        print(f"  {c:6s} {eval_result['correct'][c]:>6}/{eval_result['total'][c]:<6} "
              f"{r_str}{tag}")


def lr_lambda_factory(total_steps, warmup_fraction):
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--manifest-dir", required=True, help="output dir of build_manifests_19class.py")
    p.add_argument("--lang-map", required=True)
    p.add_argument("--yue-code", default="yue")
    p.add_argument("--val-split", default="val", help="'val' or 'dev', whichever exists")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-checkpoint", default=None,
                   help="resume model weights from a previous train_19class.py checkpoint "
                        "(e.g. Phase 1's best.pt, before Phase 3 unfreezing)")
    p.add_argument("--unfreeze", nargs="+", default=[],
                   help="embedding_model.* module-path prefixes to unfreeze (Phase 3). "
                        "Never pass blocks.0/1/2.")
    p.add_argument("--savedir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-encoder", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--train-chunk-seconds", type=float, default=3.0,
                   help="random crop length for training utterances; 0 disables cropping")
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--speed-perturb", action="store_true", default=True)
    p.add_argument("--no-speed-perturb", dest="speed_perturb", action="store_false")
    p.add_argument("--specaugment", action="store_true", default=True)
    p.add_argument("--no-specaugment", dest="specaugment", action="store_false")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    forbidden = ("embedding_model.blocks.0", "embedding_model.blocks.1", "embedding_model.blocks.2")
    for pattern in args.unfreeze:
        if any(pattern == f or pattern.startswith(f) for f in forbidden):
            sys.exit(f"error: --unfreeze {pattern} touches a forbidden low-level block "
                     f"(blocks.0/1/2). Per the plan, these must never be unfrozen.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = read_manifest_csv(os.path.join(args.manifest_dir, "train.csv"))
    val_path = os.path.join(args.manifest_dir, f"{args.val_split}.csv")
    if not os.path.exists(val_path):
        alt = "dev" if args.val_split == "val" else "val"
        alt_path = os.path.join(args.manifest_dir, f"{alt}.csv")
        if os.path.exists(alt_path):
            print(f"[info] {val_path} not found, using {alt_path} instead")
            val_path = alt_path
        else:
            sys.exit(f"error: neither {val_path} nor {alt_path} found")
    val_rows = read_manifest_csv(val_path)

    lang_map = load_lang_map(args.lang_map)
    lang_codes = [code for _, code in lang_map]

    print(f"Loading model from {args.source} ...")
    savedir = args.savedir or "./pretrained_model_cache"
    model = load_classifier(args.source, savedir, args.device)
    device = torch.device(args.device)

    print("Expanding classifier head 107 -> 18 ...")
    code_to_oldidx, idx_to_code = expand_classifier_head(
        model, lang_codes, args.yue_code, device, seed=args.seed
    )
    code_to_idx = {c: i for i, c in enumerate(idx_to_code)}
    print(f"idx_to_code = {idx_to_code}")

    if args.init_checkpoint:
        print(f"Loading weights from {args.init_checkpoint} ...")
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.mods.embedding_model.load_state_dict(ckpt["embedding_model_state"])
        model.mods.classifier.load_state_dict(ckpt["classifier_state"])
        assert ckpt["idx_to_code"] == idx_to_code, (
            "checkpoint's idx_to_code ordering doesn't match this run's lang_map.csv — "
            "class indices would be silently scrambled if loaded."
        )

    n_unfrozen = set_unfreeze_patterns(model.mods.embedding_model, args.unfreeze)
    has_unfrozen_encoder = n_unfrozen > 0
    print(f"Encoder: {'FROZEN (Phase 1)' if not has_unfrozen_encoder else f'{n_unfrozen} params unfrozen (Phase 3): {args.unfreeze}'}")
    for p_ in model.mods.classifier.parameters():
        p_.requires_grad_(True)

    train_ds = ManifestDataset(train_rows, code_to_idx,
                                chunk_seconds=args.train_chunk_seconds or None)
    val_ds = ManifestDataset(val_rows, code_to_idx, chunk_seconds=None)

    sampler = build_sampler(train_rows, code_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    speed_perturb = build_speed_perturb(device) if args.speed_perturb else None
    specaugment = build_specaugment(device) if args.specaugment else None

    head_params = [p_ for p_ in model.mods.classifier.parameters() if p_.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if has_unfrozen_encoder:
        enc_params = [p_ for p_ in model.mods.embedding_model.parameters() if p_.requires_grad]
        param_groups.append({"params": enc_params, "lr": args.lr_encoder})
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_steps, args.warmup_fraction)
    )

    # Pre-training sanity check: for the 17 known languages, log-probs argmax
    # over those indices should already look sane (near the pretrained model's
    # own behaviour) since their rows were copied verbatim.
    initial_eval = evaluate(model, val_loader, idx_to_code, device, has_unfrozen_encoder)
    print("\n=== Pre-training baseline (this run's val split) ===")
    print_epoch_report(0, float("nan"), initial_eval, idx_to_code)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, has_unfrozen_encoder)
        epoch_start = time.time()
        running_loss, n_batches = 0.0, 0

        for wavs, rel_lengths, labels in train_loader:
            wavs, rel_lengths, labels = wavs.to(device), rel_lengths.to(device), labels.to(device)

            if speed_perturb is not None:
                wavs, rel_lengths = apply_speed_perturb(speed_perturb, wavs, rel_lengths)

            log_probs = forward_batch(model, wavs, rel_lengths, has_unfrozen_encoder,
                                       specaugment_fn=specaugment, training=True)
            loss = F.nll_loss(log_probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        eval_result = evaluate(model, val_loader, idx_to_code, device, has_unfrozen_encoder)
        print_epoch_report(epoch, train_loss, eval_result, idx_to_code)
        print(f"  epoch time: {time.time() - epoch_start:.1f}s")

        if eval_result["val_loss"] < best_val_loss:
            best_val_loss = eval_result["val_loss"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "embedding_model_state": model.mods.embedding_model.state_dict(),
                "classifier_state": model.mods.classifier.state_dict(),
                "idx_to_code": idx_to_code,
                "code_to_oldidx": code_to_oldidx,
                "lang_map_path": args.lang_map,
                "yue_code": args.yue_code,
                "args": vars(args),
                "val_loss": best_val_loss,
                "val_metrics": eval_result,
            }, best_path)
            print(f"  -> saved new best checkpoint to {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\nEarly stopping: no val_loss improvement for "
                      f"{args.early_stop_patience} epochs.")
                break

    print(f"\nDone. Best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")
    print("Next: scripts/evaluate_19class.py --checkpoint " + best_path)


if __name__ == "__main__":
    main()
