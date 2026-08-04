"""Shared helpers for the 19-class (17 languages + yue) LID adaptation scripts.

Used by verify_setup.py, build_manifests_19class.py, train_19class.py,
evaluate_19class.py, diagnose_embeddings.py. Keeping this logic in one place
means the head-surgery index mapping, audio discovery, and BatchNorm-freeze
behavior are identical across every script that touches the model.
"""

import csv
import os

import torch
import torch.nn as nn

DEFAULT_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".sph", ".aac")
DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
SPLIT_NAMES = ("train", "test", "val")  # matches make_splits.py's SPLIT_NAMES


def is_audio(name, exts=DEFAULT_EXTS):
    if name.startswith("._"):  # macOS AppleDouble sidecar files
        return False
    return name.lower().endswith(exts)


def load_lang_map(path):
    """CSV with header 'folder,code'. Returns an ORDERED list of (folder, code)
    tuples, preserving file order top-to-bottom — this order defines classifier
    output indices 0..len-1, so it must stay stable across every script.
    """
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            folder = row["folder"].strip()
            code = row["code"].strip()
            if folder:
                rows.append((folder, code))
    return rows


def find_audio_by_lang(split_dir, langs, exts=DEFAULT_EXTS):
    """split_dir/<lang>/** -> {lang: [abs_path, ...]} for the requested langs only."""
    per_lang = {}
    for lang in langs:
        lang_dir = os.path.join(split_dir, lang)
        files = []
        if os.path.isdir(lang_dir):
            for dirpath, dirnames, filenames in os.walk(lang_dir):
                dirnames[:] = [d for d in dirnames if not d.startswith("._")]
                for f in sorted(filenames):
                    if is_audio(f, exts):
                        files.append(os.path.join(dirpath, f))
        per_lang[lang] = sorted(files)
    return per_lang


def split_label(text_lab):
    """'en: English' -> ('en', 'English'); falls back gracefully if no colon."""
    if ":" in text_lab:
        code, _, name = text_lab.partition(":")
        return code.strip(), name.strip()
    return text_lab.strip(), ""


def load_classifier(source, savedir, device):
    """Load the pretrained EncoderClassifier the same way evaluate_lid.py does."""
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # older speechbrain versions

    run_opts = {"device": device} if device else None
    return EncoderClassifier.from_hparams(
        source=source, savedir=savedir, run_opts=run_opts
    )


def build_code_to_oldidx(model):
    """{'en': 41, 'zh': 106, ...} from the pretrained model's label_encoder.

    The pretrained label_encoder's keys are the raw 'code: Full Name' labels
    (e.g. 'en: English'); this strips them down to the bare code so callers
    can look languages up by their lang_map.csv code.
    """
    label_encoder = model.hparams.label_encoder
    code_to_oldidx = {}
    for label, idx in label_encoder.lab2ind.items():
        code, _ = split_label(str(label))
        code_to_oldidx[code] = int(idx)
    return code_to_oldidx


def expand_classifier_head(model, lang_codes, yue_code, device, seed=1337):
    """Replace the 107-way classifier.out.w with an 18-way head.

    Indices 0..len(lang_codes)-1 are copied verbatim from the pretrained
    107-way rows for lang_codes, in that exact order (lang_map.csv order).
    The final index (yue) is freshly initialised, N(0, 0.01), bias 0, so the
    model starts as close to pretrained behaviour on the 17 known languages
    as possible while yue starts as a near-zero-signal new class.

    Returns (code_to_oldidx, idx_to_code) — idx_to_code is the new 18-entry
    ordering every downstream script must use consistently.
    """
    code_to_oldidx = build_code_to_oldidx(model)
    missing = [c for c in lang_codes if c not in code_to_oldidx]
    if missing:
        raise KeyError(
            f"lang_map.csv codes not found in the pretrained label_encoder: {missing}. "
            f"Run verify_setup.py first — this means the code/label_encoder assumption is wrong."
        )

    old_linear = model.mods.classifier.out.w
    if not isinstance(old_linear, nn.Linear):
        raise TypeError(
            f"expected classifier.out.w to be nn.Linear, got {type(old_linear)}. "
            f"The SpeechBrain classifier module layout has changed — re-run verify_setup.py."
        )

    in_features = old_linear.in_features
    n_new = len(lang_codes) + 1  # +1 for yue
    yue_idx = len(lang_codes)

    generator = torch.Generator().manual_seed(seed)
    new_linear = nn.Linear(in_features, n_new)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.bias.zero_()
        for new_idx, code in enumerate(lang_codes):
            old_idx = code_to_oldidx[code]
            new_linear.weight[new_idx].copy_(old_linear.weight[old_idx])
            new_linear.bias[new_idx].copy_(old_linear.bias[old_idx])
        new_linear.weight[yue_idx].copy_(
            torch.empty(in_features).normal_(mean=0.0, std=0.01, generator=generator)
        )
        # bias already zeroed above

    new_linear = new_linear.to(device)
    model.mods.classifier.out.w = new_linear

    idx_to_code = list(lang_codes) + [yue_code]
    return code_to_oldidx, idx_to_code


def set_bn_eval(module):
    """Recursively force every BatchNorm* submodule into eval mode.

    Must be called after every model.train() call that touches
    embedding_model, since .train() flips BN back into running-stats-update
    mode regardless of requires_grad. Running-stat drift on a frozen encoder
    is a silent forgetting channel — this is the guard against it.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def freeze_all(module):
    for p in module.parameters():
        p.requires_grad_(False)


def set_unfreeze_patterns(module, patterns):
    """Freeze every param in `module`, then re-enable grad for any named
    parameter whose name starts with one of `patterns`.

    Returns the count of parameters unfrozen, for a sanity-check print.
    """
    freeze_all(module)
    n_unfrozen = 0
    for name, param in module.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in patterns):
            param.requires_grad_(True)
            n_unfrozen += 1
    return n_unfrozen


def load_finetuned_model(checkpoint_path, source, savedir, device):
    """Load the base pretrained architecture, apply the same head surgery the
    checkpoint was trained with, then load the fine-tuned weights on top.

    Returns (model, idx_to_code). Used by evaluate_19class.py and
    diagnose_embeddings.py so both scripts reconstruct the exact same model
    train_19class.py produced.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    idx_to_code = ckpt["idx_to_code"]
    yue_code = ckpt["yue_code"]
    lang_codes = idx_to_code[:-1]
    assert idx_to_code[-1] == yue_code, (
        f"checkpoint's idx_to_code doesn't end with its own yue_code "
        f"({idx_to_code[-1]!r} != {yue_code!r}) — checkpoint may be corrupt or from "
        f"a different script version."
    )

    model = load_classifier(source, savedir, device)
    # Rebuild an 18-way head of the right shape, then overwrite with the
    # checkpoint's actual trained weights (expand_classifier_head's own
    # row-copy/init doesn't matter here since load_state_dict replaces it).
    expand_classifier_head(model, lang_codes, yue_code, device)
    model.mods.embedding_model.load_state_dict(ckpt["embedding_model_state"])
    model.mods.classifier.load_state_dict(ckpt["classifier_state"])
    model.mods.embedding_model.eval()
    model.mods.classifier.eval()
    return model, idx_to_code


def read_manifest_csv(path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["duration"] = float(row["duration"])
            rows.append(row)
    return rows


def write_manifest_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID", "duration", "wav", "label"])
        writer.writeheader()
        writer.writerows(rows)
