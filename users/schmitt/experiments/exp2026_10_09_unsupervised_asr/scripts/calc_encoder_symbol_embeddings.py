#!/usr/bin/env python3
"""
Average a trained shared encoder's states per input symbol -> one embedding per audio cluster id
and one per phoneme id, written as npz for `calc_embedding_isometry.py`.

This is the embedding side of the vecmap idea (see that script): the encoder has no subsampling,
so output position t corresponds to input position t, and averaging every state that sits on a
given symbol yields a "type" embedding for it -- the analogue of a word2vec vector.

It rebuilds the model *offline*, without loading the sisyphus graph: the training job's
`output/returnn.config` already holds the `get_model` functools.partial (network module + net_args)
that RETURNN itself used, so only that expression is evaluated (not the whole config), the class is
imported from `recipe/` and the checkpoint's `state_dict` loaded into it. No forward job, no GPU
(the venv's torch is CPU-only), a few minutes for a few thousand utterances.

Encoder inputs are UNMASKED -- the inference-time states, as in `dump_encoder_features` -- even
though training masks 30% of spans. NB a model with `codebook_opts` replaces a random
`codebook_prob` fraction of frames by their quantized code at eval too, so its averages are
non-deterministic; the variants used below (`baseline`, `baseline_lstm-gan`) have no codebook.

The audio/text utterance split matches `calc_embedding_isometry.py`: the audio side uses the first
`--num-audio-utts` equal-length cheat-seg utterances, the text side *disjoint* ones, so the two
embedding spaces are estimated from different data (as vecmap assumes).

Usage
-----
    ./calc_encoder_symbol_embeddings.py --train-job ReturnnTrainingJob.g2r39Yq1fvEG --epoch 100
    ./calc_embedding_isometry.py \\
        --audio-emb-npz <out-dir>/<job>.ep100.audio.npz \\
        --text-emb-npz <out-dir>/<job>.ep100.text.npz --ext-pca 32

Cheat-seg trainings of `config_librispeech_960_wo_sil_cheat_seg_v1` (found by grepping
`ls960_gmm_oracle_segment_clustering` in the jobs' `returnn.config`):
    g2r39Yq1fvEG  baseline (3L-512D, no codebook/discriminator)
    amUnjnqOjmRf  baseline_lstm-gan
    OrWoBo8IgwLk  baseline_codebook_*_code-prob-0.5      (non-deterministic, see above)
    E27gdwfK0Bl3  baseline_..._3L-256D
    YJfdIlMl6QTN  baseline_..._3L-128D
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, load_data

SETUP_DIR = "/u/schmitt/experiments/2026_04_09_unsupervised_asr"  # the sisyphus setup (work/, recipe/)
TRAINING_DIR = os.path.join(SETUP_DIR, "work", "i6_core", "returnn", "training")


def get_model_expr(text):
    """Source of the `get_model = ...` right-hand side, via ast (the file is never executed)."""
    import ast

    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "get_model" for t in node.targets
        ):
            return ast.get_source_segment(text, node.value)
    raise ValueError("no `get_model` assignment in the config")


def build_model(job_dir, epoch):
    """Rebuild the model from the job's returnn.config + checkpoint. Returns (model, net_args)."""
    cfg_path = os.path.join(job_dir, "output", "returnn.config")
    with open(cfg_path) as f:
        text = f.read()
    recipe = os.path.join(SETUP_DIR, "recipe")
    if recipe not in sys.path:
        sys.path.insert(0, recipe)
    expr = get_model_expr(text)
    factory = eval(expr)  # noqa: S307 -- our own generated config
    net_args = dict(factory.keywords)
    model = factory(epoch=epoch, step=0)
    ckpt = os.path.join(job_dir, "output", "models", "epoch.%03d.pt" % epoch)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    # strict=False: the checkpoints carry an unused `embedding.weight` the current Model no longer
    # builds. Nothing may be *missing*, though -- that would mean silently random weights.
    info = model.load_state_dict(sd, strict=False)
    assert not info.missing_keys, "checkpoint lacks %d params, e.g. %s" % (
        len(info.missing_keys),
        info.missing_keys[:3],
    )
    if info.unexpected_keys:
        print("  ignored %d unused checkpoint keys: %s" % (len(info.unexpected_keys), info.unexpected_keys))
    model.eval()
    return model, net_args


def symbol_means(model, seqs, vocab_size, modality, max_frames_per_batch, log_every=20):
    """Sum encoder states per input symbol over `seqs`, then average. Returns ([V, F], counts)."""
    forward = model.forward_audio if modality == "audio" else model.forward_text
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    total = None
    counts = np.zeros(vocab_size, dtype=np.int64)
    # length-sorted greedy batching on the *padded* frame count (rows x longest row), so a batch
    # of long sequences stays small -- budgeting on the shortest row OOMs on a 31 GB CPU node.
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[i]) > max_frames_per_batch:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            lens = torch.tensor([len(seqs[i]) for i in batch])
            x = torch.zeros((len(batch), int(lens.max())), dtype=torch.long)
            for r, i in enumerate(batch):
                x[r, : len(seqs[i])] = torch.from_numpy(seqs[i].astype(np.int64))
            states, _, out_lens, _ = forward(x, lens)
            states = states.numpy()
            if total is None:
                total = np.zeros((vocab_size, states.shape[-1]), dtype=np.float64)
            for r, i in enumerate(batch):
                L = int(out_lens[r])
                assert L == len(seqs[i]), "encoder is not frame-synchronous (%d vs %d)" % (L, len(seqs[i]))
                sym = seqs[i].astype(np.int64)
                np.add.at(total, sym, states[r, :L])
                np.add.at(counts, sym, 1)
            if (bi + 1) % log_every == 0 or bi + 1 == len(batches):
                print(
                    "    %s: batch %d/%d  %d symbols seen  (%.0fs)"
                    % (modality, bi + 1, len(batches), int((counts > 0).sum()), time.time() - t0)
                )
    emb = np.zeros_like(total)
    nz = counts > 0
    emb[nz] = total[nz] / counts[nz, None]
    return emb, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-job", required=True, help="ReturnnTrainingJob.<hash> (or a full path)")
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3)
    p.add_argument("--num-audio-utts", type=int, default=6000, help="must match calc_embedding_isometry.py")
    p.add_argument("--num-text-utts", type=int, default=6000)
    p.add_argument("--max-frames-per-batch", type=int, default=20000)
    p.add_argument(
        "--out-dir",
        default="/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/symbol_embeddings",
        help="NB not the home dir -- /u/schmitt has only a few GB free",
    )
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    args = p.parse_args()

    job_dir = args.train_job if os.path.isabs(args.train_job) else os.path.join(TRAINING_DIR, args.train_job)
    t0 = time.time()
    model, net_args = build_model(job_dir, args.epoch)
    print(
        "model: %s, %d enc layers, dim %d, codebook %s, discriminator %s"
        % (
            type(model).__name__,
            net_args.get("num_enc_layers"),
            net_args.get("model_dim"),
            net_args.get("codebook_opts"),
            net_args.get("discriminator_type"),
        )
    )

    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    text_tags = [t for t in common if t not in aset][: args.num_text_utts]
    print("  audio utts %d, disjoint text utts %d" % (len(audio_tags), len(text_tags)))

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "%s.ep%d" % (os.path.basename(job_dir), args.epoch))
    for modality, tags, data, vocab in (
        ("audio", audio_tags, clus, args.num_clusters),
        ("text", text_tags, phon, NUM_PHON),
    ):
        emb, counts = symbol_means(
            model, [data[t] for t in tags], vocab, modality, args.max_frames_per_batch
        )
        out = "%s.%s.npz" % (base, modality)
        np.savez(out, emb=emb.astype(np.float32), counts=counts)
        norms = np.linalg.norm(emb[counts > 0], axis=1)
        print(
            "  wrote %s  [%d, %d]  symbols %d/%d  norm %.2f-%.2f  (%.0fs)"
            % (out, emb.shape[0], emb.shape[1], (counts > 0).sum(), vocab, norms.min(), norms.max(), time.time() - t0)
        )


if __name__ == "__main__":
    main()
