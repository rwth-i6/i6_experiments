"""Sealed supervised frame-phone diagnostics for the fixed phase 4B SAE."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle

from sisyphus import Job, Task, tk


def gold_files(split):
    base = os.path.join(os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home"),
                        "hub/datasets--gilkeyio--librispeech-alignments/snapshots")
    files = sorted(glob.glob(os.path.join(base, "*", "data", split.replace("-", "_") + "-*.parquet")))
    assert files, f"missing alignment parquet: {split}"
    return files


def read_gold(files, *, labels):
    import pandas as pd
    rows = {}
    for path in files:
        for row in pd.read_parquet(path, columns=["id", "phonemes"] if labels else ["id"]).itertuples():
            assert str(row.id) not in rows, f"duplicate alignment ID {row.id}"
            rows[str(row.id)] = row.phonemes if labels else None
    return rows


def frame_labels(phonemes, length):
    import numpy as np
    from i6_experiments.users.wu.experiments.ssl.analysis import repr_audit as RA
    gold = np.full(length, -1, dtype=np.int64)
    centers = (np.arange(length) + 0.5) / 50.0
    unknown = np.zeros(length, dtype=bool)
    covered = np.zeros(length, dtype=bool)
    # Match the source's speech-over-silence and last-speech-interval overlap policy.
    for speech_pass in (False, True):
        for interval in phonemes:
            label = str(interval["phoneme"])
            canonical = label.rstrip("0123456789")
            if canonical in RA.PHONE2ID:
                pid = RA.PHONE2ID[canonical]
            elif label.lower() in {"sil", "sp", "spn", "noise", "nsn", "lau", "<eps>"}:
                pid = RA.SIL_ID
            else:
                pid = -1
            if (pid >= 0 and pid != RA.SIL_ID) != speech_pass:
                continue
            lo, hi = np.searchsorted(centers, [float(interval["start"]), float(interval["end"])])
            gold[lo:hi] = pid
            covered[lo:hi] = True
            unknown[lo:hi] = pid < 0
    return gold, {"unknown_frames": int(unknown.sum()), "uncovered_frames": int((~covered).sum())}


def feature_map(counts):
    import numpy as np
    best = counts.argmax(axis=1)
    return np.where(2 * counts[np.arange(len(counts)), best] > counts.sum(axis=1), best, -1)


def mapped_predictions(codes, mapping, gold):
    import numpy as np
    rows, features = codes.nonzero()
    values = codes.data
    named = mapping[features] >= 0
    rows, features, values = rows[named], features[named], values[named]
    hit = np.zeros(len(gold), dtype=bool)
    np.logical_or.at(hit, rows, mapping[features] == gold[rows])
    pred = np.full(len(gold), -1, dtype=np.int64)
    # Stable sorting makes ties follow ascending feature index from CSR storage.
    order = np.argsort(-values, kind="stable")
    _, first = np.unique(rows[order], return_index=True)
    chosen = order[first]
    pred[rows[chosen]] = mapping[features[chosen]]
    return hit, pred


def fit_stats(features):
    import numpy as np
    from scipy import sparse
    if sparse.issparse(features):
        mu = np.asarray(features.mean(axis=0)).ravel()
        second = np.asarray(features.power(2).mean(axis=0)).ravel()
        sd = np.sqrt(np.maximum(second - mu * mu, 0))
    else:
        mu, sd = features.mean(axis=0), features.std(axis=0)
    return mu.astype(np.float32), sd.astype(np.float32) + 1e-5


def fit_probe(features, labels, num_phones, device):
    import torch
    from scipy import sparse
    mu, sd = fit_stats(features)
    torch.manual_seed(0)
    net = torch.nn.Linear(features.shape[1], num_phones).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    mt, st = torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)
    yt = torch.from_numpy(labels).to(device)
    for _ in range(25):
        perm = torch.randperm(len(labels))
        for start in range(0, len(labels), 4096):
            ids = perm[start:start + 4096].numpy()
            batch = features[ids]
            if sparse.issparse(batch):
                batch = batch.toarray()
            x = (torch.from_numpy(batch).to(device) - mt) / st
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(x), yt[torch.from_numpy(ids).to(device)]).backward()
            opt.step()
    return net.eval(), mu, sd


def paired_delta(rows, baseline, subset):
    import numpy as np
    speakers = sorted({r["speaker"] for r in rows})
    counts = np.array([[sum(r[subset]["correct"]["sae_probe"] - r[subset]["correct"][baseline]
                               for r in rows if r["speaker"] == speaker),
                        sum(r[subset]["frames"] for r in rows if r["speaker"] == speaker)]
                       for speaker in speakers], dtype=np.float64)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        diff, total = counts[rng.integers(len(counts), size=len(counts))].sum(axis=0)
        draws.append(diff / total)
    return {"delta": float(counts[:, 0].sum() / counts[:, 1].sum()),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "speakers": len(speakers)}


class BatchTopKProbeJob(Job):
    def __init__(self, checkpoint: tk.Path, av_states: tk.Path, artifact_root: str):
        super().__init__()
        self.checkpoint = checkpoint
        self.av_states = av_states
        self.artifact_root = artifact_root
        self.out_metrics = self.output_path("metrics.json")
        self.out_per_utterance = self.output_path("per_utterance.json")
        self.out_features = self.output_path("feature_phone_counts.npz")
        self.out_probes = self.output_path("probes.pt")
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "cpu": 4, "mem": 64, "time": 11.5}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np
        import torch
        from scipy import sparse
        from i6_experiments.users.wu.experiments.ssl.analysis import repr_audit as RA
        from .sparse_autoencoder import load_model, normalize_frames

        artifact_dir = os.path.join(self.artifact_root, self._sis_id())
        os.makedirs(artifact_dir, exist_ok=True)
        for output in (self.out_metrics, self.out_per_utterance, self.out_features, self.out_probes):
            path = os.path.join(artifact_dir, os.path.basename(output.get_path()))
            if not os.path.lexists(output.get_path()):
                os.symlink(path, output.get_path())
        device = "cuda"
        torch.set_num_threads(self.rqmt["cpu"])
        checkpoint = torch.load(self.checkpoint.get_path(), map_location="cpu", weights_only=False)
        assert checkpoint["completed_updates"] == 200000
        train_speakers = set(checkpoint["source_speaker_ids"])
        training_sources = {key: checkpoint[key] for key in
                            ("source_hdfs", "source_frames", "source_utterances", "source_speaker_ids")}
        weighted_l1 = checkpoint.get("sae_type") == "weighted_l1"
        if weighted_l1:
            from .sparse_autoencoder_l1 import load_model, normalize_frames

            sae_metadata = {key: checkpoint[key] for key in
                            ("sae_type", "model_config", "objective", "lambda_target", "lambda_warmup_updates",
                             "activation_normalization", "input_scale", "normalization_source",
                             "decoder_initial_column_norm", "decoder_initial_direction_distribution",
                             "encoder_initialization", "bias_initialization",
                             "decoder_normalization", "optimizer_config", "gradient_clip_global_norm",
                             "lr_decay_start_update", "max_updates", "lr_schedule", "batch_size", "sampling", "seed")}
        del checkpoint
        model = load_model(self.checkpoint.get_path(), device)
        model.eval().requires_grad_(False)
        with open(self.av_states.get_path(), "rb") as handle:
            states = pickle.load(handle)
        files = {split: gold_files(split) for split in ("dev-clean", "dev-other")}
        ids = {split: sorted(read_gold(paths, labels=False)) for split, paths in files.items()}
        assert not set(ids["dev-clean"]) & set(ids["dev-other"])
        speakers = {split: sorted({u.split("-")[0] for u in utts}) for split, utts in ids.items()}
        assert not set(speakers["dev-clean"]) & set(speakers["dev-other"])
        assert not train_speakers & (set(speakers["dev-clean"]) | set(speakers["dev-other"]))
        for split in ids:
            assert set(ids[split]) <= set(states), f"raw cache missing {split} utterances"
        selected = set(ids["dev-clean"]) | set(ids["dev-other"])
        excluded = sorted(set(states) - selected)
        excluded_speakers = {u.split("-")[0] for u in excluded}
        assert not excluded_speakers & (set(speakers["dev-clean"]) | set(speakers["dev-other"]))
        metrics = {"sources": {"checkpoint": os.path.realpath(self.checkpoint.get_path()),
                    "raw_l15": os.path.realpath(self.av_states.get_path()), "alignment_parquets": files,
                    "sae_training": training_sources, "completed_updates": 200000},
                   "split_speakers": speakers, "speaker_disjoint": True,
                   "arithmetic": {"dtype": "float32", "device": device,
                                  "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                  "torch_version": torch.__version__},
                   "excluded_cache_utterances": len(excluded),
                   "excluded_cache_frames": sum(len(states[u]) for u in excluded),
                   "protocol": {"fit": "ALL dev-clean aligned frames", "heldout": "ALL dev-other aligned frames",
                    "clock_hz": 50, "frame_center": "(t+0.5)/50", "phone_inventory": list(RA.PHONE2ID),
                    "rasterizer_reference": "i6_experiments/users/wu/experiments/ssl/analysis/repr_audit.py:frame_phone_labels",
                    "unknown_and_uncovered": "exclude (-1); empty and <unk> labels are unknown",
                    "sparse_rule": "positive top 50*T over each entire utterance separately",
                    "primary": "nonsilence frame accuracy", "secondary": "all aligned frames including SIL",
                    "probe": {"hidden": 0, "epochs": 25, "selection": "final", "seed": 0,
                              "optimizer": "Adam", "lr": 1e-3, "weight_decay": 1e-5,
                              "batch_size": 4096, "scaling": "fit-only mean/std + 1e-5"},
                    "bootstrap": {"samples": 2000, "seed": 0, "cluster": "speaker", "percentiles": [2.5, 97.5],
                                  "source": "speech_llm/sae/emc/eval_jobs.py:1353-1355",
                                  "scope": "conditional on this fixed trained pair; no seed-robustness claim"},
                    "coverage": "descriptive supervised frame-hit; not PER"}}
        if weighted_l1:
            decoder_norms = model.decoder.weight.norm(dim=0)
            metrics["sources"]["sae_metadata"] = sae_metadata
            metrics["protocol"].update({
                "sparse_rule": "ReLU; no TopK",
                "codes": "g_i = f_i * ||W_dec[:,i]||_2; raw f_i = ReLU(W_enc x + b_enc)",
                "normalization": "global SAE-train RMS scalar c*x; E_train ||c*x||^2 = 1024",
                "input_scale": model.input_scale,
                "raw_activity": "f_i > 0", "contribution_activity": "g_i > 0",
                "zero_decoder_columns": int((decoder_norms == 0).sum().item())})
            metrics["activity"] = {}
        with open(self.checkpoint.get_path(), "rb") as handle:
            metrics["sources"]["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        for u in excluded:
            del states[u]
        with torch.no_grad():
            n_features = model.encode(torch.zeros(1, 1024, device=device)).shape[1]
        reconstruction = {}

        def encode_split(split, raw_gold):
            feature_counts = np.zeros((n_features, RA.NUM_PHONES), dtype=np.int64)
            phone_counts = np.zeros(RA.NUM_PHONES, dtype=np.int64)
            utterances, dense, codes, labels = [], [], [], []
            report = {"source_utterances": len(ids[split]), "included_utterances": 0,
                      "excluded_utterances": 0, "raw_frames": 0, "aligned_frames": 0,
                      "unknown_frames": 0, "uncovered_frames": 0, "utterances": {}}
            squared_error, active, frames = 0.0, 0, 0
            alive = np.zeros(n_features, dtype=bool)
            if weighted_l1:
                input_squared_norm = 0.0
                raw_alive = np.zeros(n_features, dtype=bool)
                histograms = {kind: np.zeros(n_features + 1, dtype=np.int64)
                              for kind in ("raw_relu", "contribution")}
            for utt in ids[split]:
                values = states.pop(utt)
                y, excluded_frames = frame_labels(raw_gold[utt], len(values))
                report["raw_frames"] += len(values)
                for name, value in excluded_frames.items():
                    report[name] += value
                valid = y >= 0
                report["aligned_frames"] += int(valid.sum())
                report["included_utterances"] += int(valid.any())
                report["excluded_utterances"] += int(not valid.any())
                report["utterances"][utt] = {"speaker": utt.split("-")[0], "raw_frames": len(values),
                                              "aligned_frames": int(valid.sum()), **excluded_frames}
                with torch.no_grad():
                    x = torch.from_numpy(values.astype(np.float32)).to(device)
                    x = normalize_frames(x, model.input_scale) if weighted_l1 else normalize_frames(x)
                    z = model.encode(x)
                    squared_error += float((model.decode(z) - x).square().sum().item())
                    if weighted_l1:
                        input_squared_norm += float(x.square().sum().item())
                        raw_positive = z > 0
                        raw_alive |= raw_positive.any(dim=0).cpu().numpy()
                        histograms["raw_relu"] += np.bincount(
                            raw_positive.sum(dim=1).cpu().numpy(), minlength=n_features + 1)
                        z = z * decoder_norms
                        histograms["contribution"] += np.bincount(
                            (z > 0).sum(dim=1).cpu().numpy(), minlength=n_features + 1)
                    frames += len(x)
                    coo = z.to_sparse().coalesce()
                    ij = coo.indices().cpu().numpy()
                    vals = coo.values().cpu().numpy()
                    code = sparse.csr_matrix((vals, ij), shape=tuple(z.shape))
                    active += code.nnz
                    alive[code.indices] = True
                    x = x.cpu().numpy()
                aligned = code[valid]
                rr, ff = aligned.nonzero()
                np.add.at(feature_counts, (ff, y[valid][rr]), 1)
                phone_counts += np.bincount(y[valid], minlength=RA.NUM_PHONES)
                utterances.append((utt, y, code, x))
                if split == "dev-clean":
                    dense.append(x[valid])
                    codes.append(aligned)
                    labels.append(y[valid])
            report["excluded_frames"] = report["raw_frames"] - report["aligned_frames"]
            report["excluded_frame_fraction"] = report["excluded_frames"] / report["raw_frames"]
            metrics[split] = report
            reconstruction[split] = {"mse_per_dimension": squared_error / (frames * 1024),
                                      "mean_l0": active / frames, "dead_feature_rate": float((~alive).mean()),
                                      "frames": frames, "domain": "all raw unit-normalized frames"}
            if weighted_l1:
                activity = {}
                for kind, histogram in histograms.items():
                    cumulative = histogram.cumsum()
                    quantile_positions = np.array([0.5, 0.9]) * (frames - 1)
                    lower = np.searchsorted(cumulative, np.floor(quantile_positions), side="right")
                    upper = np.searchsorted(cumulative, np.ceil(quantile_positions), side="right")
                    quantiles = lower + (upper - lower) * (quantile_positions % 1)
                    occupied = np.flatnonzero(histogram)
                    activity[kind] = {"histogram": histogram.tolist(), "frames": frames,
                                      "mean_l0": float(np.dot(np.arange(n_features + 1), histogram) / frames),
                                      "min": int(occupied[0]), "median": float(quantiles[0]),
                                      "p90": float(quantiles[1]), "max": int(occupied[-1]),
                                      "zero_frame_fraction": float(histogram[0] / frames)}
                metrics["activity"][split] = activity
                reconstruction[split].update({
                    "domain": "all raw globally rescaled frames",
                    "normalized_mse": squared_error / input_squared_norm,
                    "mean_l0": activity["raw_relu"]["mean_l0"],
                    "dead_feature_rate": float((~raw_alive).mean()),
                    "activity_definition": "raw ReLU f_i > 0",
                    "contribution_mean_l0": activity["contribution"]["mean_l0"],
                    "contribution_dead_feature_rate": float((~alive).mean())})
            return utterances, dense, codes, labels, feature_counts, phone_counts

        fit = encode_split("dev-clean", read_gold(files["dev-clean"], labels=True))
        mapping = feature_map(fit[4])
        majority = int(fit[5].argmax())
        nonsil_counts = fit[5].copy()
        nonsil_counts[RA.SIL_ID] = 0
        majority_nonsil = int(nonsil_counts.argmax())
        print("[probe] fit dev-clean linear probes", flush=True)
        dense_probe = fit_probe(np.concatenate(fit[1]), np.concatenate(fit[3]), RA.NUM_PHONES, device)
        sae_probe = fit_probe(sparse.vstack(fit[2], format="csr"), np.concatenate(fit[3]), RA.NUM_PHONES, device)
        torch.save({"dense": {"state": dense_probe[0].state_dict(), "mu": dense_probe[1], "sd": dense_probe[2]},
                    "sae": {"state": sae_probe[0].state_dict(), "mu": sae_probe[1], "sd": sae_probe[2]},
                    "feature_map": mapping, "majority": majority, "majority_nonsil": majority_nonsil,
                    "protocol": metrics["protocol"], "checkpoint_sha256": metrics["sources"]["checkpoint_sha256"]},
                   self.out_probes.get_path())
        fit_counts = fit[4]
        del fit
        print("[probe] fixed probes saved; read dev-other gold", flush=True)
        test = encode_split("dev-other", read_gold(files["dev-other"], labels=True))
        rows = []
        for utt, y, code, x in test[0]:
            hit, named = mapped_predictions(code, mapping, y)
            predictions = {"any_named_hit": hit, "largest_named": named == y,
                           "majority": y == majority, "majority_nonsil": y == majority_nonsil}
            for arm, (net, mu, sd), inputs in (("dense_probe", dense_probe, x), ("sae_probe", sae_probe, code)):
                prediction = []
                with torch.no_grad():
                    for start in range(0, len(y), 4096):
                        batch = inputs[start:start + 4096]
                        if sparse.issparse(batch):
                            batch = batch.toarray()
                        prediction.append(net(torch.from_numpy((batch - mu) / sd).to(device)).argmax(1).cpu().numpy())
                predictions[arm] = np.concatenate(prediction) == y
            row = {"utterance": utt, "speaker": utt.split("-")[0], "raw_frames": len(y),
                   "excluded_frames": int((y < 0).sum())}
            for subset, mask in (("all", y >= 0), ("nonsil", (y >= 0) & (y != RA.SIL_ID))):
                row[subset] = {"frames": int(mask.sum()),
                               "correct": {arm: int(correct[mask].sum()) for arm, correct in predictions.items()},
                               "no_named_feature_frames": int(((named < 0) & mask).sum())}
            rows.append(row)
        summary = {}
        for subset in ("all", "nonsil"):
            total = sum(row[subset]["frames"] for row in rows)
            summary[subset] = {"frames": total, "accuracy": {
                arm: sum(row[subset]["correct"][arm] for row in rows) / total
                for arm in rows[0][subset]["correct"]}, "paired_sae_minus": {
                    arm: paired_delta(rows, arm, subset) for arm in ("dense_probe", "majority", "majority_nonsil")}}
        feature_totals = test[4].sum(axis=1)
        tp = np.zeros(n_features, dtype=np.int64)
        named = mapping >= 0
        tp[named] = test[4][np.flatnonzero(named), mapping[named]]
        recall_denom = np.zeros(n_features, dtype=np.int64)
        recall_denom[named] = test[5][mapping[named]]
        precision = np.divide(tp, feature_totals, out=np.full(n_features, np.nan), where=named & (feature_totals > 0))
        recall = np.divide(tp, recall_denom, out=np.full(n_features, np.nan), where=recall_denom > 0)
        np.savez_compressed(self.out_features.get_path(), mapping=mapping, fit_counts=fit_counts,
                            heldout_counts=test[4], heldout_phone_counts=test[5],
                            heldout_precision=precision, heldout_recall=recall, phone_names=np.array(list(RA.PHONE2ID)))
        metrics.update({"summary": summary, "reconstruction": reconstruction,
                        "named_features": int(named.sum()), "fit_majority_phone": RA.ARPABET_39[majority] if majority != RA.SIL_ID else RA.SIL,
                        "fit_nonsil_majority_phone": RA.ARPABET_39[majority_nonsil]})
        with open(self.out_per_utterance.get_path(), "w") as handle:
            json.dump(rows, handle, indent=2)
        with open(self.out_metrics.get_path(), "w") as handle:
            json.dump(metrics, handle, indent=2)
