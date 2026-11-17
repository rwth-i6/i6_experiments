"""Paired, deterministic additive-white-noise pilot; never select by model output."""

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sisyphus import Job, Task


def stable_seed(seed, seq_id):
    return int.from_bytes(hashlib.sha256(f"{seed}:{seq_id}".encode()).digest()[:8], "little")


def select_ids(ids, count, seed):
    """Round-robin through hashed chapters, then hashed utterances within chapter."""
    if len(set(ids)) != len(ids) or count < 1 or count > len(ids):
        raise ValueError("Invalid pilot size or duplicate source IDs")
    groups = defaultdict(list)
    for seq_id in ids:
        fields = seq_id.split("-")
        if len(fields) != 3 or not all(p.isdigit() for p in fields):
            raise ValueError(f"Expected a LibriSpeech utterance ID: {seq_id}")
        groups["-".join(fields[:2])].append(seq_id)
    order = sorted(groups, key=lambda k: (stable_seed(seed, k), k))
    groups = {k: sorted(v, key=lambda i: (stable_seed(seed, i), i)) for k, v in groups.items()}
    selected, depth = [], 0
    while len(selected) < count:
        for chapter in order:
            if depth < len(groups[chapter]):
                selected.append(groups[chapter][depth])
                if len(selected) == count:
                    break
        depth += 1
    return selected


def noise_versions(waveform, *, seed, snrs=(20.0, 10.0)):
    """Whole-segment RMS SNR, one shared unit-noise draw, common peak-safe gain.

    Gain is applied to clean AND noisy copies, never separately per condition.
    No VAD, clipping, lossy codec, reference text or recognition output is used.
    """
    clean = np.asarray(waveform, dtype=np.float64)
    if clean.ndim != 1 or not len(clean) or not np.isfinite(clean).all():
        raise ValueError("Expected finite nonempty mono audio")
    power = np.mean(clean ** 2)
    if power <= 0 or not snrs or not all(np.isfinite(s) for s in snrs):
        raise ValueError("Cannot set SNR for silent audio or invalid noise levels")
    unit = np.random.Generator(np.random.PCG64(seed)).standard_normal(len(clean))
    unit /= np.sqrt(np.mean(unit ** 2))
    versions = {"clean": clean}
    for snr in snrs:
        label = f"snr{snr:g}"
        if label in versions:
            raise ValueError("Duplicate SNR")
        versions[label] = clean + unit * np.sqrt(power) * 10 ** (-snr / 20)
    gain = min(1.0, 0.99 / max(float(np.max(np.abs(w))) for w in versions.values()))
    return {k: (w * gain).astype(np.float32) for k, w in versions.items()}, gain


class PrepareNoisePilotJob(Job):
    def __init__(self, dataset, references, *, count=256, seed=20261117, snrs=(20.0, 10.0), code_version):
        self.dataset, self.references = dataset, references
        self.count, self.seed, self.snrs, self.code_version = count, seed, snrs, code_version
        self.levels = ("clean",) + tuple(f"snr{s:g}" for s in snrs)
        self.out_datasets = {k: self.output_path(k + "/dataset", directory=True) for k in self.levels}
        self.out_references = self.output_path("references.jsonl")
        self.out_manifest = self.output_path("manifest.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=12, time=3))

    def run(self):
        import datasets
        import soundfile as sf
        from datasets import Audio, Dataset, Features, Value, load_from_disk

        original_dataset = load_from_disk(self.dataset.get_path())
        ds = original_dataset.cast_column("audio", Audio(decode=False))
        ids = list(map(str, ds["id"]))
        selected = select_ids(ids, self.count, self.seed)
        refs = [json.loads(line) for line in Path(self.references.get_path()).read_text().splitlines() if line.strip()]
        ref_map = {r["id"]: r for r in refs}
        if len(ref_map) != len(refs) or set(ids) != set(ref_map):
            raise ValueError("Dataset/reference coverage mismatch")
        for seq_id in selected:
            if ref_map[seq_id]["group_id"] != seq_id or ref_map[seq_id]["chunk_index"] != 0:
                raise ValueError("Noise pilot requires independent LibriSpeech segments")
        index = {seq_id: i for i, seq_id in enumerate(ids)}
        features = Features({"id": Value("string"), "text": Value("string"), "audio": Audio(sampling_rate=16000)})
        metadata, audio_seconds = [], 0.0
        with TemporaryDirectory(prefix="noise-pilot-") as scratch:
            rows = {k: [] for k in self.levels}
            for seq_id in selected:
                sample = ds[index[seq_id]]
                audio = sample["audio"]
                source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
                waveform, rate = sf.read(source, dtype="float32")
                if rate != 16000:
                    raise ValueError("Source must already be 16 kHz")
                local_seed = stable_seed(self.seed, seq_id)
                versions, gain = noise_versions(waveform, seed=local_seed, snrs=self.snrs)
                audio_seconds += len(waveform) / rate
                measured = {}
                for level, signal in versions.items():
                    filename = Path(scratch) / f"{seq_id}-{level}.wav"
                    sf.write(filename, signal, rate, subtype="FLOAT")
                    rows[level].append(dict(id=seq_id, text=sample["text"], audio=str(filename)))
                    if level != "clean":
                        measured[level] = float(10 * np.log10(np.mean(versions['clean'].astype('float64') ** 2) /
                            np.mean((signal.astype('float64') - versions['clean']) ** 2)))
                metadata.append(dict(id=seq_id, seed=local_seed, samples=len(waveform), common_gain=gain,
                                     source_audio_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(), measured_snr_db=measured))
            for level in self.levels:
                # save_to_disk embeds the WAV bytes; no dependency on scratch after this scope.
                Dataset.from_list(rows[level], features=features).save_to_disk(self.out_datasets[level].get_path())
        with open(self.out_references.get_path(), "w") as dst:
            for seq_id in selected:
                dst.write(json.dumps(ref_map[seq_id]) + "\n")
        manifest = dict(source_dataset=self.dataset.get_path(), source_fingerprint=original_dataset._fingerprint,
                        references_sha256=hashlib.sha256(Path(self.references.get_path()).read_bytes()).hexdigest(),
                        selection="hashed chapter round-robin, independent of text/hypotheses", seed=self.seed,
                        noise="PCG64 Gaussian white noise, same realization across SNRs", snrs_db=self.snrs,
                        snr_definition="whole-segment RMS, no VAD", encoding="float32 WAV, no clipping",
                        common_gain="min(1, .99 / peak across all paired conditions), applied to clean too",
                        count=len(selected), num_chapters=len({i.rsplit('-', 1)[0] for i in selected}),
                        audio_seconds=audio_seconds, numpy_version=np.__version__, soundfile_version=sf.__version__,
                        datasets_version=datasets.__version__, units=metadata)
        Path(self.out_manifest.get_path()).write_text(json.dumps(manifest, indent=2) + "\n")
