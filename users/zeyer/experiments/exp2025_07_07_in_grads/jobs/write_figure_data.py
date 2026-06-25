"""Dump the grad-align figure DATA (manifest + binary blobs) for one example utterance.

Sibling of :class:`PlotGradAlignJob`, but it renders nothing:
it does the SAME data assembly
(the DP-consumed emission = log-softmax-over-time per token of the energy-weighted grad,
the silence/blank row, the reproduced DP boundaries, the gold word boundaries,
the per-token char labels),
and writes them to ``fig.json`` plus ``blobs/*.npy`` for a local renderer.
This is the figure analog of the table data-job:
numbers + presentation-independent binary, rendered to PDF off the cluster.
"""

from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class WriteFigureDataJob(Job):
    """Assemble + dump the per-token grad emission, silence row, log-mel, and word boundaries for one seq.

    Same assembly contract as :class:`PlotGradAlignJob` (so the figure matches the reported alignment),
    plus the no-energy emission, the energy envelope, a log-mel panel, and the per-seq WBE.

    :param grad_score_hdf: from :class:`ExtractInGradsPerTokenJob`.
    :param align_opts / audio_energy_pow / blank_silence_energy_scale: must match the production
        :class:`WordAlignFromPerTokenGradsJob` so the dumped emission + boundaries equal the reported ones.
    :param seq_idx: which dataset seq to dump.
    :param model / family: labels carried into the manifest (for multi-model figures).
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
        seq_idx: int,
        audio_energy_pow: float = 0.0,
        blank_silence_energy_scale: float = 0.0,
        boundary_source: str = "word_detail",
        model: str = "",
        family: str = "",
        dataset: str = "",
        tokenizer_dir: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts
        self.seq_idx = int(seq_idx)
        self.audio_energy_pow = float(audio_energy_pow)
        self.blank_silence_energy_scale = float(blank_silence_energy_scale)
        self.boundary_source = boundary_source
        self.model = model
        self.family = family
        self.dataset = dataset
        self.tokenizer_dir = tokenizer_dir
        self.returnn_root = returnn_root
        self.out_dir = self.output_path("figdata", directory=True)
        self.out_manifest = self.output_path("figdata/fig.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _token_labels(self, words: List[str], tokens_per_word, n_tok: int) -> Optional[List[str]]:
        """Reconstruct per-token labels from the reference words (char-level layouts only)."""
        for sep in (None, "␣"):
            cand: List[str] = []
            for wi, w in enumerate(words):
                if sep is not None and wi > 0:
                    cand.append(sep)
                cand.extend(list(w))
            if len(cand) == n_tok:
                return cand
        return None

    def _logmel(self, audio, samplerate):
        """Log-mel spectrogram for the display panel. librosa if available, else a binned-STFT fallback.

        Returns ``(logmel [n_mels, n_frames], method)`` at ~100 frames/sec, 80 mel bands.
        """
        import numpy as np

        n_mels = 80
        hop = max(int(round(samplerate / 100.0)), 1)  # ~100 fps
        n_fft = 1 << (max(int(round(0.025 * samplerate)), 1) - 1).bit_length()  # ~25 ms, next pow2
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32), sr=samplerate, n_mels=n_mels, n_fft=n_fft, hop_length=hop
            )
            return librosa.power_to_db(mel, ref=np.max).astype(np.float32), "librosa"
        except Exception:  # fallback: framed |rFFT|, log, average into n_mels bands
            x = audio.astype(np.float64)
            n_frames = 1 + max(0, (len(x) - n_fft) // hop)
            win = np.hanning(n_fft)
            spec = np.empty((n_fft // 2 + 1, max(n_frames, 1)), dtype=np.float64)
            for i in range(n_frames):
                fr = x[i * hop : i * hop + n_fft]
                if len(fr) < n_fft:
                    fr = np.pad(fr, (0, n_fft - len(fr)))
                spec[:, i] = np.abs(np.fft.rfft(fr * win))
            logp = np.log(spec + 1e-9)
            edges = np.linspace(0, logp.shape[0], n_mels + 1).astype(int)
            mel = np.stack([logp[edges[b] : max(edges[b] + 1, edges[b + 1])].mean(0) for b in range(n_mels)])
            mel = mel - mel.max()
            return mel.astype(np.float32), "stft"

    def run(self):
        import os
        import sys
        import json
        import numpy as np

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import Aligner
        from datasets import load_dataset

        si = self.seq_idx
        hdf = HDFDataset([self.grad_score_hdf.get_path()])
        hdf.initialize()
        hdf.init_seq_order(epoch=1)
        hdf.load_seqs(0, si + 1)

        data = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key][si]
        samplerate = data["audio"]["sampling_rate"]
        audio = np.asarray(data["audio"]["array"], dtype=np.float64)
        audio_len_secs = len(audio) / samplerate

        n_tf = int(hdf.get_data(si, "num_input_frames")[0, 0])
        n_tok = int(hdf.get_data(si, "num_tokens")[0, 0])
        assert n_tok > 0, f"seq {si} has no tokens (empty recog) -- pick another seq_idx"
        tokens_per_word = hdf.get_data(si, "num_tokens_per_word").reshape(-1).astype(int)
        grad_mat = hdf.get_data(si, self.grad_score_key).reshape(n_tok, n_tf).astype(np.float64)
        secs_per_tf = audio_len_secs / n_tf

        if n_tok > n_tf:  # char tokens on a coarse grid -> upsample time, same as the align job
            up = -(-n_tok // n_tf)
            grad_mat = np.repeat(grad_mat, up, axis=1)
            n_tf *= up
            secs_per_tf = audio_len_secs / n_tf

        # Energy envelope on the grad frame grid (same as align).
        win = max(int(0.025 * samplerate), 1)
        hann = np.hanning(win)
        hann = hann / hann.sum()
        env = np.sqrt(np.convolve(audio * audio, hann, mode="same") + 1e-9)
        centers = ((np.arange(n_tf) + 0.5) / n_tf * (len(env) - 1)).astype(int)
        e = env[centers]
        e = e / (e.max() + 1e-9)

        # Blank/silence emission row from the ORIGINAL (pre-energy) grads, as WordAlign computes blank_override.
        _s0 = np.log(grad_mat + 1e-12)
        _s0 = _s0 - max(float(_s0.max()), 0.0)
        _m0 = _s0.max(axis=1, keepdims=True)
        _ls0 = _s0 - _m0 - np.log(np.sum(np.exp(_s0 - _m0), axis=1, keepdims=True))
        _mean_t, _std_t = _ls0.mean(0), _ls0.std(0)
        blank_score = float(self.align_opts.get("blank_score", -5))
        if self.blank_silence_energy_scale:
            _ze = (e - e.mean()) / (e.std() + 1e-9)
            blank_row = _mean_t - self.blank_silence_energy_scale * _ze * _std_t
        else:
            blank_row = np.full(n_tf, blank_score, dtype=np.float64)

        def _emission(energy_pow):
            g = grad_mat * (e[None, :] ** energy_pow) if energy_pow else grad_mat
            _e = np.log(g + 1e-12)
            _e = _e - _e.max(axis=1, keepdims=True)
            return np.exp(_e - np.log(np.sum(np.exp(_e), axis=1, keepdims=True) + 1e-12))

        emis = _emission(self.audio_energy_pow)  # [n_tok, n_tf], rows ~sum to 1
        emis_noen = _emission(0.0)

        # The boundaries the production DP produces (faithful: same Aligner, same blank_override).
        grad_for_align = grad_mat * (e[None, :] ** self.audio_energy_pow) if self.audio_energy_pow else grad_mat
        aligner = Aligner(**self.align_opts)
        tok_se = aligner.align(grad_for_align, blank_override=(blank_row if self.blank_silence_energy_scale else None))
        assert len(tok_se) == n_tok
        pred, cur = [], 0
        for k in tokens_per_word:
            pred.append([tok_se[cur][0] * secs_per_tf, tok_se[cur + k - 1][1] * secs_per_tf])
            cur += k

        words: List[str] = list(data[self.boundary_source]["utterance"])
        scale = self.dataset_offset_factors / samplerate
        gold = [
            [s * scale, en * scale]
            for s, en in zip(data[self.boundary_source]["start"], data[self.boundary_source]["stop"])
        ]
        assert len(words) == len(tokens_per_word) == len(gold), (len(words), len(tokens_per_word), len(gold))

        tok_labels = self._token_labels(words, tokens_per_word, n_tok)
        if tok_labels is None and self.tokenizer_dir is not None:
            from transformers import AutoTokenizer

            _tk = AutoTokenizer.from_pretrained(get_content_dir_from_hub_cache_dir(self.tokenizer_dir))
            _ids = _tk(" " + " ".join(words), add_special_tokens=False).input_ids
            if len(_ids) == n_tok:
                tok_labels = [(_tk.decode([i]).replace(" ", "␣") or "·") for i in _ids]
        if tok_labels is None:  # subword without tokenizer: fall back to per-token index placeholders
            tok_labels = [""] * n_tok

        per_word_wbe = [(abs(pred[i][0] - gold[i][0]) + abs(pred[i][1] - gold[i][1])) / 2.0 for i in range(len(words))]
        per_seq_wbe_ms = float(np.mean(per_word_wbe) * 1000.0)

        logmel, logmel_method = self._logmel(audio, samplerate)

        blobs_dir = os.path.join(self.out_dir.get_path(), "blobs")
        os.makedirs(blobs_dir, exist_ok=True)
        np.save(os.path.join(blobs_dir, "emission.npy"), emis.astype(np.float32))
        np.save(os.path.join(blobs_dir, "emission_noen.npy"), emis_noen.astype(np.float32))
        np.save(os.path.join(blobs_dir, "silence.npy"), blank_row.astype(np.float32))
        np.save(os.path.join(blobs_dir, "energy.npy"), e.astype(np.float32))
        np.save(os.path.join(blobs_dir, "logmel.npy"), logmel)

        manifest = {
            "name": f"{self.model}-{self.dataset}-seq{si}".strip("-"),
            "model": self.model,
            "family": self.family,
            "dataset": self.dataset,
            "seq_idx": si,
            "seq_tag": str(data.get("id", f"seq-{si}")),
            "audio_len_s": float(audio_len_secs),
            "n_tok": int(n_tok),
            "n_tf": int(n_tf),
            "secs_per_tf": float(secs_per_tf),
            "tok_labels": list(tok_labels),
            "tokens_per_word": [int(x) for x in tokens_per_word],
            "words": list(words),
            "gold": gold,
            "grad_align": pred,
            "energy_pow": float(self.audio_energy_pow),
            "sil_scale": float(self.blank_silence_energy_scale),
            "per_seq_wbe_ms": per_seq_wbe_ms,
            "logmel_method": logmel_method,
            "logmel_n_frames": int(logmel.shape[1]),
            "logmel_n_mels": int(logmel.shape[0]),
            "blobs": {
                "emission": "blobs/emission.npy",
                "emission_noenergy": "blobs/emission_noen.npy",
                "silence": "blobs/silence.npy",
                "energy": "blobs/energy.npy",
                "logmel": "blobs/logmel.npy",
            },
        }
        with open(self.out_manifest.get_path(), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        print(
            f"seq {si}: {n_tok} tok, {len(words)} words, {audio_len_secs:.2f}s, "
            f"wbe={per_seq_wbe_ms:.1f}ms, logmel={logmel_method} -> {self.out_manifest.get_path()}"
        )
