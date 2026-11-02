"""
EN+Blank (Uncollapsed 20ms Phoneme + Blank Frame Sequence) Data Preparation Jobs for Phoneme-Level UNMT.

Provides:
  - EN_BLANK_PHONEMES: Canonical 40-token vocabulary (0='BLK', 1..39=ARPAbet AA..ZH).
  - ExtractEnBlankFromPhonemeRefHdfJob: Extracts uncollapsed 20ms EN+Blank token sequences
    directly from zyang's LibriSpeech 960h GMM VAD-resampled HDF shards (phoneme_ref.{000..019}.hdf).
  - PrepareLibriSpeechDevOtherEnBlankEvalJob: Prepares paired (EN+Blank, EN ARPAbet phonemes)
    dev and test splits from LibriSpeech dev-other GMM alignments and reference transcripts.
  - ExtractCoVoST2AudioEnBlankJob: Extracts uncollapsed 20ms EN+Blank frame sequences from
    CommonVoice English audio clips for CoVoST-2 English-to-German Speech Translation dev/test splits
    using the 20ms wav2vec2 phoneme CTC model (Layer 9 + log-prior renormalization).
  - FilterAlignedParallelCorpusJob: Filters parallel (src, trg) text files so both sides are
    strictly non-empty and within optional token length bounds while preserving 1-to-1 alignment.
"""

from typing import Dict, Iterator, List, Optional, Tuple
import csv
import glob
import gzip
import json
import logging
import os
import re
import numpy as np

from sisyphus import Job, Task, tk
from .phonemize import ENGLISH_ARPABET_PHONEMES
from ..cluster_adapter import get_nltk_data_paths, setup_extra_python_paths


BLANK_TOKEN = "BLK"
EN_BLANK_PHONEMES: List[str] = [BLANK_TOKEN] + list(ENGLISH_ARPABET_PHONEMES)  # 40 tokens (0..39)


def _trim_boundary_zeros(arr: np.ndarray, keep_boundary_zeros: int = 3) -> np.ndarray:
    """
    Trims excessive leading and trailing silence/blank runs (value 0), keeping at most
    `keep_boundary_zeros` frames on each side to match rVAD speech-interval trimming.
    """
    if arr.size == 0:
        return arr
    non_zero = np.flatnonzero(arr != 0)
    if non_zero.size == 0:
        return arr[: min(arr.size, max(1, keep_boundary_zeros))]
    start = max(0, int(non_zero[0]) - keep_boundary_zeros)
    end = min(int(arr.size), int(non_zero[-1]) + 1 + keep_boundary_zeros)
    return arr[start:end]


def _resample_10ms_to_20ms(arr_10ms: np.ndarray) -> np.ndarray:
    """
    Resamples a 10ms frame-level label sequence (values 0..39) to 20ms (stride 2).
    In each 2-frame window [2j, 2j+2), if both match, take that label; if one is speech (>0)
    and one is silence (0), prefer the speech phoneme (matching speech-overlap majority rule).
    """
    n_10 = int(arr_10ms.size)
    if n_10 == 0:
        return arr_10ms
    n_20 = (n_10 + 1) // 2
    out = np.zeros(n_20, dtype=np.int32)
    for j in range(n_20):
        a = int(arr_10ms[2 * j])
        b = int(arr_10ms[2 * j + 1]) if (2 * j + 1) < n_10 else a
        if a == b:
            out[j] = a
        elif a != 0:
            out[j] = a
        else:
            out[j] = b
    return out


class ExtractEnBlankFromPhonemeRefHdfJob(Job):
    """
    Reads precomputed 20ms frame-level phoneme reference HDF shards (phoneme_ref.{000..019}.hdf)
    and writes uncollapsed space-separated EN+Blank sequences (0 -> 'BLK', 1..39 -> AA..ZH).
    Does NOT collapse repeated phones or repeated blank tokens when collapse=False (default).
    """
    _version = 1

    def __init__(
        self,
        hdf_files: List[tk.Path],
        collapse: bool = False,
        min_frames: int = 2,
        max_frames: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ):
        super().__init__()
        self.hdf_files = list(hdf_files)
        self.collapse = collapse
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_sentences = max_sentences

        self.out_en_blank_text = self.output_path("en_blank.tokens.txt")
        self.out_seq_tags = self.output_path("en_blank.seq_tags.txt")
        self.out_stats_json = self.output_path("en_blank.stats.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        import h5py

        logging.basicConfig(level=logging.INFO)
        id2sym = EN_BLANK_PHONEMES

        total_written = 0
        total_skipped_len = 0
        total_frames = 0
        total_blank_frames = 0
        lengths: List[int] = []

        with (
            open(self.out_en_blank_text.get_path(), "w", encoding="utf-8") as f_txt,
            open(self.out_seq_tags.get_path(), "w", encoding="utf-8") as f_tags,
        ):
            for shard_idx, hdf_path_obj in enumerate(self.hdf_files):
                hdf_path = hdf_path_obj.get_path()
                logging.info(f"Reading HDF shard [{shard_idx + 1}/{len(self.hdf_files)}]: {hdf_path}")
                with h5py.File(hdf_path, "r") as hf:
                    inputs = hf["inputs"][:]
                    seq_lens = hf["seqLengths"][:].reshape(-1)
                    seq_tags = hf["seqTags"][:]

                offset = 0
                for seq_len_raw, raw_tag in zip(seq_lens, seq_tags):
                    L = int(seq_len_raw)
                    seq_arr = inputs[offset : offset + L]
                    offset += L

                    if self.collapse:
                        collapsed_ids: List[int] = []
                        prev = -1
                        for v in seq_arr.tolist():
                            vi = int(v)
                            if vi != prev:
                                if vi != 0:
                                    collapsed_ids.append(vi)
                                prev = vi
                        seq_list = collapsed_ids
                    else:
                        seq_list = [int(v) if 0 <= int(v) < 40 else 0 for v in seq_arr.tolist()]

                    cur_len = len(seq_list)
                    if cur_len < self.min_frames:
                        total_skipped_len += 1
                        continue
                    if self.max_frames is not None and cur_len > self.max_frames:
                        total_skipped_len += 1
                        continue

                    tag_str = raw_tag.decode("utf-8", errors="replace") if isinstance(raw_tag, (bytes, np.bytes_)) else str(raw_tag)
                    f_txt.write(" ".join(id2sym[i] for i in seq_list) + "\n")
                    f_tags.write(tag_str + "\n")

                    total_written += 1
                    total_frames += cur_len
                    total_blank_frames += sum(1 for i in seq_list if i == 0)
                    lengths.append(cur_len)

                    if self.max_sentences is not None and total_written >= self.max_sentences:
                        break

                if self.max_sentences is not None and total_written >= self.max_sentences:
                    break

        lens_np = np.array(lengths, dtype=np.int64) if lengths else np.zeros(1, dtype=np.int64)
        stats = {
            "num_utterances": int(total_written),
            "num_skipped_length": int(total_skipped_len),
            "collapse": bool(self.collapse),
            "total_tokens": int(total_frames),
            "blank_tokens": int(total_blank_frames),
            "blank_percentage": round(float(total_blank_frames) / max(1, total_frames) * 100.0, 3),
            "mean_length": round(float(np.mean(lens_np)), 2),
            "median_length": round(float(np.median(lens_np)), 2),
            "p95_length": round(float(np.percentile(lens_np, 95)), 2),
            "max_length": int(np.max(lens_np)),
        }
        with open(self.out_stats_json.get_path(), "w", encoding="utf-8") as fs:
            json.dump(stats, fs, indent=2)

        logging.info(f"Extracted {total_written} EN+Blank utterances: {stats}")


class PrepareLibriSpeechDevOtherEnBlankEvalJob(Job):
    """
    Prepares aligned dev and test splits for EN+Blank <-> EN evaluation from LibriSpeech dev-other:
      - Reads 10ms EOW GMM alignments (lbs_mono_phone_eow_lexicon_dev_other/alignment_{1..10}.hdf),
        folds EOW states (c if c < 40 else c - 39), trims outer non-speech silence runs, resamples
        to 20ms (stride 2) matching phoneme_ref, and keeps all frames UNCOLLAPSED.
      - Reads reference transcripts from LibriSpeech/dev-other/*/*/*.trans.txt and phonemizes them
        to pure 39 ARPAbet phonemes using CMUDict + g2p_en.
      - Splits deterministically into `dev` (first `dev_split_size` utterances) and `test` (remainder).
    """
    __sis_version__ = 2

    def __init__(
        self,
        dev_other_align_hdfs: List[tk.Path],
        librispeech_corpus_dir: tk.Path,
        subset_name: str = "dev-other",
        dev_split_size: int = 500,
        max_frames: Optional[int] = 1500,
    ):
        super().__init__()
        self.dev_other_align_hdfs = list(dev_other_align_hdfs)
        self.librispeech_corpus_dir = librispeech_corpus_dir
        self.subset_name = subset_name
        self.dev_split_size = dev_split_size
        self.max_frames = max_frames

        self.out_dev_en_blank_txt = self.output_path("dev.en_blank.txt")
        self.out_dev_en_phn_txt = self.output_path("dev.en.phn.txt")
        self.out_dev_en_raw_txt = self.output_path("dev.en.raw.txt")
        self.out_dev_utt_ids_txt = self.output_path("dev.utt_ids.txt")

        self.out_test_en_blank_txt = self.output_path("test.en_blank.txt")
        self.out_test_en_phn_txt = self.output_path("test.en.phn.txt")
        self.out_test_en_raw_txt = self.output_path("test.en.raw.txt")
        self.out_test_utt_ids_txt = self.output_path("test.utt_ids.txt")
        self.out_stats_json = self.output_path("dev_other_eval_stats.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1})

    def run(self):
        import h5py

        logging.basicConfig(level=logging.INFO)
        setup_extra_python_paths()

        import nltk
        for p in get_nltk_data_paths():
            if os.path.exists(p) and p not in nltk.data.path:
                nltk.data.path.insert(0, p)

        from nltk.corpus import cmudict
        from g2p_en import G2p

        cmu = cmudict.dict()
        g2p = G2p()
        word_cache: Dict[str, List[str]] = {}

        def word_to_phones(word: str) -> List[str]:
            if word in word_cache:
                return word_cache[word]
            w_clean = re.sub(r"[^a-zA-Z0-9\']", "", word).lower()
            if not w_clean:
                word_cache[word] = []
                return []
            if w_clean in cmu:
                phones = [re.sub(r"\d+", "", p) for p in cmu[w_clean][0]]
            else:
                res = g2p(w_clean)
                phones = [re.sub(r"\d+", "", p) for p in res if p.strip() and p.isalnum()]
            word_cache[word] = phones
            return phones

        # 1. Load all LibriSpeech transcripts for subset_name (e.g. dev-other)
        subset_dir = os.path.join(self.librispeech_corpus_dir.get_path(), self.subset_name)
        utt2text: Dict[str, str] = {}
        for trans_file in sorted(glob.glob(os.path.join(subset_dir, "*", "*", "*.trans.txt"))):
            with open(trans_file, "r", encoding="utf-8") as ft:
                for line in ft:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt2text[parts[0]] = parts[1]

        logging.info(f"Loaded {len(utt2text)} reference transcripts from {subset_dir}")

        # 2. Extract and pair 20ms uncollapsed EN+Blank sequences with phonemized transcripts
        records: List[Tuple[str, str, str, str]] = []
        id2sym = EN_BLANK_PHONEMES

        for hdf_obj in self.dev_other_align_hdfs:
            with h5py.File(hdf_obj.get_path(), "r") as hf:
                inputs = hf["inputs"][:]
                seq_lens = hf["seqLengths"][:].reshape(-1)
                seq_tags = hf["seqTags"][:]

            offset = 0
            for seq_len_raw, raw_tag in zip(seq_lens, seq_tags):
                L = int(seq_len_raw)
                arr_10ms = inputs[offset : offset + L].astype(np.int32)
                offset += L

                tag_str = raw_tag.decode("utf-8", errors="replace") if isinstance(raw_tag, (bytes, np.bytes_)) else str(raw_tag)
                utt_id = tag_str.strip().split("/")[-1]
                if utt_id not in utt2text:
                    continue

                # Fold EOW (40..78 -> 1..39), trim boundary silence (keep <= 6 frames @ 10ms = 3 frames @ 20ms), resample to 20ms
                folded = np.where(arr_10ms >= 40, arr_10ms - 39, arr_10ms)
                folded = np.where((folded >= 0) & (folded < 40), folded, 0)
                trimmed_10ms = _trim_boundary_zeros(folded, keep_boundary_zeros=6)
                arr_20ms = _resample_10ms_to_20ms(trimmed_10ms)

                if arr_20ms.size < 2:
                    continue
                if self.max_frames is not None and arr_20ms.size > self.max_frames:
                    arr_20ms = arr_20ms[: self.max_frames]

                raw_text = utt2text[utt_id]
                phn_tokens: List[str] = []
                for w in raw_text.split():
                    phn_tokens.extend(word_to_phones(w))
                if not phn_tokens:
                    continue

                en_blank_str = " ".join(id2sym[int(v)] for v in arr_20ms.tolist())
                en_phn_str = " ".join(phn_tokens)
                records.append((utt_id, en_blank_str, en_phn_str, raw_text))

        records.sort(key=lambda x: x[0])
        assert len(records) > self.dev_split_size, (
            f"Expected > {self.dev_split_size} paired records in {self.subset_name}, got {len(records)}"
        )

        dev_recs = records[: self.dev_split_size]
        test_recs = records[self.dev_split_size :]

        def _write_split(recs, p_blank, p_phn, p_raw, p_ids):
            with (
                open(p_blank, "w", encoding="utf-8") as fb,
                open(p_phn, "w", encoding="utf-8") as fp,
                open(p_raw, "w", encoding="utf-8") as fr,
                open(p_ids, "w", encoding="utf-8") as fi,
            ):
                for u_id, b_s, p_s, r_s in recs:
                    fb.write(b_s + "\n")
                    fp.write(p_s + "\n")
                    fr.write(r_s + "\n")
                    fi.write(u_id + "\n")

        _write_split(
            dev_recs,
            self.out_dev_en_blank_txt.get_path(),
            self.out_dev_en_phn_txt.get_path(),
            self.out_dev_en_raw_txt.get_path(),
            self.out_dev_utt_ids_txt.get_path(),
        )
        _write_split(
            test_recs,
            self.out_test_en_blank_txt.get_path(),
            self.out_test_en_phn_txt.get_path(),
            self.out_test_en_raw_txt.get_path(),
            self.out_test_utt_ids_txt.get_path(),
        )

        stats = {
            "total_paired": len(records),
            "dev_utterances": len(dev_recs),
            "test_utterances": len(test_recs),
            "dev_mean_en_blank_len": round(float(np.mean([len(r[1].split()) for r in dev_recs])), 2),
            "dev_mean_en_phn_len": round(float(np.mean([len(r[2].split()) for r in dev_recs])), 2),
            "test_mean_en_blank_len": round(float(np.mean([len(r[1].split()) for r in test_recs])), 2),
            "test_mean_en_phn_len": round(float(np.mean([len(r[2].split()) for r in test_recs])), 2),
        }
        with open(self.out_stats_json.get_path(), "w", encoding="utf-8") as fs:
            json.dump(stats, fs, indent=2)
        logging.info(f"Prepared LibriSpeech {self.subset_name} EN+Blank <-> EN splits: {stats}")


class ExtractCoVoST2AudioEnBlankJob(Job):
    """
    Extracts uncollapsed 20ms EN+Blank frame sequences from CommonVoice English audio clips
    for CoVoST-2 English-to-German Speech Translation (`dev` or `test` split) using the 20ms
    wav2vec2 phoneme CTC checkpoint (Layer 9 + log-prior renormalization) and writes line-aligned
    raw German translation text for downstream German phonemization.
    """
    _version = 1

    def __init__(
        self,
        covost2_tsv: tk.Path,
        commonvoice_audio_dir: tk.Path,
        wav2vec2_ctc_ckpt: tk.Path,
        wav2vec2_ctc_prior: tk.Path,
        hf_cache_dir: str,
        split: str = "test",
        max_utterances: Optional[int] = 2000,
        prior_scale: float = 0.7,
        max_frames: int = 1200,
        use_gpu: bool = True,
    ):
        super().__init__()
        assert split in ("dev", "test", "train"), f"Unsupported CoVoST-2 split: {split}"
        self.covost2_tsv = covost2_tsv
        self.commonvoice_audio_dir = commonvoice_audio_dir
        self.wav2vec2_ctc_ckpt = wav2vec2_ctc_ckpt
        self.wav2vec2_ctc_prior = wav2vec2_ctc_prior
        self.hf_cache_dir = hf_cache_dir
        self.split = split
        self.max_utterances = max_utterances
        self.prior_scale = prior_scale
        self.max_frames = max_frames
        self.use_gpu = use_gpu

        self.out_en_blank_txt = self.output_path(f"covost2_{split}.en_blank.txt")
        self.out_de_raw_txt = self.output_path(f"covost2_{split}.de.raw.txt")
        self.out_de_tok_txt = self.output_path(f"covost2_{split}.de.tok.txt")
        self.out_clip_ids_txt = self.output_path(f"covost2_{split}.clip_ids.txt")
        self.out_stats_json = self.output_path(f"covost2_{split}.stats.json")

    def tasks(self) -> Iterator[Task]:
        rqmt = {"cpu": 4, "mem": 16, "time": 2}
        if self.use_gpu:
            rqmt["gpu"] = 1
        yield Task("run", rqmt=rqmt)

    def run(self):
        import soundfile as sf
        import torch
        import torch.nn as nn
        from .wmt14_de_en import simple_moses_tokenize

        logging.basicConfig(level=logging.INFO)
        setup_extra_python_paths()

        if os.path.exists(self.hf_cache_dir):
            os.environ["HF_HOME"] = self.hf_cache_dir
            os.environ["HF_HUB_CACHE"] = os.path.join(self.hf_cache_dir, "hub")
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.hf_cache_dir, "transformers")

        from transformers import Wav2Vec2Config, Wav2Vec2Model

        try:
            from sacremoses import MosesTokenizer
            de_tokenizer = MosesTokenizer(lang="de")
        except Exception:
            de_tokenizer = None

        device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")

        if os.path.exists(self.hf_cache_dir):
            hf_cfg = Wav2Vec2Config.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir=self.hf_cache_dir,
                local_files_only=True,
            )
        else:
            hf_cfg = Wav2Vec2Config()

        class _W2V2PhonemeCTC(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.wav2vec2 = Wav2Vec2Model(cfg)
                self.classifiers = nn.ModuleList([nn.Linear(768, 80) for _ in range(4)])

            def forward_layer9(self, wav_batch: torch.Tensor) -> torch.Tensor:
                hs = self.wav2vec2(
                    input_values=wav_batch,
                    return_dict=True,
                    output_hidden_states=True,
                ).hidden_states[9]
                return torch.log_softmax(self.classifiers[2](hs), dim=-1)

        model = _W2V2PhonemeCTC(hf_cfg)
        ckpt = torch.load(self.wav2vec2_ctc_ckpt.get_path(), map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)

        log_prior = torch.from_numpy(
            np.loadtxt(self.wav2vec2_ctc_prior.get_path(), dtype=np.float32)
        ).to(device)

        audio_dir = self.commonvoice_audio_dir.get_path()
        arpabet = ENGLISH_ARPABET_PHONEMES

        def _trim_waveform_silence(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
            win = int(0.02 * sr)  # 20ms energy window
            if wav.size <= 2 * win:
                return wav
            n_wins = wav.size // win
            frames = wav[: n_wins * win].reshape(n_wins, win)
            rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
            thresh = max(1e-4, 0.03 * float(np.max(rms)))
            active = np.flatnonzero(rms >= thresh)
            if active.size == 0:
                return wav
            s_sample = max(0, (int(active[0]) - 2) * win)
            e_sample = min(wav.size, (int(active[-1]) + 3) * win)
            return wav[s_sample:e_sample]

        written = 0
        missing_audio = 0
        lengths: List[int] = []
        blank_counts = 0
        total_frames = 0

        with (
            open(self.covost2_tsv.get_path(), "r", encoding="utf-8") as f_tsv,
            open(self.out_en_blank_txt.get_path(), "w", encoding="utf-8") as f_enb,
            open(self.out_de_raw_txt.get_path(), "w", encoding="utf-8") as f_der,
            open(self.out_de_tok_txt.get_path(), "w", encoding="utf-8") as f_det,
            open(self.out_clip_ids_txt.get_path(), "w", encoding="utf-8") as f_cid,
        ):
            reader = csv.DictReader(f_tsv, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row.get("split") != self.split:
                    continue
                de_text = (row.get("translation") or "").strip()
                rel_path = (row.get("path") or "").strip()
                if not de_text or not rel_path:
                    continue

                cand_wav = os.path.join(audio_dir, re.sub(r"\.mp3$", ".wav", rel_path))
                cand_orig = os.path.join(audio_dir, rel_path)
                audio_path = cand_wav if os.path.exists(cand_wav) else (cand_orig if os.path.exists(cand_orig) else None)
                if audio_path is None:
                    missing_audio += 1
                    continue

                try:
                    wav, sr = sf.read(audio_path, dtype="float32")
                except Exception as e:
                    logging.warning(f"Failed reading {audio_path}: {e}")
                    missing_audio += 1
                    continue

                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000 and wav.size > 0:
                    # Linear interpolation fallback if any clip is not 16 kHz
                    target_len = int(round(wav.size * 16000.0 / float(sr)))
                    wav = np.interp(
                        np.linspace(0.0, 1.0, target_len, endpoint=False),
                        np.linspace(0.0, 1.0, wav.size, endpoint=False),
                        wav,
                    ).astype(np.float32)

                wav = _trim_waveform_silence(wav, sr=16000)
                if wav.size < 640:
                    continue

                # Match training preprocessing: preemphasis(0.97) + peak_normalization
                wav_pe = np.empty_like(wav, dtype=np.float32)
                wav_pe[0] = wav[0]
                wav_pe[1:] = wav[1:] - 0.97 * wav[:-1]
                peak = float(np.max(np.abs(wav_pe)))
                if peak > 1e-8:
                    wav_pe = wav_pe / peak

                with torch.no_grad():
                    inp = torch.from_numpy(wav_pe).unsqueeze(0).to(device)
                    lp = model.forward_layer9(inp)[0] - self.prior_scale * log_prior
                    pred_ids = lp.argmax(dim=-1).detach().cpu().tolist()

                mapped_ids: List[int] = []
                for idx in pred_ids:
                    if idx in (39, 79):
                        mapped_ids.append(0)
                    elif 0 <= idx < 39:
                        mapped_ids.append(idx + 1)
                    elif 40 <= idx < 79:
                        mapped_ids.append(idx - 39)
                    else:
                        mapped_ids.append(0)

                trimmed_ids = _trim_boundary_zeros(np.array(mapped_ids, dtype=np.int32), keep_boundary_zeros=3)
                if trimmed_ids.size < 2 or int( np.sum(trimmed_ids != 0) ) == 0:
                    continue
                if self.max_frames is not None and trimmed_ids.size > self.max_frames:
                    trimmed_ids = trimmed_ids[: self.max_frames]

                de_tok = simple_moses_tokenize(de_text, lang="de", tokenizer=de_tokenizer)
                if not de_tok:
                    continue

                tok_list = [EN_BLANK_PHONEMES[int(v)] for v in trimmed_ids.tolist()]
                f_enb.write(" ".join(tok_list) + "\n")
                f_der.write(de_text + "\n")
                f_det.write(de_tok + "\n")
                f_cid.write(rel_path + "\n")

                written += 1
                cur_len = len(tok_list)
                lengths.append(cur_len)
                total_frames += cur_len
                blank_counts += sum(1 for v in trimmed_ids.tolist() if int(v) == 0)

                if written % 250 == 0:
                    logging.info(f"Extracted {written} CoVoST-2 ({self.split}) EN+Blank clips...")
                if self.max_utterances is not None and written >= self.max_utterances:
                    break

        lens_np = np.array(lengths, dtype=np.int64) if lengths else np.zeros(1, dtype=np.int64)
        stats = {
            "split": self.split,
            "num_utterances": int(written),
            "missing_audio": int(missing_audio),
            "prior_scale": float(self.prior_scale),
            "blank_percentage": round(float(blank_counts) / max(1, total_frames) * 100.0, 2),
            "mean_en_blank_len": round(float(np.mean(lens_np)), 2),
            "median_en_blank_len": round(float(np.median(lens_np)), 2),
        }
        with open(self.out_stats_json.get_path(), "w", encoding="utf-8") as fs:
            json.dump(stats, fs, indent=2)
        logging.info(f"Finished CoVoST-2 {self.split} EN+Blank extraction: {stats}")


class FilterAlignedParallelCorpusJob(Job):
    """
    Ensures two parallel line-by-line text files (and an optional raw reference file) contain
    strictly non-empty sequences on both sides so DumpTextToReturnnHdfJob and ComputeBleuJob
    preserve exact 1-to-1 sequence alignment.
    """
    __sis_version__ = 2

    def __init__(
        self,
        src_txt: tk.Path,
        trg_txt: tk.Path,
        raw_trg_txt: Optional[tk.Path] = None,
        clip_ids_txt: Optional[tk.Path] = None,
        min_src_len: int = 1,
        min_trg_len: int = 1,
        max_src_len: Optional[int] = None,
        max_trg_len: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ):
        super().__init__()
        self.src_txt = src_txt
        self.trg_txt = trg_txt
        self.raw_trg_txt = raw_trg_txt
        self.clip_ids_txt = clip_ids_txt
        self.min_src_len = min_src_len
        self.min_trg_len = min_trg_len
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.max_sentences = max_sentences

        self.out_src_txt = self.output_path("aligned.src.txt")
        self.out_trg_txt = self.output_path("aligned.trg.txt")
        self.out_raw_trg_txt = self.output_path("aligned.raw_trg.txt")
        self.out_clip_ids_txt = self.output_path("aligned.clip_ids.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.src_txt.get_path(), "r", encoding="utf-8") as fs:
            src_lines = [l.strip() for l in fs]
        with open(self.trg_txt.get_path(), "r", encoding="utf-8") as ft:
            trg_lines = [l.strip() for l in ft]

        if self.raw_trg_txt is not None:
            with open(self.raw_trg_txt.get_path(), "r", encoding="utf-8") as fr:
                raw_lines = [l.strip() for l in fr]
        else:
            raw_lines = list(trg_lines)

        clip_ids = None
        if self.clip_ids_txt is not None and os.path.exists(self.clip_ids_txt.get_path()):
            with open(self.clip_ids_txt.get_path(), "r", encoding="utf-8") as fc:
                clip_ids = [l.strip() for l in fc]

        assert len(src_lines) == len(trg_lines) == len(raw_lines), (
            f"Input parallel files must have equal line counts: {len(src_lines)}, {len(trg_lines)}, {len(raw_lines)}"
        )
        if clip_ids is not None:
            assert len(clip_ids) == len(src_lines), f"Clip IDs count mismatch: {len(clip_ids)} vs {len(src_lines)}"

        kept = 0
        with (
            open(self.out_src_txt.get_path(), "w", encoding="utf-8") as f_os,
            open(self.out_trg_txt.get_path(), "w", encoding="utf-8") as f_ot,
            open(self.out_raw_trg_txt.get_path(), "w", encoding="utf-8") as f_or,
            open(self.out_clip_ids_txt.get_path(), "w", encoding="utf-8") as f_oc,
        ):
            for idx, (s, t, r) in enumerate(zip(src_lines, trg_lines, raw_lines)):
                s_toks = s.split()
                t_toks = t.split()
                if len(s_toks) < self.min_src_len or len(t_toks) < self.min_trg_len:
                    continue
                if self.max_src_len is not None and len(s_toks) > self.max_src_len:
                    continue
                if self.max_trg_len is not None and len(t_toks) > self.max_trg_len:
                    continue
                f_os.write(" ".join(s_toks) + "\n")
                f_ot.write(" ".join(t_toks) + "\n")
                f_or.write(r + "\n")
                cid = clip_ids[idx] if clip_ids is not None else f"line-{idx}"
                f_oc.write(cid + "\n")
                kept += 1
                if self.max_sentences is not None and kept >= self.max_sentences:
                    break

        assert kept > 0, "FilterAlignedParallelCorpusJob produced 0 valid parallel pairs!"


class FilterNonEmptyMonolingualTextJob(Job):
    """
    Filters a monolingual token sequence file to retain non-empty sequences with length in
    [min_tokens, max_tokens], optionally capping at max_sentences.
    """
    _version = 1

    def __init__(
        self,
        text_file: tk.Path,
        min_tokens: int = 2,
        max_tokens: Optional[int] = 400,
        max_sentences: Optional[int] = None,
    ):
        super().__init__()
        self.text_file = text_file
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.out_text = self.output_path("filtered.tokens.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        in_path = self.text_file.get_path()
        open_fn = gzip.open if in_path.endswith(".gz") else open
        kept = 0
        with open_fn(in_path, "rt", encoding="utf-8", errors="replace") as f_in, open(
            self.out_text.get_path(), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                toks = line.strip().split()
                if len(toks) < self.min_tokens:
                    continue
                if self.max_tokens is not None and len(toks) > self.max_tokens:
                    continue
                f_out.write(" ".join(toks) + "\n")
                kept += 1
                if self.max_sentences is not None and kept >= self.max_sentences:
                    break
        logging.info(f"Filtered {kept} non-empty sequences to {self.out_text.get_path()}")
