"""SAE §1d — word-level flashlight+kenlm decode of the CTC student (the no-Kaldi word-decode lever).

The §1d student is a phone-CTC model; a word-level readout needs a lexicon-constrained beam decode
against a word LM. This is fairseq's own `examples.speech_recognition.new.infer` with
`decoding.type=kenlm` (KenLMDecoder), plumbed exactly as the reference code expects — matched
against `examples/speech_recognition/new/decoders/flashlight_decoder.py` of the installed fairseq
0.12.2 (w2vu env site-packages):

  * imports (flashlight_decoder.py:29-42): `flashlight.lib.text.decoder` (KenLM, LexiconDecoder,
    Trie, ...) + `flashlight.lib.text.dictionary.{create_word_dict, load_words}` — provided by the
    `flashlight-text` wheel (0.0.7, aarch64), verified importable in the w2vu env.
  * lexicon format (line 61, `load_words(cfg.lexicon)`): plain text, one entry per line,
    whitespace-split, field 0 = word, remaining fields = the phone spelling; repeated word lines
    accumulate alternate pronunciations.
  * every spelling token must resolve in the model's dict.phn.txt: lines 73-76 do
    `tgt_dict.index(token)` and hard-assert `tgt_dict.unk() not in spelling_idxs` — hence
    BuildFlashlightLexiconJob strips stress digits and drops out-of-inventory entries below.
  * LM (line 65, `KenLM(cfg.lmpath, self.word_dict)`): kenlm loads the (gunzipped) ARPA directly.
  * decoder options (lines 80-92): `beam` / `lmweight` / `wordscore` map to our beam / lm_weight /
    word_score constructor args (decoder_config.py:44-72); blank = tgt_dict.bos() = <s> (index 0,
    same as training) and silence falls back to eos (base_decoder.py:19-29 — no <sep>/| in our dict).
  * infer.py writes per-utt words to `hypo.word` (infer.py:128 + 278-279/290: `hyp_words` =
    `" ".join(hypo["words"])`, line format `"<words> (None-<sid>)"`; the words come from
    `word_dict.get_entry` at flashlight_decoder.py:159-161) — the same (None-<idx>)/.uid join the
    phone decode uses.

fairseq's own "Word error rate:" line is NOT used here: with `task.labels=phn` the references it
scores against are the phone strings from `{split}.phn` (infer.py:283-301), while kenlm hypotheses
are words — that number is meaningless for this job. Word WER is computed in `collect` against the
LibriSpeech transcripts (LibriSpeechWordRefsJob), dev splits only: train stays label-free.
"""

from __future__ import annotations

import os
import re
import subprocess as sp
from typing import Dict, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.selftrain import (
    _build_decode_data_dir, _fairseq_dir, _parse_hypo_file, _read_lines, _torch_unsafe_load_env,
)
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

# openslr resources/11: the official LibriSpeech word 4-gram LM + the 200k-word pronunciation lexicon
_LS_LM_BASE_URL = "https://www.openslr.org/resources/11"


class FetchLibrispeechLmJob(Job):
    """Download the LibriSpeech 4-gram word ARPA (gunzipped) + the pronunciation lexicon.

    mini_task -> short/login engine, the only side with internet (settings.py). Sanity-checked here
    (ARPA header, size floor, lexicon line count) so a truncated download fails loudly instead of
    deep inside kenlm/flashlight.
    """

    def __init__(self, *, base_url: str = _LS_LM_BASE_URL):
        super().__init__()
        self.base_url = base_url
        self.out_arpa = self.output_path("4-gram.arpa")
        self.out_lexicon = self.output_path("librispeech-lexicon.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        gz = os.path.abspath("4-gram.arpa.gz")
        sp.check_call(["wget", "-q", "-O", gz, f"{self.base_url}/4-gram.arpa.gz"])
        with open(self.out_arpa.get_path(), "wb") as f:
            sp.check_call(["gunzip", "-c", gz], stdout=f)
        os.remove(gz)
        sp.check_call(
            ["wget", "-q", "-O", self.out_lexicon.get_path(), f"{self.base_url}/librispeech-lexicon.txt"])

        with open(self.out_arpa.get_path()) as f:
            head = [f.readline() for _ in range(5)]
        assert any(l.strip() == "\\data\\" for l in head), f"no ARPA \\data\\ header in {head!r}"
        arpa_bytes = os.path.getsize(self.out_arpa.get_path())
        assert arpa_bytes > 1_000_000_000, f"4-gram.arpa suspiciously small: {arpa_bytes} bytes"
        n_lex = sum(1 for l in open(self.out_lexicon.get_path()) if l.strip())
        assert n_lex > 100_000, f"lexicon suspiciously short: {n_lex} lines"
        print(f"fetched 4-gram.arpa ({arpa_bytes} bytes) + lexicon ({n_lex} lines)", flush=True)


class LibriSpeechWordRefsJob(Job):
    """Dev word transcripts {split: {uid: "WORD WORD ..."}} from the on-disk HF LS dataset.

    References for the word WER only. Routed dev-clean/dev-other by the gold id->split map (the same
    routing LibriStAudioJob used, so uid coverage matches the decode manifests exactly). train is
    deliberately NOT extracted: §1d treats it as unlabeled. Runs under the default (speech_llm) env,
    which carries `datasets`.
    """

    def __init__(self, *, hf_data_dir: tk.Path, gold: tk.Path, dev_split: str = "dev"):
        super().__init__()
        self.hf_data_dir = hf_data_dir
        self.gold = gold
        self.dev_split = dev_split

        self.out_refs = self.output_path("word_refs.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json

        from datasets import load_from_disk

        gold = json.load(open(self.gold.get_path()))
        id2split = {uid: s for s, d in gold.items() for uid in d}

        ds = load_from_disk(self.hf_data_dir.get_path())[self.dev_split]
        ds = ds.remove_columns([c for c in ds.column_names if c not in ("id", "text")])

        refs: Dict[str, Dict[str, str]] = {s: {} for s in gold}
        for ex in ds:
            uid = str(ex["id"])
            s = id2split.get(uid)
            if s is not None:
                refs[s][uid] = " ".join(str(ex["text"]).strip().split())
        for s, d in gold.items():
            missing = set(d) - set(refs[s])
            assert not missing, f"{s}: {len(missing)} gold uids missing a transcript, e.g. {sorted(missing)[:3]}"
            print(f"{s}: {len(refs[s])} word refs", flush=True)

        with open(self.out_refs.get_path(), "w") as f:
            json.dump(refs, f, indent=2)


def convert_lexicon_lines(
    lines: Sequence[str], inventory: set
) -> Tuple[List[Tuple[str, Tuple[str, ...]]], Dict[str, int]]:
    """CMUdict-style lexicon lines -> flashlight lexicon entries [(word, spelling)], + counters.

    Target format = what `flashlight.lib.text.dictionary.load_words` parses (used at
    flashlight_decoder.py:61): whitespace-split, word first, spelling after; one line per
    pronunciation. Stress digits are stripped (AH0 -> AH) to land in the 39-phone dict.phn.txt
    inventory; entries with any remaining out-of-inventory phone are dropped (flashlight_decoder.py:
    73-76 asserts every spelling token is in tgt_dict); duplicate (word, spelling) pairs — which
    stress-stripping creates — are removed.
    """
    entries: List[Tuple[str, Tuple[str, ...]]] = []
    seen = set()
    stats = {"lines": 0, "no_pron": 0, "oov_phone": 0, "duplicate": 0, "kept": 0}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        stats["lines"] += 1
        word, phones = fields[0], fields[1:]
        if not phones:
            stats["no_pron"] += 1
            continue
        spelling = tuple(re.sub(r"\d+$", "", p) for p in phones)
        if any(p not in inventory for p in spelling):
            stats["oov_phone"] += 1
            continue
        key = (word, spelling)
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        entries.append(key)
    stats["kept"] = len(entries)
    return entries, stats


class BuildFlashlightLexiconJob(Job):
    """librispeech-lexicon.txt -> flashlight/fairseq lexicon restricted to our dict.phn.txt phones."""

    def __init__(self, *, lexicon: tk.Path, dict_phn: tk.Path):
        super().__init__()
        self.lexicon = lexicon
        self.dict_phn = dict_phn

        self.out_lexicon = self.output_path("lexicon.phn.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        inventory = {l.split()[0] for l in _read_lines(self.dict_phn.get_path())}
        with open(self.lexicon.get_path()) as f:
            entries, stats = convert_lexicon_lines(f.readlines(), inventory)
        assert entries, "empty converted lexicon"
        assert stats["oov_phone"] < 0.05 * stats["lines"], f"dropped too many entries: {stats}"
        with open(self.out_lexicon.get_path(), "w") as f:
            for word, spelling in entries:
                print(f"{word}\t{' '.join(spelling)}", file=f)
        print(f"lexicon: {stats}", flush=True)


class Wav2Vec2KenlmDecodeJob(Job):
    """flashlight+kenlm lexicon beam decode of the §1d CTC student -> word hyps (+ dev word WER).

    One GPU decode task per split (the flashlight beam search itself runs on CPU over the emissions,
    so train's 28.5k utts get their own 11.5h window), then a cheap collect task joins `hypo.word`
    to the `.uid` files (permutation-asserted) and scores the dev splits against the LibriSpeech
    transcripts. Gold-less splits reuse the placeholder-label mechanics of the phone decode
    (`_build_decode_data_dir`) and yield hypotheses only.
    """

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        audio_dir: tk.Path,        # LibriStAudioJob.out_dir ({split}.tsv/.uid + audio/)
        checkpoint: tk.Path,       # Wav2Vec2CtcFinetuneJob.out_last
        gold: tk.Path,             # GoldPhonesJob json -> which splits have (phone) gold
        dict_phn: tk.Path,         # Wav2Vec2CtcFinetuneJob.out_dict (same index map as training)
        lexicon: tk.Path,          # BuildFlashlightLexiconJob.out_lexicon
        arpa: tk.Path,             # FetchLibrispeechLmJob.out_arpa (word 4-gram)
        word_refs: Optional[tk.Path] = None,  # LibriSpeechWordRefsJob.out_refs -> word-WER refs
        splits: Sequence[str] = ("dev-clean", "dev-other", "train"),
        beam: int = 500,
        lm_weight: float = 2.0,
        word_score: float = -1.0,
        max_tokens: int = 1100000,
        python_exe: tk.Path = W2VU_PYTHON,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.checkpoint = checkpoint
        self.gold = gold
        self.dict_phn = dict_phn
        self.lexicon = lexicon
        self.arpa = arpa
        self.word_refs = word_refs
        self.splits = list(splits)
        self.beam = beam
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.max_tokens = max_tokens
        self.python_exe = python_exe

        self.out_wer = self.output_path("word_wer.json")
        self.out_hyps = self.output_path("word_hyps.json")
        # mem: kenlm holds the multi-GB text ARPA in RAM inside the decode subprocess
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 64, "time": 11.5, "cpu": 8}

    def tasks(self):
        yield Task("decode", rqmt=self.rqmt, args=[[s] for s in self.splits])
        yield Task("collect", mini_task=True)

    def decode(self, split: str):
        assert_w2vu_env(self.python_exe)
        fs = _fairseq_dir(self.python_exe)
        # per-split data dir: decode tasks run concurrently in the same work dir
        data, _ = _build_decode_data_dir(
            audio_dir=self.audio_dir.get_path(), dict_phn=self.dict_phn.get_path(),
            gold_path=self.gold.get_path(), splits=[split], dest=os.path.abspath(f"data_{split}"))
        conf = os.path.join(fs, "examples", "speech_recognition", "new", "conf")
        resdir = os.path.abspath(f"decode_{split}")
        os.makedirs(resdir, exist_ok=True)
        args = [
            os.fspath(self.python_exe), "-m", "examples.speech_recognition.new.infer",
            f"--config-dir={conf}", "--config-name=infer",
            "task=audio_finetuning", f"task.data={data}", "task.labels=phn",
            "decoding.type=kenlm",
            f"decoding.lexicon={self.lexicon.get_path()}",
            f"decoding.lmpath={self.arpa.get_path()}",
            f"decoding.beam={self.beam}",
            f"decoding.lmweight={self.lm_weight}",
            f"decoding.wordscore={self.word_score}",
            "common_eval.post_process=none",
            f"common_eval.path={self.checkpoint.get_path()}",
            f"dataset.gen_subset={split}", f"dataset.max_tokens={self.max_tokens}",
            "dataset.skip_invalid_size_inputs_valid_test=false",  # dropped utts would break the uid join
            "distributed_training.distributed_world_size=1",
            f"common_eval.results_path={resdir}", f"decoding.results_path={resdir}",
            f"hydra.run.dir={resdir}",
        ]
        print("RUN:", " ".join(args), flush=True)
        out = sp.check_output(args, stderr=sp.STDOUT, text=True, env=_torch_unsafe_load_env())
        print(out, flush=True)
        # the infer-internal "Word error rate:" scores words against .phn refs -> ignored on purpose
        assert os.path.exists(os.path.join(resdir, "hypo.word")), f"no hypo.word for {split}"

    def collect(self):
        import json

        refs = json.load(open(self.word_refs.get_path())) if self.word_refs is not None else {}

        wers, hyps = {}, {}
        for s in self.splits:
            uids = _read_lines(os.path.join(self.audio_dir.get_path(), f"{s}.uid"))
            hyps[s] = _parse_hypo_file(os.path.abspath(os.path.join(f"decode_{s}", "hypo.word")), uids)
            n_empty = sum(1 for v in hyps[s].values() if not v)
            print(f"{s}: {len(hyps[s])} hyps, {n_empty} empty", flush=True)
            if s in refs:
                errs = total = 0
                for uid in uids:
                    # both sides are uppercase already (lexicon words / LS transcripts); upper() pins it
                    ref = refs[s][uid].upper().split()
                    errs += _levenshtein(hyps[s][uid].upper().split(), ref)
                    total += len(ref)
                wers[s] = errs / total
                print(f"{s}: word WER {wers[s]:.4f} ({errs}/{total})", flush=True)

        with open(self.out_wer.get_path(), "w") as f:
            json.dump(wers, f, indent=2)
        with open(self.out_hyps.get_path(), "w") as f:
            json.dump(hyps, f, indent=2)
        print("word WER:", json.dumps(wers, indent=2), flush=True)


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Plain two-row edit distance (the collect task must not depend on the w2vu env's editdistance)."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = curr
    return prev[-1]
