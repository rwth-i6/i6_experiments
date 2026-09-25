"""New in the port: the decodes of the section 1d CTC student (``training.w2vu2_ctc``), run by fairseq's
own inference CLI (``examples.speech_recognition.new.infer``) as an external tool.

Ported from the live ``w2vu2/selftrain.py`` (``Wav2Vec2CtcDecodeJob``, ``_build_decode_data_dir``,
``_parse_hypo_file``) and ``w2vu2/word_decode.py`` (``convert_lexicon_lines``,
``BuildFlashlightLexiconJob``, ``Wav2Vec2KenlmDecodeJob``, ``_shard_manifest``) of the reference
setup.  The banked production decodes are ``Wav2Vec2CtcDecodeJob`` (viterbi phone PER) and
``Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`` (lexicon + KenLM word decode, dev-clean / dev-other WER
0.1796 / 0.2187).

Why fairseq's CLI and not a RETURNN forward: the decoder production ran is fairseq's
``KenLMDecoder`` (``decoders/flashlight_decoder.py``) over the raw ``wav2vec_ctc`` logits; running the
same CLI on the same checkpoint reproduces it without a fairseq -> HF weight conversion, a second
implementation of the decoder, or flashlight-text in the RETURNN env.

* :class:`FlashlightLexiconJob` -- ``librispeech-lexicon.txt`` (``lm.word_lm.official_lexicon``, the
  same sha256 as production's download) to the flashlight lexicon over the student's
  ``dict.phn.txt`` phones: stress digits stripped, entries with an out-of-inventory phone dropped,
  duplicate (word, spelling) pairs removed; ``"<word>\\t<phones>"`` per line.
* :class:`CtcPhoneDecodeJob` -- ``decoding.type=viterbi`` (argmax, collapse, drop blank) per split;
  PER = sum of Levenshtein distances / sum of reference lengths against the dev gold, computed here
  (the formula of fairseq's ``Word error rate`` line production read, without its 4-digit rounding).
* :class:`CtcWordDecodeJob` -- ``decoding.type=kenlm`` with production's beam 500, LM weight 2.0,
  word score -1.0; word WER = sum of Levenshtein distances over ``upper().split()`` tokens / sum of
  reference words, production's ``collect`` convention (i6_core's ``ScliteJob`` is not used: sclite
  aligns with its own weighted costs and its input would need a CTM/STM round trip).
* :class:`OggZipWordRefsJob` -- the dev word references ``{split: {utt_id: "WORD ..."}}`` from the
  ``text`` of the LibriSpeech ogg zips (the bliss orthography), whitespace-normalised as production's
  ``LibriSpeechWordRefsJob`` normalised the HF ``text``.

The manifests are :class:`..training.w2vu2_ctc.FairseqAudioManifestJob` outputs (``<split>.tsv`` +
``<split>.uid``).  Every decode writes ``<split>.phn`` with one placeholder phone per utterance: the
task needs a label file to load, the hypotheses do not depend on it, and fairseq's own ``Word error
rate`` line (scored against it) is not read.  Hypotheses are joined to utterance ids through the
``(None-<row>)`` suffix of fairseq's hypothesis files, asserted to be a permutation of the manifest rows.

The child process gets ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` (fairseq 0.12.2 calls ``torch.load``
without ``weights_only``; torch >= 2.6 then refuses the checkpoints' pickled ``argparse.Namespace``)
and a work-dir shim exposing fairseq's ``examples`` as a top-level package.  The jobs carry the
class attribute ``requires_env = "w2vu"`` (not hashed) like production's, for a ``settings.py`` that
routes the fairseq env.
"""

from __future__ import annotations

import json
import os
import re
import subprocess as sp
from typing import Dict, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

from .per import edit_counts

__all__ = [
    "convert_lexicon_lines",
    "FlashlightLexiconJob",
    "OggZipWordRefsJob",
    "CtcPhoneDecodeJob",
    "CtcWordDecodeJob",
    "WORD_DECODE_BEAM",
    "WORD_DECODE_LM_WEIGHT",
    "WORD_DECODE_WORD_SCORE",
    "DECODE_MAX_TOKENS",
]

#: production ``Wav2Vec2KenlmDecodeJob`` settings (AQw3EcUo6rks)
WORD_DECODE_BEAM = 500
WORD_DECODE_LM_WEIGHT = 2.0
WORD_DECODE_WORD_SCORE = -1.0
#: production ``dataset.max_tokens`` of both decodes
DECODE_MAX_TOKENS = 1100000


def _read_lines(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _edit_distance(hyp: Sequence[str], ref: Sequence[str]) -> int:
    return sum(edit_counts(list(hyp), list(ref)))


# -------------------------------------------------------------------------------------------------
# lexicon and references
# -------------------------------------------------------------------------------------------------
def convert_lexicon_lines(
    lines: Sequence[str], inventory: set
) -> Tuple[List[Tuple[str, Tuple[str, ...]]], Dict[str, int]]:
    """CMUdict-style lexicon lines -> flashlight lexicon entries [(word, spelling)] + counters
    (production ``word_decode.convert_lexicon_lines``, unchanged)."""
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


class FlashlightLexiconJob(Job):
    """``librispeech-lexicon.txt`` -> flashlight lexicon restricted to the student's ``dict.phn.txt``."""

    def __init__(self, *, lexicon: tk.Path, dict_phn: tk.Path):
        super().__init__()
        self.lexicon = lexicon
        self.dict_phn = dict_phn

        self.out_lexicon = self.output_path("lexicon.phn.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        inventory = {line.split()[0] for line in _read_lines(self.dict_phn.get_path())}
        with open(self.lexicon.get_path()) as f:
            entries, stats = convert_lexicon_lines(f.readlines(), inventory)
        assert entries, "empty converted lexicon"
        assert stats["oov_phone"] < 0.05 * stats["lines"], f"dropped too many entries: {stats}"
        with open(self.out_lexicon.get_path(), "w") as f:
            for word, spelling in entries:
                print(f"{word}\t{' '.join(spelling)}", file=f)
        print(f"lexicon: {stats}", flush=True)


class OggZipWordRefsJob(Job):
    """Dev word references ``{split: {utt_id: "WORD WORD ..."}}`` from LibriSpeech ogg zips."""

    def __init__(self, *, ogg_zips: Dict[str, tk.Path]):
        """:param ogg_zips: split name -> ogg zip (``data.librispeech.get_ogg_zip``)"""
        super().__init__()
        self.ogg_zips = ogg_zips

        self.out_refs = self.output_path("word_refs.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from ..data.ogg_zip import read_ogg_zip_index, utt_id_of_segment

        refs: Dict[str, Dict[str, str]] = {}
        for split, zip_path in sorted(self.ogg_zips.items()):
            d = refs[split] = {}
            for entry in read_ogg_zip_index(zip_path.get_path()):
                uid = utt_id_of_segment(entry["seq_name"])
                assert uid not in d, f"{split}: duplicate utterance {uid}"
                d[uid] = " ".join(str(entry["text"]).strip().split())
            print(f"{split}: {len(d)} word refs", flush=True)
        with open(self.out_refs.get_path(), "w") as f:
            json.dump(refs, f, indent=2)


# -------------------------------------------------------------------------------------------------
# fairseq inference plumbing
# -------------------------------------------------------------------------------------------------
_HYPO_LINE_RE = re.compile(r"^(.*)\(None-(\d+)\)\s*$")


def parse_hypo_file(path: str, uids: List[str]) -> Dict[str, str]:
    """infer's ``hypo.units`` / ``hypo.word`` -> {utt_id: hypothesis}, joined by the ``(None-<row>)``
    manifest row index, which must be a permutation of ``range(len(uids))``."""
    entries: Dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = _HYPO_LINE_RE.match(line)
            assert m, f"unparseable hypo line in {path}: {line!r}"
            idx = int(m.group(2))
            assert idx not in entries, f"duplicate hypo index {idx} in {path}"
            entries[idx] = m.group(1).strip()
    missing = set(range(len(uids))) - set(entries)
    extra = set(entries) - set(range(len(uids)))
    assert not missing and not extra, (
        f"{path}: {len(entries)} hyps vs {len(uids)} uids; "
        f"missing (first 5) {sorted(missing)[:5]}, out of range (first 5) {sorted(extra)[:5]}"
    )
    return {uids[i]: entries[i] for i in range(len(uids))}


def _examples_dir(fairseq_root: str) -> str:
    """fairseq's ``examples`` package: ``<root>/examples`` (a source checkout) or
    ``<root>/fairseq/examples`` (the pip wheel's site-packages layout)."""
    for cand in (os.path.join(fairseq_root, "examples"), os.path.join(fairseq_root, "fairseq", "examples")):
        if os.path.isfile(os.path.join(cand, "speech_recognition", "new", "infer.py")):
            return cand
    raise FileNotFoundError(f"no examples/speech_recognition/new/infer.py under {fairseq_root}")


def _infer_env(shim_dir: str, examples: str) -> Dict[str, str]:
    """The child env: ``examples`` importable through ``shim_dir``, fairseq checkpoints loadable."""
    os.makedirs(shim_dir, exist_ok=True)
    link = os.path.join(shim_dir, "examples")
    if not os.path.lexists(link):
        os.symlink(examples, link)
    assert os.path.realpath(link) == os.path.realpath(examples), (link, examples)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [shim_dir, env.get("PYTHONPATH", "")] if p)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def _build_decode_data_dir(*, manifest_dir: str, split: str, dict_phn: str, dest: str,
                           shard: int = 0, num_shards: int = 1) -> List[str]:
    """``dest`` = ``dict.phn.txt`` + ``<split>.tsv`` (rows ``[shard::num_shards]``) + placeholder
    ``<split>.phn``; returns the utterance ids of the kept rows, in row order."""
    import shutil

    os.makedirs(dest, exist_ok=True)
    shutil.copyfile(dict_phn, os.path.join(dest, "dict.phn.txt"))
    placeholder = _read_lines(dict_phn)[0].split()[0]  # first dict symbol, always in the vocabulary
    with open(os.path.join(manifest_dir, f"{split}.tsv")) as f:
        header, *rows = [line.rstrip("\n") for line in f if line.strip()]
    uids = _read_lines(os.path.join(manifest_dir, f"{split}.uid"))
    assert len(rows) == len(uids), f"{split}: {len(rows)} tsv rows vs {len(uids)} uids"
    keep = range(shard, len(rows), num_shards)
    with open(os.path.join(dest, f"{split}.tsv"), "w") as f:
        f.write(header + "\n")
        for i in keep:
            f.write(rows[i] + "\n")
    with open(os.path.join(dest, f"{split}.phn"), "w") as f:
        for _ in keep:
            f.write(placeholder + "\n")
    return [uids[i] for i in keep]


def _run_infer(*, python_exe: str, fairseq_root: str, data: str, split: str, checkpoint: str,
               max_tokens: int, resdir: str, shim_dir: str, decoding: List[str]) -> str:
    """One ``examples.speech_recognition.new.infer`` call (production's command line); returns its output."""
    examples = _examples_dir(fairseq_root)
    conf = os.path.join(examples, "speech_recognition", "new", "conf")
    os.makedirs(resdir, exist_ok=True)
    args = [
        python_exe, "-m", "examples.speech_recognition.new.infer",
        f"--config-dir={conf}", "--config-name=infer",
        "task=audio_finetuning", f"task.data={data}", "task.labels=phn",
        *decoding,
        "common_eval.post_process=none",
        f"common_eval.path={checkpoint}",
        f"dataset.gen_subset={split}", f"dataset.max_tokens={max_tokens}",
        "dataset.skip_invalid_size_inputs_valid_test=false",  # a dropped utterance would break the join
        "distributed_training.distributed_world_size=1",
        f"common_eval.results_path={resdir}", f"decoding.results_path={resdir}",
        f"hydra.run.dir={resdir}",
    ]
    print("RUN:", " ".join(args), flush=True)
    out = sp.check_output(args, stderr=sp.STDOUT, text=True, env=_infer_env(shim_dir, examples))
    print(out, flush=True)
    return out


# -------------------------------------------------------------------------------------------------
# decode jobs
# -------------------------------------------------------------------------------------------------
class CtcPhoneDecodeJob(Job):
    """Viterbi phone decode of the CTC student per split -> ``hyps.json`` ``{split: {utt_id: "P1 P2"}}``
    and ``per.json`` ``{split: {"per", "errors", "reference_phones"}}`` for the splits with gold."""

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        manifests: Dict[str, tk.Path],
        checkpoint: tk.Path,
        dict_phn: tk.Path,
        gold: tk.Path,
        fairseq_python_exe: tk.Path,
        fairseq_root: tk.Path,
        max_tokens: int = DECODE_MAX_TOKENS,
    ):
        """
        :param manifests: split -> ``FairseqAudioManifestJob.out_dir``
        :param checkpoint: the student checkpoint (``training.w2vu2_ctc.get_last_checkpoint``)
        :param dict_phn: the student's ``FairseqCtcDataJob.out_dict_phn`` (its output index map)
        :param gold: ``GoldPhonesJob`` json ``{split: {utt_id: [phones]}}``
        """
        super().__init__()
        self.manifests = manifests
        self.checkpoint = checkpoint
        self.dict_phn = dict_phn
        self.gold = gold
        self.fairseq_python_exe = fairseq_python_exe
        self.fairseq_root = fairseq_root
        self.max_tokens = max_tokens

        self.out_per = self.output_path("per.json")
        self.out_hyps = self.output_path("hyps.json")
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 3, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with open(self.gold.get_path()) as f:
            gold = json.load(f)
        pers, hyps = {}, {}
        for split in sorted(self.manifests):
            data = os.path.abspath(f"data_{split}")
            uids = _build_decode_data_dir(manifest_dir=self.manifests[split].get_path(), split=split,
                                          dict_phn=self.dict_phn.get_path(), dest=data)
            resdir = os.path.abspath(f"decode_{split}")
            _run_infer(python_exe=self.fairseq_python_exe.get_path(), fairseq_root=self.fairseq_root.get_path(),
                       data=data, split=split, checkpoint=self.checkpoint.get_path(),
                       max_tokens=self.max_tokens, resdir=resdir, shim_dir=os.path.abspath(f"shim_{split}"),
                       decoding=["decoding.type=viterbi"])
            hyps[split] = parse_hypo_file(os.path.join(resdir, "hypo.units"), uids)
            if split in gold:
                errors = sum(_edit_distance(hyps[split][u].split(), gold[split][u]) for u in uids)
                n_ref = sum(len(gold[split][u]) for u in uids)
                pers[split] = {"per": errors / n_ref, "errors": errors, "reference_phones": n_ref}
                print(f"{split}: PER {errors / n_ref:.4f} ({errors}/{n_ref})", flush=True)
        with open(self.out_per.get_path(), "w") as f:
            json.dump(pers, f, indent=2)
        with open(self.out_hyps.get_path(), "w") as f:
            json.dump(hyps, f, indent=2)


class CtcWordDecodeJob(Job):
    """Lexicon + KenLM word decode of the CTC student -> ``word_hyps.json`` ``{split: {utt_id: words}}``
    and ``word_wer.json`` ``{split: wer}`` for the splits with references.

    One ``decode`` task per (split, shard); ``train_shards`` splits the ``train`` manifest into
    interleaved row slices ``[k::n]`` (production's ``_shard_manifest``); ``collect`` joins and scores.
    """

    requires_env = "w2vu"
    __sis_hash_exclude__ = {"train_shards": 1}

    def __init__(
        self,
        *,
        manifests: Dict[str, tk.Path],
        checkpoint: tk.Path,
        dict_phn: tk.Path,
        lexicon: tk.Path,
        lm: tk.Path,
        fairseq_python_exe: tk.Path,
        fairseq_root: tk.Path,
        word_refs: Optional[tk.Path] = None,
        beam: int = WORD_DECODE_BEAM,
        lm_weight: float = WORD_DECODE_LM_WEIGHT,
        word_score: float = WORD_DECODE_WORD_SCORE,
        max_tokens: int = DECODE_MAX_TOKENS,
        train_shards: int = 1,
    ):
        """
        :param manifests: split -> ``FairseqAudioManifestJob.out_dir``
        :param checkpoint: the student checkpoint
        :param dict_phn: the student's ``dict.phn.txt``
        :param lexicon: :class:`FlashlightLexiconJob` ``out_lexicon``
        :param lm: the word 4-gram ARPA, plain or gzipped (``lm.word_lm.official_4gram_arpa``; KenLM reads gzip)
        :param word_refs: :class:`OggZipWordRefsJob` ``out_refs``; splits without references get no WER
        """
        super().__init__()
        self.manifests = manifests
        self.checkpoint = checkpoint
        self.dict_phn = dict_phn
        self.lexicon = lexicon
        self.lm = lm
        self.fairseq_python_exe = fairseq_python_exe
        self.fairseq_root = fairseq_root
        self.word_refs = word_refs
        self.beam = beam
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.max_tokens = max_tokens
        self.train_shards = train_shards

        self.out_wer = self.output_path("word_wer.json")
        self.out_hyps = self.output_path("word_hyps.json")
        # KenLM holds the text ARPA in RAM inside the decode child (production 64 GB)
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 64, "time": 11.5, "cpu": 8}

    def _n_shards(self, split: str) -> int:
        return self.train_shards if split == "train" else 1

    def _tag(self, split: str, shard: int) -> str:
        return split if self._n_shards(split) == 1 else f"{split}_shard{shard}"

    def tasks(self):
        yield Task("decode", rqmt=self.rqmt,
                   args=[[s, k] for s in sorted(self.manifests) for k in range(self._n_shards(s))])
        yield Task("collect", mini_task=True)

    def decode(self, split: str, shard: int = 0):
        tag = self._tag(split, shard)
        data = os.path.abspath(f"data_{tag}")
        uids = _build_decode_data_dir(manifest_dir=self.manifests[split].get_path(), split=split,
                                      dict_phn=self.dict_phn.get_path(), dest=data,
                                      shard=shard, num_shards=self._n_shards(split))
        with open(os.path.join(data, f"{split}.uid"), "w") as f:  # the row -> utterance map of this task
            f.write("\n".join(uids) + "\n")
        resdir = os.path.abspath(f"decode_{tag}")
        _run_infer(python_exe=self.fairseq_python_exe.get_path(), fairseq_root=self.fairseq_root.get_path(),
                   data=data, split=split, checkpoint=self.checkpoint.get_path(), max_tokens=self.max_tokens,
                   resdir=resdir, shim_dir=os.path.abspath(f"shim_{tag}"),
                   decoding=["decoding.type=kenlm",
                             f"decoding.lexicon={self.lexicon.get_path()}",
                             f"decoding.lmpath={self.lm.get_path()}",
                             f"decoding.beam={self.beam}",
                             f"decoding.lmweight={self.lm_weight}",
                             f"decoding.wordscore={self.word_score}"])
        # infer's own "Word error rate" scores words against the placeholder .phn -> not read
        assert os.path.exists(os.path.join(resdir, "hypo.word")), f"no hypo.word for {tag}"

    def collect(self):
        refs = {}
        if self.word_refs is not None:
            with open(self.word_refs.get_path()) as f:
                refs = json.load(f)
        wers, hyps = {}, {}
        for split in sorted(self.manifests):
            merged: Dict[str, str] = {}
            for k in range(self._n_shards(split)):
                tag = self._tag(split, k)
                uids = _read_lines(os.path.abspath(os.path.join(f"data_{tag}", f"{split}.uid")))
                merged.update(parse_hypo_file(os.path.abspath(os.path.join(f"decode_{tag}", "hypo.word")), uids))
            all_uids = _read_lines(os.path.join(self.manifests[split].get_path(), f"{split}.uid"))
            assert set(merged) == set(all_uids), f"{split}: {len(merged)} hyps vs {len(all_uids)} uids"
            hyps[split] = {uid: merged[uid] for uid in all_uids}
            print(f"{split}: {len(all_uids)} hyps, {sum(1 for v in merged.values() if not v)} empty", flush=True)
            if split in refs:
                errs = total = 0
                for uid in all_uids:
                    ref = refs[split][uid].upper().split()
                    errs += _edit_distance(hyps[split][uid].upper().split(), ref)
                    total += len(ref)
                wers[split] = errs / total
                print(f"{split}: word WER {wers[split]:.4f} ({errs}/{total})", flush=True)
        with open(self.out_wer.get_path(), "w") as f:
            json.dump(wers, f, indent=2)
        with open(self.out_hyps.get_path(), "w") as f:
            json.dump(hyps, f, indent=2)
