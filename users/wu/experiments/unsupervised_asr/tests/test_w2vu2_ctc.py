"""Tests of the section 1d CTC student: the fairseq training job (``training.w2vu2_ctc``) and its fairseq
CLI decodes (``analysis.w2vu2_ctc_decode``).

Fast tests use tiny synthetic inputs.  The ``artefact`` tests read the banked production outputs
(``Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0``, ``Wav2Vec2CtcDecodeJob.1C2fmWJR1mcM``,
``Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks`` and their inputs) and run fairseq under the source's ``w2vu``
python in a subprocess; they are skipped unless ``SAE_ARTEFACT_DIR`` is set (``conftest.py``) and skip
when a banked path is absent.  Nothing in the package imports fairseq.

Run (CPU, login node)::

    PYTHONPATH=<recipe>:<recipe>/returnn:<sisyphus> SAE_ARTEFACT_DIR=1 \\
        python -m pytest -p no:cacheprovider tests/test_w2vu2_ctc.py
"""

from __future__ import annotations

import ast
import json
import os
import subprocess as sp
import zipfile

import numpy as np
import pytest
import yaml
from sisyphus import gs, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import w2vu2_ctc_decode as D
from i6_experiments.users.wu.experiments.unsupervised_asr.training import w2vu2_ctc as T

W2VU_PYTHON = "/e/project1/spell/wu24/env/conda/envs/w2vu/bin/python"
FAIRSEQ_ROOT = "/e/project1/spell/wu24/env/conda/envs/w2vu/lib/python3.9/site-packages"
_WORK = "/e/project1/spell/wu24/2026-07-13_unsupervised/work"
_ST = f"{_WORK}/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/selftrain"
_WD = f"{_WORK}/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode"
BANKED_TRAIN_LOG = f"{_ST}/Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0/output/train.log"
BANKED_CKPT = f"{_ST}/Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0/output/train/checkpoint_last.pt"
BANKED_DICT = f"{_ST}/Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0/output/dict.phn.txt"
BANKED_AUDIO = f"{_ST}/LibriStAudioJob.ToncoVtTsLHc/output/audio"
BANKED_PHONE_HYPS = f"{_ST}/Wav2Vec2CtcDecodeJob.1C2fmWJR1mcM/output/hyps.json"
BANKED_GOLD = f"{_WORK}/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json"
BANKED_WORD_HYPS = f"{_WD}/Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks/output/word_hyps.json"
BANKED_WORD_REFS = f"{_WD}/LibriSpeechWordRefsJob.1EsLvSbyl06D/output/word_refs.json"
BANKED_LEXICON = f"{_WD}/BuildFlashlightLexiconJob.6jvZkBm5gPbS/output/lexicon.phn.txt"
BANKED_RAW_LEXICON = f"{_WD}/FetchLibrispeechLmJob.bTGUMQQciD5S/output/librispeech-lexicon.txt"
BANKED_ARPA = f"{_WD}/FetchLibrispeechLmJob.bTGUMQQciD5S/output/4-gram.arpa"
# the FLAC bliss corpora of the setup (i6_experiments.common get_bliss_corpus_dict(audio_format="flac"))
BLISS = {
    "dev-clean": f"{_WORK}/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.wjSkfzJS1Ge2/output/corpus.xml.gz",
    "dev-other": f"{_WORK}/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.qlLkwjdH203i/output/corpus.xml.gz",
}
REFERENCE_FFMPEG = "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/ffmpeg"
N_DEV_OTHER = 20  # utterances of the decoder-equivalence check (the first rows of production's dev-other manifest)


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"banked path absent: {p}")


def _outputs(job, tmp, names):
    for attr, fn in names.items():
        setattr(job, attr, tk.Path(os.path.join(str(tmp), fn)))


@pytest.fixture
def sis_work(tmp_path, monkeypatch):
    """Job outputs under ``tmp_path/work``, cwd ``tmp_path/cwd`` (a job's work dir)."""
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})
    monkeypatch.setattr(gs, "WORK_DIR", str(tmp_path / "work"))
    os.makedirs(tmp_path / "cwd")
    monkeypatch.chdir(tmp_path / "cwd")
    return tmp_path


def _w2vu_env():
    """The w2vu child env as the reference ``settings.py`` builds it (no speech_llm libraries)."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


# -------------------------------------------------------------------------------------------------
# training: config
# -------------------------------------------------------------------------------------------------
def _training_job(tmp):
    data = tk.Path(os.path.join(str(tmp), "data"))
    w2v = tk.Path(os.path.join(str(tmp), "wav2vec_vox_new.pt"))
    job = T.build_ctc_training_job(data_dir=data, w2v_path=w2v, fairseq_python_exe=tk.Path(W2VU_PYTHON),
                                   fairseq_root=tk.Path(FAIRSEQ_ROOT))
    return job, data, w2v


def test_training_job_carries_production_config(sis_work):
    job, data, w2v = _training_job(sis_work)
    cfg = job.fairseq_hydra_config
    assert cfg.post_config_dict == {"optimization": {"max_epoch": 0, "max_update": 40000},
                                    "checkpoint": {"save_interval": 1}}
    expect = json.loads(json.dumps(T.PRODUCTION_FINETUNE_CONFIG))
    import copy

    got = copy.deepcopy(cfg.config_dict)
    assert got["task"].pop("data").get_path() == data.get_path()
    assert got["model"].pop("w2v_path").get_path() == w2v.get_path()
    assert json.loads(json.dumps(got)) == expect
    assert job.rqmt == {"gpu": 4, "gpu_mem": 80, "cpu": 16, "mem": 60, "time": 11.5}
    assert job.out_models == {}
    assert T.get_last_checkpoint(job).get_path() == os.path.join(job.out_checkpoint_dir.get_path(), "checkpoint_last.pt")
    # rqmt is not hashed; every hashed knob is
    job2 = T.build_ctc_training_job(data_dir=data, w2v_path=w2v, fairseq_python_exe=tk.Path(W2VU_PYTHON),
                                    fairseq_root=tk.Path(FAIRSEQ_ROOT), rqmt={"gpu": 4, "time": 2})
    assert job2 is job

    # (d) the yaml the job writes parses back to the config with the paths as strings
    os.makedirs(os.path.dirname(job.out_fairseq_hydra_yaml.get_path()), exist_ok=True)
    job.create_files()
    written = yaml.safe_load(open(job.out_fairseq_hydra_yaml.get_path()))
    assert written["task"]["data"] == data.get_path() and written["model"]["w2v_path"] == w2v.get_path()
    assert written["optimization"]["max_update"] == 40000 and written["optimization"]["max_epoch"] == 0
    assert written["optimizer"]["adam_eps"] == 1e-08 and written["optimization"]["lr"] == [3e-05]
    cmd = job._get_run_cmd()
    assert cmd[:2] == [W2VU_PYTHON, os.path.join(FAIRSEQ_ROOT, "fairseq_cli", "hydra_train.py")]
    assert cmd[-1] == "checkpoint.save_dir=" + job.out_checkpoint_dir.get_path()


def _parse_logged_cfg(line: str) -> dict:
    """The ``logger.info(cfg)`` dict of fairseq's train.py (bare enum names become strings)."""

    class _Names(dict):
        def __missing__(self, key):
            return key

    return eval(line[line.index("{"):], {"__builtins__": {}}, _Names())  # noqa: S307 (banked log line)


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, float) and np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _diff(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b)):
        p = f"{prefix}{k}"
        if k not in a or k not in b:
            out.append((p, a.get(k, "<absent>"), b.get(k, "<absent>")))
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += _diff(a[k], b[k], p + ".")
        elif a[k] != b[k]:
            out.append((p, a[k], b[k]))
    return out


# the fairseq config dump the w2vu python runs: fairseq_cli/hydra_train.py exactly as the job calls it,
# with call_main replaced by a dump of the config it would train with (the same object train.py logs)
_DUMP_CFG = r"""
import json, os, runpy, sys
import fairseq.distributed.utils as du
def _dump(cfg, main, **kw):
    from omegaconf import OmegaConf
    with open(os.environ["CFG_DUMP"], "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, default=str)
    sys.stdout.flush(); os._exit(0)
du.call_main = _dump
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

#: runtime values, not inputs: the init method fairseq's distributed launch writes into the config, and two
#: fields whose dataclass default is ``max(1, torch.cuda.device_count())`` (4 on the 4-GPU node, 1 here)
_RUNTIME_SET = {"distributed_training.distributed_init_method", "distributed_training.distributed_num_procs",
                "distributed_training.nprocs_per_node"}
#: ``model.data`` is fairseq's interpolation of ``task.data``
_PATHS = {"task.data", "model.data", "model.w2v_path", "checkpoint.save_dir"}


@pytest.mark.artefact
@pytest.mark.slow
def test_resolved_config_equals_production(sis_work):
    """(a) the config fairseq resolves from the job's yaml and command line == the banked run's, except paths."""
    _need(BANKED_TRAIN_LOG, W2VU_PYTHON)
    job, _, _ = _training_job(sis_work)
    os.makedirs(os.path.dirname(job.out_fairseq_hydra_yaml.get_path()), exist_ok=True)
    job.create_files()
    cmd = job._get_run_cmd()
    env = _w2vu_env()
    env["PYTHONPATH"] = os.pathsep.join([FAIRSEQ_ROOT, env.get("PYTHONPATH", "")])  # as the job's run()
    env["CFG_DUMP"] = str(sis_work / "resolved.json")
    res = sp.run([cmd[0], "-c", _DUMP_CFG] + cmd[1:], env=env, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    assert res.returncode == 0 and os.path.exists(env["CFG_DUMP"]), res.stdout[-3000:]
    port = _norm(json.load(open(env["CFG_DUMP"])))
    with open(BANKED_TRAIN_LOG) as f:
        line = next(l for l in f if "[fairseq_cli.train][INFO] - {'_name'" in l)
    banked = _norm(_parse_logged_cfg(line))
    diffs = _diff(banked, port)
    def _leaves(d):
        return sum(_leaves(v) if isinstance(v, dict) else 1 for v in d.values())

    print(f"resolved config: {_leaves(banked)} banked / {_leaves(port)} port settings, {len(diffs)} differ")
    for p, b, o in diffs:
        print(f"DIFF {p}: banked={b!r} port={o!r}")
    assert {p for p, _, _ in diffs if p not in _PATHS} <= _RUNTIME_SET, diffs
    assert port["task"]["data"] == job.fairseq_hydra_config.config_dict["task"]["data"]  # instanciated in place
    assert port["checkpoint"]["save_dir"] == job.out_checkpoint_dir.get_path()


# -------------------------------------------------------------------------------------------------
# training: data
# -------------------------------------------------------------------------------------------------
def test_manifest_job_writes_production_layout(sis_work):
    from test_data_ogg_zip import make_zip  # sibling test module (pytest prepends this dir)

    import soundfile as sf

    from i6_experiments.users.wu.experiments.unsupervised_asr.data.ogg_zip import iter_ogg_zip_audio

    zip_path, _ = make_zip(sis_work, corpus="dev-other")
    job = T.FairseqAudioManifestJob(ogg_zip=tk.Path(zip_path), name="dev-other")
    _outputs(job, sis_work, {"out_dir": "manifest"})
    job.run()
    root = str(sis_work / "manifest")
    tsv = open(os.path.join(root, "dev-other.tsv")).read().splitlines()
    uids = open(os.path.join(root, "dev-other.uid")).read().split()
    decoded = dict(iter_ogg_zip_audio(zip_path))
    assert tsv[0] == os.path.join(root, "audio") and uids == list(decoded)
    for line, uid in zip(tsv[1:], uids):
        rel, n = line.split("\t")
        assert rel == f"dev-other/{uid}.flac" and int(n) == len(decoded[uid])
        info = sf.info(os.path.join(tsv[0], rel))
        assert (info.format, info.subtype, info.samplerate, info.frames) == ("FLAC", "PCM_16", 16000, int(n))
        pcm, _ = sf.read(os.path.join(tsv[0], rel), dtype="int16")
        # soundfile's float -> PCM_16 write of the decoded waveform (production's LibriStAudioJob call)
        assert np.abs(pcm.astype(np.int32) - np.round(decoded[uid] * 32767).astype(np.int32)).max() <= 1


def test_ctc_data_job_matches_production_rules(sis_work):
    man = sis_work / "man"
    os.makedirs(man)
    (man / "train.tsv").write_text("/root/audio\ntrain/a.flac\t100\ntrain/b.flac\t200\ntrain/c.flac\t300\n")
    (man / "train.uid").write_text("a\nb\nc\n")
    labels = sis_work / "labels.json"
    labels.write_text(json.dumps({"labels": {"a": "AH B", "b": "  ", "c": "Z AH"}}))  # b: empty -> dropped
    gold = sis_work / "gold.json"
    gold.write_text(json.dumps({"dev-other": {"x": ["AA", "B"]}}))
    job = T.FairseqCtcDataJob(manifest_dir=tk.Path(str(man)), labels=tk.Path(str(labels)), gold=tk.Path(str(gold)))
    _outputs(job, sis_work, {"out_data_dir": "data", "out_dict_phn": "data/dict.phn.txt"})
    job.run()
    d = sis_work / "data"
    assert (d / "train.tsv").read_text() == "/root/audio\ntrain/a.flac\t100\ntrain/c.flac\t300\n"
    assert (d / "train.phn").read_text() == "AH B\nZ AH\n"
    assert (d / "dict.phn.txt").read_text() == "AA 1\nAH 1\nB 1\nZ 1\n"


@pytest.mark.artefact
def test_ctc_data_dict_equals_production(sis_work):
    """The dictionary rule on the banked pseudo labels and gold reproduces the banked dict.phn.txt."""
    labels = f"{_ST}/GanPseudoLabelJob.xjn6QnNqwEEH/output/labels.json"
    _need(labels, BANKED_GOLD, BANKED_DICT, f"{BANKED_AUDIO}/train.tsv")
    job = T.FairseqCtcDataJob(manifest_dir=tk.Path(BANKED_AUDIO), labels=tk.Path(labels), gold=tk.Path(BANKED_GOLD))
    _outputs(job, sis_work, {"out_data_dir": "data", "out_dict_phn": "data/dict.phn.txt"})
    job.run()
    assert open(sis_work / "data" / "dict.phn.txt").read() == open(BANKED_DICT).read()
    assert len(open(sis_work / "data" / "train.phn").read().splitlines()) == 28539


# -------------------------------------------------------------------------------------------------
# decode: helpers
# -------------------------------------------------------------------------------------------------
def test_parse_hypo_file_and_data_dir(tmp_path):
    p = tmp_path / "hypo.word"
    p.write_text("B WORD (None-1)\n (None-0)\n")
    assert D.parse_hypo_file(str(p), ["u0", "u1"]) == {"u0": "", "u1": "B WORD"}
    p.write_text("A (None-0)\n")
    with pytest.raises(AssertionError):
        D.parse_hypo_file(str(p), ["u0", "u1"])

    man = tmp_path / "man"
    os.makedirs(man)
    (man / "train.tsv").write_text("/r\na\t1\nb\t2\nc\t3\n")
    (man / "train.uid").write_text("ua\nub\nuc\n")
    dct = tmp_path / "dict.phn.txt"
    dct.write_text("AA 1\nAE 1\n")
    uids = D._build_decode_data_dir(manifest_dir=str(man), split="train", dict_phn=str(dct),
                                    dest=str(tmp_path / "d"), shard=1, num_shards=2)
    assert uids == ["ub"]
    assert (tmp_path / "d" / "train.tsv").read_text() == "/r\nb\t2\n"
    assert (tmp_path / "d" / "train.phn").read_text() == "AA\n"


def test_convert_lexicon_lines():
    entries, stats = D.convert_lexicon_lines(
        ["A AH0\n", "A AH1\n", "B B IY1\n", "C\n", "D ZZ1\n", "\n"], {"AH", "B", "IY"})
    assert entries == [("A", ("AH",)), ("B", ("B", "IY"))]
    assert stats == {"lines": 5, "no_pron": 1, "oov_phone": 1, "duplicate": 1, "kept": 2}


def test_word_collect_uses_production_wer_convention(sis_work):
    man = sis_work / "man"
    os.makedirs(man)
    (man / "dev-other.tsv").write_text("/r\na\t1\nb\t2\n")
    (man / "dev-other.uid").write_text("ua\nub\n")
    refs = sis_work / "refs.json"
    refs.write_text(json.dumps({"dev-other": {"ua": "the cat sat", "ub": "A DOG"}}))
    job = D.CtcWordDecodeJob(manifests={"dev-other": tk.Path(str(man))}, checkpoint=tk.Path("/c.pt"),
                             dict_phn=tk.Path("/d"), lexicon=tk.Path("/l"), lm=tk.Path("/lm"),
                             fairseq_python_exe=tk.Path(W2VU_PYTHON), fairseq_root=tk.Path(FAIRSEQ_ROOT),
                             word_refs=tk.Path(str(refs)))
    _outputs(job, sis_work, {"out_wer": "wer.json", "out_hyps": "hyps.json"})
    os.makedirs("data_dev-other")
    open("data_dev-other/dev-other.uid", "w").write("ua\nub\n")
    os.makedirs("decode_dev-other")
    open("decode_dev-other/hypo.word", "w").write("A DOG DOG (None-1)\nTHE BAT SAT (None-0)\n")
    job.collect()
    assert json.load(open(sis_work / "wer.json")) == {"dev-other": 2 / 5}  # 1 sub + 1 ins over 5 ref words
    assert json.load(open(sis_work / "hyps.json")) == {"dev-other": {"ua": "THE BAT SAT", "ub": "A DOG DOG"}}


# -------------------------------------------------------------------------------------------------
# decode: against production (artefact)
# -------------------------------------------------------------------------------------------------
@pytest.mark.artefact
def test_lexicon_job_equals_production(sis_work):
    _need(BANKED_RAW_LEXICON, BANKED_DICT, BANKED_LEXICON)
    job = D.FlashlightLexiconJob(lexicon=tk.Path(BANKED_RAW_LEXICON), dict_phn=tk.Path(BANKED_DICT))
    _outputs(job, sis_work, {"out_lexicon": "lexicon.phn.txt"})
    job.run()
    assert open(sis_work / "lexicon.phn.txt", "rb").read() == open(BANKED_LEXICON, "rb").read()


@pytest.mark.artefact
def test_bliss_orth_equals_production_word_refs():
    """The ogg zip ``text`` is the whitespace-normalised bliss orth (bliss-to-ogg-zip.py); on the
    setup's FLAC bliss corpora it equals production's HF references for every dev utterance."""
    from i6_core.lib import corpus

    _need(BANKED_WORD_REFS, *BLISS.values())
    banked = json.load(open(BANKED_WORD_REFS))
    for split, path in BLISS.items():
        c = corpus.Corpus()
        c.load(path)
        orth = {s.name: " ".join(s.orth.split()) for s in c.segments()}
        assert orth == banked[split], split
        print(f"{split}: {len(orth)} references equal")


def _first_dev_other_uids():
    return open(f"{BANKED_AUDIO}/dev-other.uid").read().split()[:N_DEV_OTHER]


@pytest.fixture
def port_dev_other(sis_work):
    """The port's audio path for the first N dev-other utterances: FLAC bliss -> pinned-ffmpeg Ogg
    Vorbis (i6_core's BlissChangeEncodingJob command) -> RETURNN ogg zip -> FairseqAudioManifestJob."""
    from i6_core.lib import corpus

    from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

    _need(BLISS["dev-other"], f"{BANKED_AUDIO}/dev-other.uid", REFERENCE_FFMPEG)
    uids = _first_dev_other_uids()
    c = corpus.Corpus()
    c.load(BLISS["dev-other"])
    recs = {r.name: r for r in c.all_recordings()}
    zip_path = str(sis_work / "dev-other.ogg.zip")
    name = "dev-other.ogg"
    entries = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for uid in uids:
            rec = recs[uid]
            ogg = FP.encode_recording(REFERENCE_FFMPEG, tk.Path(BLISS["dev-other"]), rec, str(sis_work / "ogg"))
            f = f"{uid}/{os.path.basename(ogg)}"
            zf.write(ogg, f"{name}/{f}")
            entries.append({"text": rec.segments[0].orth, "speaker_name": None, "file": f,
                            "seq_name": f"dev-other/{uid}/{uid}", "duration": 0.0})
        zf.writestr(f"{name}.txt", repr(entries))
    job = T.FairseqAudioManifestJob(ogg_zip=tk.Path(zip_path), name="dev-other")
    _outputs(job, sis_work, {"out_dir": "manifest"})
    job.run()
    return uids, tk.Path(zip_path), tk.Path(str(sis_work / "manifest"))


@pytest.mark.artefact
@pytest.mark.slow
def test_decodes_reproduce_production(port_dev_other, sis_work, monkeypatch):
    """(c) the port's audio path + both decode jobs on the first 20 dev-other utterances reproduce
    production's phone hypotheses and its lexicon + KenLM word hypotheses."""
    import soundfile as sf

    _need(BANKED_CKPT, BANKED_DICT, BANKED_ARPA, BANKED_WORD_HYPS, BANKED_PHONE_HYPS, BANKED_GOLD, W2VU_PYTHON)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    uids, zip_path, manifest = port_dev_other

    # audio: the port's FLAC against production's FLAC (both PCM_16)
    lsb = []
    for uid in uids:
        a, _ = sf.read(os.path.join(manifest.get_path(), "audio", "dev-other", f"{uid}.flac"), dtype="int16")
        b, _ = sf.read(os.path.join(BANKED_AUDIO, "audio", "dev", f"{uid}.flac"), dtype="int16")
        assert a.shape == b.shape, uid
        lsb.append(int(np.abs(a.astype(np.int32) - b).max()))
    print(f"audio: {sum(x == 0 for x in lsb)}/{len(uids)} bit-identical FLAC PCM, max |diff| {max(lsb)} LSB")

    refs_job = D.OggZipWordRefsJob(ogg_zips={"dev-other": zip_path})
    _outputs(refs_job, sis_work, {"out_refs": "word_refs.json"})
    refs_job.run()
    banked_refs = json.load(open(BANKED_WORD_REFS))["dev-other"]
    assert json.load(open(sis_work / "word_refs.json"))["dev-other"] == {u: banked_refs[u] for u in uids}

    lex_job = D.FlashlightLexiconJob(lexicon=tk.Path(BANKED_RAW_LEXICON), dict_phn=tk.Path(BANKED_DICT))
    _outputs(lex_job, sis_work, {"out_lexicon": "lexicon.phn.txt"})
    lex_job.run()

    # phone decode
    os.makedirs(sis_work / "phone_cwd")
    monkeypatch.chdir(sis_work / "phone_cwd")
    pjob = D.CtcPhoneDecodeJob(manifests={"dev-other": manifest}, checkpoint=tk.Path(BANKED_CKPT),
                               dict_phn=tk.Path(BANKED_DICT), gold=tk.Path(BANKED_GOLD),
                               fairseq_python_exe=tk.Path(W2VU_PYTHON), fairseq_root=tk.Path(FAIRSEQ_ROOT))
    _outputs(pjob, sis_work, {"out_per": "per.json", "out_hyps": "phone_hyps.json"})
    pjob.run()
    phyps = json.load(open(sis_work / "phone_hyps.json"))["dev-other"]
    banked_p = json.load(open(BANKED_PHONE_HYPS))["dev-other"]
    p_equal = sum(phyps[u] == banked_p[u] for u in uids)
    gold = json.load(open(BANKED_GOLD))["dev-other"]
    b_err = sum(D._edit_distance(banked_p[u].split(), gold[u]) for u in uids)
    per = json.load(open(sis_work / "per.json"))["dev-other"]
    print(f"phone decode: {p_equal}/{len(uids)} hypotheses equal to production; "
          f"PER port {per['per']:.4f} ({per['errors']}/{per['reference_phones']}) production {b_err / per['reference_phones']:.4f}")

    # word decode
    os.makedirs(sis_work / "word_cwd")
    monkeypatch.chdir(sis_work / "word_cwd")
    wjob = D.CtcWordDecodeJob(manifests={"dev-other": manifest}, checkpoint=tk.Path(BANKED_CKPT),
                              dict_phn=tk.Path(BANKED_DICT), lexicon=lex_job.out_lexicon, lm=tk.Path(BANKED_ARPA),
                              fairseq_python_exe=tk.Path(W2VU_PYTHON), fairseq_root=tk.Path(FAIRSEQ_ROOT),
                              word_refs=refs_job.out_refs)
    _outputs(wjob, sis_work, {"out_wer": "word_wer.json", "out_hyps": "word_hyps.json"})
    wjob.decode("dev-other", 0)
    wjob.collect()
    whyps = json.load(open(sis_work / "word_hyps.json"))["dev-other"]
    banked_w = json.load(open(BANKED_WORD_HYPS))["dev-other"]
    w_equal = sum(whyps[u] == banked_w[u] for u in uids)
    for u in uids:
        if whyps[u] != banked_w[u]:
            print(f"WORD DIFF {u}\n  port:       {whyps[u]}\n  production: {banked_w[u]}")
    print(f"word decode: {w_equal}/{len(uids)} hypotheses equal to production word for word; "
          f"WER port {json.load(open(sis_work / 'word_wer.json'))['dev-other']:.4f}")
    assert p_equal == len(uids) and w_equal == len(uids)
