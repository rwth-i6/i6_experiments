"""Tests of the section 1c GAN inputs and training graph: ``w2vu2_tools``, ``data.w2vu2_features``,
``lm.w2vu2_text`` and ``training.w2vu2_gan``.

Fast tests use tiny synthetic inputs.  The ``artefact`` tests read the banked production outputs of
the source setup (the job dirs named below) and skip when a banked path is absent; they are skipped
unless ``SAE_ARTEFACT_DIR`` is set (``conftest.py``).  The resolved-config test runs the ``w2vu``
python (fairseq 0.12.2) in a subprocess; nothing in the package imports fairseq.

Run (CPU, login node)::

    PYTHONPATH=<recipe>:<recipe>/returnn:<sisyphus> SAE_ARTEFACT_DIR=1 \\
        python -m pytest -p no:cacheprovider tests/test_w2vu2_gan_inputs.py
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
import struct
import subprocess as sp
import sys
from collections import Counter

import numpy as np
import pytest
import yaml
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr import w2vu2_tools as TOOLS
from i6_experiments.users.wu.experiments.unsupervised_asr.data import w2vu2_features as F
from i6_experiments.users.wu.experiments.unsupervised_asr.lm import w2vu2_text as TXT
from i6_experiments.users.wu.experiments.unsupervised_asr.training import w2vu2_gan as G

# ---- the banked production outputs (source setup 2026-07-13_unsupervised) ---------------------------
_WORK = "/e/project1/spell/wu24/2026-07-13_unsupervised/work"
_W2VU2 = f"{_WORK}/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2"
BANKED_GAN = {
    0: f"{_W2VU2}/gan/FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output",
    1: f"{_W2VU2}/gan/FairseqW2vu2TrainJob.KPeBBDiEJMxT/output",
    2: f"{_W2VU2}/gan/FairseqW2vu2TrainJob.MbO9o9hBZs2G/output",
    3: f"{_W2VU2}/gan/FairseqW2vu2TrainJob.zygHGPvCrQZn/output",
    4: f"{_W2VU2}/gan/FairseqW2vu2TrainJob.otFs6lCBF3wX/output",
}
BANKED_DATA = f"{_W2VU2}/features/MergeW2vu2DataJob.wxBxCIaJpqS2/output/data"
BANKED_CENTROIDS = f"{_W2VU2}/features/MfccKmeansJob.yRS4DGsXvqKm/output/mfcc_centroids.npy"
BANKED_SIL_TEXT = f"{_W2VU2}/text/PhonemizeWithSilJob.DbFgvZOGZQ8F/output/text.phn.gz"
BANKED_TEXT_DATA = f"{_W2VU2}/text/FairseqPreprocessTextJob.bi2B89fES77z/output/text_data"
BANKED_KENLM_BIN = "/e/project1/spell/wu24/2026-07-13_unsupervised/output/sae/1a/phoneme_lm_o4.bin"
BANKED_LM_CORPUS = f"{_WORK}/i6_core/tools/download/DownloadJob.g4jClO48cAvP/output/librispeech-lm-norm.txt.gz"
BANKED_BLISS_LEXICON = f"{_WORK}/i6_core/lexicon/modification/MergeLexiconJob.qKaOAPqURCkK/output/lexicon.xml.gz"
BANKED_G2P_LEXICON = f"{_WORK}/i6_core/g2p/apply/ApplyG2PModelJob.myTIGtmrUIFq/output/g2p.lexicon"
BANKED_PHON_TEXT = (f"{_WORK}/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/"
                    "TextToPhonemeJob.THKMON3k9LJQ/output/phon.txt.gz")
BANKED_VAD = f"{_WORK}/speech_llm/sae/emc/blankfree_data_jobs/BlankfreeVadHdfJob.SAjz8y1cT06g/output"
BANKED_HF = f"{_WORK}/i6_core/datasets/huggingface/TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb/output/dataset"
# the source's w2vu env (fairseq 0.12.2 wheel); its fairseq_cli/, fairseq/config/ and
# fairseq/examples/ equal the v0.12.2 tag's (w2vu2_tools docstring)
W2VU_PREFIX = "/e/project1/spell/wu24/env/conda/envs/w2vu"
W2VU_SITE = f"{W2VU_PREFIX}/lib/python3.9/site-packages"


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"banked path absent: {p}")


# ===================================================================================================
# lm.w2vu2_text: the fairseq-preprocess reproduction
# ===================================================================================================
def test_fairseq_dictionary_order_threshold_and_eos_count():
    counts = Counter({"b": 5, "a": 5, "c": 7, "d": 2})
    symbols, sym_counts = TXT.fairseq_dictionary(counts, n_lines=4, threshold=3)
    # specials first; </s> counted once per line (+1 for its own add_symbol); ties alphabetical
    assert symbols == ["<s>", "<pad>", "</s>", "<unk>", "c", "a", "b"]
    assert sym_counts == [1, 1, 5, 1, 7, 5, 5]


def test_write_fairseq_text_data_layout(tmp_path):
    text = tmp_path / "text.txt"
    text.write_text("x y x\ny z\nx\n")
    out = {k: str(tmp_path / k) for k in ("dict.txt", "train.bin", "train.idx")}
    info = TXT.write_fairseq_text_data(str(text), dict_path=out["dict.txt"], bin_path=out["train.bin"],
                                      idx_path=out["train.idx"], threshold=2)
    assert open(out["dict.txt"]).read() == "x 3\ny 2\n"  # z (1) is below the threshold -> <unk>
    ids = np.fromfile(out["train.bin"], dtype="<u2")
    assert ids.tolist() == [4, 5, 4, 2, 5, 3, 2, 4, 2]
    assert info == {"lines": 3, "tokens": 9, "symbols": 6, "unk": 1}
    raw = open(out["train.idx"], "rb").read()
    assert raw[:9] == TXT.MMAP_INDEX_MAGIC
    version, code, n = struct.unpack("<QBQ", raw[9:26])
    assert (version, code, n) == (1, 8, 3)
    sizes = np.frombuffer(raw[26:26 + 4 * n], dtype="<i4")
    pointers = np.frombuffer(raw[26 + 4 * n:], dtype="<i8")
    assert sizes.tolist() == [4, 3, 2] and pointers.tolist() == [0, 8, 14]


@pytest.mark.artefact
@pytest.mark.slow
def test_text_data_byte_equal_to_banked(tmp_path):
    """(b) the p_sil 0.5 text of the source (DbFgvZOGZQ8F) binarized in-process equals
    FairseqPreprocessTextJob.bi2B89fES77z byte for byte (~15 min, 7 GB in tmp_path)."""
    _need(BANKED_SIL_TEXT, BANKED_TEXT_DATA)
    out = {k: str(tmp_path / k) for k in ("dict.txt", "train.bin", "train.idx")}
    TXT.write_fairseq_text_data(BANKED_SIL_TEXT, dict_path=out["dict.txt"], bin_path=out["train.bin"],
                                idx_path=out["train.idx"], threshold=1000)
    for name, path in out.items():
        assert sp.call(["cmp", os.path.join(BANKED_TEXT_DATA, name), path]) == 0, name


@pytest.mark.artefact
def test_phone_lm_text_prefix_equal_to_banked(tmp_path):
    """(d, prefix) TextToPhonemeJob on the first 5000 lines of the LM corpus, with the banked lexicon
    and g2p, equals the first lines of the banked TextToPhonemeJob.THKMON3k9LJQ output."""
    _need(BANKED_LM_CORPUS, BANKED_BLISS_LEXICON, BANKED_G2P_LEXICON, BANKED_PHON_TEXT)
    head = tmp_path / "head.txt"
    with gzip.open(BANKED_LM_CORPUS, "rt") as f, open(head, "w") as g:
        for i, line in enumerate(f):
            if i == 5000:
                break
            g.write(line)
    job = TXT.TextToPhonemeJob(text_file=tk.Path(str(head)), bliss_lexicon=tk.Path(BANKED_BLISS_LEXICON),
                               g2p_lexicon=tk.Path(BANKED_G2P_LEXICON), gzip_output=False)
    job.out_text = tk.Path(str(tmp_path / "phon.txt"))
    job.out_unresolved = tk.Path(str(tmp_path / "unresolved.txt"))
    for v in ("out_num_sentences_in", "out_num_sentences_out", "out_num_unresolved"):
        setattr(job, v, _Var())
    job.run()
    mine = open(tmp_path / "phon.txt").read().splitlines()
    assert len(mine) > 4000
    with gzip.open(BANKED_PHON_TEXT, "rt") as f:
        banked = [next(f).rstrip("\n") for _ in range(len(mine))]
    assert mine == banked


class _Var:
    def set(self, v):
        self.value = v


# ===================================================================================================
# training.w2vu2_gan: the config
# ===================================================================================================
def test_base_config_is_the_yaml_with_the_two_source_edits():
    raw = yaml.safe_load(G.W2VU2_YAML)
    cfg = G.w2vu2_base_config()
    assert "hydra" in raw and "hydra" not in cfg
    for group in ("generator", "discriminator"):
        assert raw["optimizer"]["groups"][group]["optimizer"]["amsgrad"] is False
        assert "amsgrad" not in cfg["optimizer"]["groups"][group]["optimizer"]
    raw.pop("hydra")
    for group in ("generator", "discriminator"):
        raw["optimizer"]["groups"][group]["optimizer"].pop("amsgrad")
    assert raw == cfg


def test_base_config_refuses_amsgrad_true(monkeypatch):
    monkeypatch.setattr(G, "W2VU2_YAML", G.W2VU2_YAML.replace("amsgrad: false", "amsgrad: true", 1))
    with pytest.raises(AssertionError, match="amsgrad"):
        G.w2vu2_base_config()


def _paths():
    return dict(data_dir=tk.Path("/d/data"), text_data=tk.Path("/d/text"), kenlm_path=tk.Path("/d/lm.bin"))


def test_gan_config_values():
    ov = G.w2vu2_overrides(input_dim=1024, generator_stride=3, generator_kernel=9, seed=3)
    cfg = G.w2vu2_gan_config(overrides=ov, user_dir=tk.Path("/fs/examples/wav2vec/unsupervised"), **_paths())
    d = cfg.config_dict
    assert d["model"]["input_dim"] == 1024 and d["model"]["generator_stride"] == 3
    assert d["model"]["generator_kernel"] == 9 and d["model"]["generator_batch_norm"] == 30
    assert d["model"]["target_downsample_rate"] == 1 and d["model"]["mmi_weight"] == 0.5
    assert (d["model"]["gradient_penalty"], d["model"]["smoothness_weight"], d["model"]["code_penalty"]) == (1.0, 1.5, 3.0)
    assert d["common"]["seed"] == 3 and d["task"]["aux_target_postfix"] == "km"
    assert d["distributed_training"]["distributed_world_size"] == 1
    assert d["lr_scheduler"] == {"_name": "pass_through"}
    # owned by FairseqHydraTrainingJob / its command line
    assert "max_update" not in d["optimization"] and "save_interval" not in d["checkpoint"]
    assert "save_dir" not in d["checkpoint"]


def test_build_w2vu2_gan_job(tmp_path):
    root = tk.Path("/fs/root")
    job = G.build_w2vu2_gan(seed=0, fairseq_python_exe=tk.Path("/env/bin/w2vu-python"), fairseq_root=root, **_paths())
    assert job.rqmt == {"gpu": 1, "cpu": 8, "mem": 100, "time": 11.5, "gpu_mem": 80}
    assert job.keep_epochs == set()
    path = tmp_path / "c.yaml"
    job.fairseq_hydra_config.write(str(path))
    written = yaml.safe_load(path.read_text())
    assert written["optimization"]["max_update"] == 150000 and written["optimization"]["max_epoch"] == 0
    assert written["checkpoint"]["save_interval"] == 1000
    assert written["common"]["user_dir"] == "/fs/root/examples/wav2vec/unsupervised"
    assert written["task"]["data"] == "/d/data" and written["task"]["kenlm_path"] == "/d/lm.bin"
    other = G.build_w2vu2_gan(seed=1, fairseq_python_exe=tk.Path("/env/bin/w2vu-python"), fairseq_root=root, **_paths())
    assert other._sis_id() != job._sis_id()


def test_tools_fixed_hashes(monkeypatch):
    from sisyphus import gs

    monkeypatch.setattr(gs, "W2VU_PYTHON", None, raising=False)
    with pytest.raises(RuntimeError, match="W2VU_PYTHON"):
        TOOLS.get_w2vu_python()
    monkeypatch.setattr(gs, "W2VU_PYTHON", "/a/bin/w2vu-python", raising=False)
    a = TOOLS.get_w2vu_python()
    monkeypatch.setattr(gs, "W2VU_PYTHON", "/b/bin/w2vu-python", raising=False)
    from sisyphus.hash import short_hash

    assert short_hash(a) == short_hash(TOOLS.get_w2vu_python())
    monkeypatch.setattr(gs, "W2VU_FAIRSEQ_ROOT", "/some/fairseq", raising=False)
    given = TOOLS.get_fairseq_root()
    monkeypatch.setattr(gs, "W2VU_FAIRSEQ_ROOT", None, raising=False)
    cloned = TOOLS.get_fairseq_root()
    assert short_hash(given) == short_hash(cloned)
    assert cloned.creator.files_to_checkout == TOOLS.FAIRSEQ_CHECKOUT_FILES


@pytest.mark.artefact
def test_vendored_yaml_equals_the_installed_one():
    src = f"{W2VU_SITE}/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu2.yaml"
    _need(src)
    assert open(src).read() == G.W2VU2_YAML


_STOP_AFTER_CFG = r'''
import os, runpy, sys
real = sys.argv[1]
sys.argv = [real] + sys.argv[2:]
sys.path[0] = os.path.dirname(real)
import fairseq, fairseq.tasks
assert fairseq.__version__ == "0.12.2", fairseq.__version__
print("PROBE_FAIRSEQ", fairseq.__file__, flush=True)
class Stop(Exception):
    pass
def _stop(*a, **k):
    raise Stop()
fairseq.tasks.setup_task = _stop   # train.main logs the resolved cfg right before setup_task
try:
    runpy.run_path(real, run_name="__main__")
except Stop:
    pass
'''


def _resolved_cfg(log_text: str) -> dict:
    for line in log_text.splitlines():
        if "[fairseq_cli.train][INFO] - {" in line:
            s = re.sub(r"<(\w+\.\w+): -?\d+>", r"'\1'", line[line.index(" - {") + 3:].strip())
            return ast.literal_eval(s)
    raise AssertionError("no resolved config in the log")


def _flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else str(k)))
    else:
        out[p] = d
    return out


@pytest.mark.artefact
def test_resolved_config_equals_banked_s0_except_paths(tmp_path):
    """(a) the ported s0 job's command (i6_core's argv and PYTHONPATH) resolves in fairseq to the
    config the source's s0 run (HOb2GgtYT7Bc) logged, except checkpoint.save_dir and common.user_dir.

    The fairseq root is the sparse-checkout layout built from the w2vu env's own copies of the tag's
    fairseq_cli/, fairseq/config/ and examples/ (w2vu2_tools docstring)."""
    banked_log = f"{BANKED_GAN[0]}/train.log"
    _need(banked_log, f"{W2VU_PREFIX}/bin/python", BANKED_DATA, BANKED_TEXT_DATA, BANKED_KENLM_BIN)
    root = tmp_path / "fairseq"
    (root / "fairseq").mkdir(parents=True)
    os.symlink(f"{W2VU_SITE}/fairseq_cli", root / "fairseq_cli")
    os.symlink(f"{W2VU_SITE}/fairseq/examples", root / "examples")
    os.symlink(f"{W2VU_SITE}/fairseq/config", root / "fairseq" / "config")
    job = G.build_w2vu2_gan(seed=0, data_dir=tk.Path(BANKED_DATA), text_data=tk.Path(BANKED_TEXT_DATA),
                            kenlm_path=tk.Path(BANKED_KENLM_BIN), fairseq_python_exe=tk.Path(f"{W2VU_PREFIX}/bin/python"),
                            fairseq_root=tk.Path(str(root)))
    job.out_fairseq_hydra_yaml = tk.Path(str(tmp_path / "out" / "fairseq_hydra_config.yaml"))
    job.out_checkpoint_dir = tk.Path(str(tmp_path / "out" / "checkpoints"))
    os.makedirs(job.out_checkpoint_dir.get_path())
    job.fairseq_hydra_config.write(job.out_fairseq_hydra_yaml.get_path())
    cmd = job._get_run_cmd()
    shim = tmp_path / "stop_after_cfg.py"
    shim.write_text(_STOP_AFTER_CFG)
    env = dict(os.environ)
    # i6_core prepends the root; then what the w2vu-python wrapper of env/build_w2vu_env.sh sets
    env["PYTHONPATH"] = ":".join([f"{W2VU_PREFIX}/fairseq_shim", str(root)] + env.get("PYTHONPATH", "").split(":"))
    env["LD_LIBRARY_PATH"] = f"{W2VU_PREFIX}/lib:{W2VU_SITE}/torch/lib"
    env["PYTHONNOUSERSITE"] = "1"
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    res = sp.run([cmd[0], str(shim)] + cmd[1:], env=env, cwd=cwd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True,
                 timeout=900)
    assert f"PROBE_FAIRSEQ {W2VU_SITE}/fairseq/__init__.py" in res.stdout, res.stdout[-3000:]
    port = _flat(_resolved_cfg(res.stdout))
    banked = _flat(_resolved_cfg(open(banked_log).read()))
    diff = sorted(k for k in set(port) | set(banked) if port.get(k, "<absent>") != banked.get(k, "<absent>"))
    assert diff == ["checkpoint.save_dir", "common.user_dir"], [(k, banked.get(k), port.get(k)) for k in diff]
    assert port["common.user_dir"] == f"{root}/examples/wav2vec/unsupervised"


# ===================================================================================================
# training.w2vu2_gan: the selection
# ===================================================================================================
def _fake_run(d, best_value, best_update, last_update=150000):
    import torch

    os.makedirs(d, exist_ok=True)
    torch.save({"extra_state": {"best": best_value, "val_loss": best_value + 1},
                "optimizer_history": [{"num_updates": last_update}]}, os.path.join(d, "checkpoint_last.pt"))
    torch.save({"extra_state": {"best": best_value, "val_loss": best_value},
                "optimizer_history": [{"num_updates": best_update}]}, os.path.join(d, "checkpoint_best.pt"))


def _run_select(tmp_path, dirs):
    yamls = {}
    for s, d in dirs.items():
        y = os.path.join(d, "fairseq_hydra_config.yaml")
        open(y, "w").write(f"seed: {s}\n")
        yamls[s] = tk.Path(y)
    job = G.W2vu2GanSelectJob(checkpoint_dirs={s: tk.Path(d) for s, d in dirs.items()}, hydra_yamls=yamls)
    out = tmp_path / "sel"
    out.mkdir()
    job.out_checkpoint = tk.Path(str(out / "checkpoint_best.pt"))
    job.out_hydra_yaml = tk.Path(str(out / "fairseq_hydra_config.yaml"))
    job.out_selection = tk.Path(str(out / "selection.json"))
    job.out_seed = _Var()
    job.run()
    return job, json.load(open(out / "selection.json"))


def test_select_is_argmin_weighted_lm_ppl(tmp_path):
    dirs = {s: str(tmp_path / f"s{s}") for s in range(3)}
    for s, (v, u) in enumerate([(20.0, 5000), (15.5, 7000), (15.5, 9000)]):
        _fake_run(dirs[s], v, u)
    job, sel = _run_select(tmp_path, dirs)
    assert sel["selected_seed"] == 1 and job.out_seed.value == 1  # tie -> the lower seed
    assert os.readlink(job.out_checkpoint.get_path()) == os.path.join(dirs[1], "checkpoint_best.pt")
    assert open(job.out_hydra_yaml.get_path()).read() == "seed: 1\n"
    assert sel["per_seed"]["2"] == {"weighted_lm_ppl": 15.5, "best_update": 9000, "last_update": 150000}


@pytest.mark.artefact
def test_select_reproduces_banked_seed0(tmp_path):
    """(e) the five banked seeds select seed 0 at weighted_lm_ppl 15.85."""
    dirs = {s: f"{d}/train" for s, d in BANKED_GAN.items()}
    _need(*[f"{d}/checkpoint_last.pt" for d in dirs.values()])
    job = G.W2vu2GanSelectJob(checkpoint_dirs={s: tk.Path(d) for s, d in dirs.items()},
                              hydra_yamls={s: tk.Path(f"{BANKED_GAN[s]}/train.log") for s in dirs})
    out = tmp_path / "sel"
    out.mkdir()
    job.out_checkpoint = tk.Path(str(out / "checkpoint_best.pt"))
    job.out_hydra_yaml = tk.Path(str(out / "yaml"))
    job.out_selection = tk.Path(str(out / "selection.json"))
    job.out_seed = _Var()
    job.run()
    sel = json.load(open(out / "selection.json"))
    assert sel["selected_seed"] == 0
    assert round(sel["per_seed"]["0"]["weighted_lm_ppl"], 2) == 15.85
    assert sel["per_seed"]["0"]["best_update"] == 148000


# ===================================================================================================
# data.w2vu2_features: the converter
# ===================================================================================================
def _write_stream(tmp_path, name, utts):
    """RETURNN HDF layout of BlankfreeVadHdfJob: feats (N, D) f16, raw_index (N,) i32, orig_length (n,) i32."""
    import h5py

    paths = [str(tmp_path / f"{k}.{name}.hdf") for k in ("feats", "raw_index", "orig_length")]
    tags = np.array([u.encode() for u, _, _, _ in utts])
    with h5py.File(paths[0], "w") as f:
        f["inputs"] = np.concatenate([x for _, x, _, _ in utts]).astype("<f2")
        f["seqLengths"] = np.array([[len(x), 1] for _, x, _, _ in utts], dtype=np.int32)
        f["seqTags"] = tags
    with h5py.File(paths[1], "w") as f:
        f["inputs"] = np.concatenate([r for _, _, r, _ in utts]).astype(np.int32)
        f["seqLengths"] = np.array([[len(r), 1] for _, _, r, _ in utts], dtype=np.int32)
        f["seqTags"] = tags
    with h5py.File(paths[2], "w") as f:
        f["inputs"] = np.array([o for _, _, _, o in utts], dtype=np.int32)
        f["seqLengths"] = np.ones((len(utts), 2), dtype=np.int32)
        f["seqTags"] = tags
    return tuple(paths)


def test_convert_split_frame_rule_and_layout(tmp_path):
    pytest.importorskip("torchaudio")
    rng = np.random.RandomState(0)
    audio = {u: (0.1 * rng.randn(16000 * s)).astype(np.float32) for u, s in (("a", 1), ("b", 1), ("c", 1))}
    n_mf = len(F.mfcc_39(audio["a"], 2))  # 1 s at 100 Hz, every 2nd frame
    D = F.FEATURE_DIM
    # a: kept frames incl. two at/after len(mfcc) (dropped by min(t, len(mfcc))); b: 2 kept (< min_length)
    ra = np.array([0, 3, 4, 10, n_mf - 1, n_mf, n_mf + 1])
    utts = [("a", rng.randn(len(ra), D), ra, n_mf + 2), ("b", rng.randn(2, D), np.array([1, 2]), 40),
            ("c", rng.randn(3, D), np.array([5, 6, 7]), 30)]
    stream = _write_stream(tmp_path, "x", utts)
    cent = rng.randn(64, 39).astype(np.float32)
    out = tmp_path / "out"
    out.mkdir()
    stats = F.convert_split(name="train", streams=[stream], audio=audio, centroids=cent, out_dir=str(out))
    assert stats["utts"] == 2 and stats["utts_dropped_short"] == 1 and stats["frames_beyond_mfcc"] == 2
    assert stats["frames_raw"] == n_mf + 40 + 30
    assert open(out / "train.ids").read() == "a\nc\n"
    assert open(out / "train.lengths").read() == f"{len(ra) - 2}\n3\n"
    x = np.load(out / "train.npy")
    assert x.dtype == np.float16 and x.shape == (len(ra) - 2 + 3, D)
    assert np.array_equal(x[: len(ra) - 2], utts[0][1][: len(ra) - 2].astype("<f2"))
    assert np.array_equal(x[len(ra) - 2:], utts[2][1].astype("<f2"))
    km = open(out / "train.km").read().splitlines()
    mf = F.mfcc_39(audio["a"], 2)
    exp = F.assign_km(mf[ra[:-2]], cent, (cent ** 2).sum(1))
    assert km[0] == " ".join(map(str, exp.tolist())) and len(km[1].split()) == 3


def test_silence_mask_truncates_or_pads_as_silence():
    class _Vad:
        def __call__(self, wav, sr):
            return np.array([1, 1, 0, 0, 1, 1]), None  # 3 encoder frames at 2 subframes: speech, sil, speech

    wav = np.zeros(16000, dtype=np.float32)
    assert F.silence_mask(wav, 2, 2, _Vad()).tolist() == [False, True]
    assert F.silence_mask(wav, 5, 2, _Vad()).tolist() == [False, True, False, True, True]


@pytest.mark.artefact
@pytest.mark.slow
def test_converter_reproduces_banked_data(tmp_path):
    """(c) convert_split on the first 25 utterances of every SAjz8y1cT06g split, with the banked
    audio (the HF dataset the source dumped from) and the banked centroids: lengths and .km equal
    wxBxCIaJpqS2's, the features are the HDF's rows; the feature difference to the source's own dump
    (a different wav2vec2 forward) is printed, not asserted."""
    pytest.importorskip("torchaudio")
    _need(BANKED_VAD, BANKED_HF, BANKED_DATA, BANKED_CENTROIDS)
    import h5py
    from datasets import Audio, load_from_disk

    hf = load_from_disk(BANKED_HF)
    cent = np.load(BANKED_CENTROIDS)
    n = 25
    for name, vad_splits, hf_split in (("train", ["train"], "train"), ("valid", ["dev-clean", "dev-other"], "dev")):
        ds = hf[hf_split].cast_column("audio", Audio(sampling_rate=16000))
        pos = {str(u): i for i, u in enumerate(ds["id"])}

        class _Audio:
            def __getitem__(self, utt):
                return np.asarray(ds[pos[utt]]["audio"]["array"], dtype=np.float32)

        streams = [tuple(f"{BANKED_VAD}/{k}.{s}.shard0.hdf" for k in ("feats", "raw_index", "orig_length"))
                   for s in vad_splits]
        out = tmp_path / name
        out.mkdir()
        F.convert_split(name=name, streams=streams, audio=_Audio(), centroids=cent, out_dir=str(out), limit=n)
        ids = open(out / f"{name}.ids").read().split()
        p_len = np.loadtxt(out / f"{name}.lengths", dtype=np.int64, ndmin=1)
        p_npy = np.load(out / f"{name}.npy")
        p_km = open(out / f"{name}.km").read().splitlines()
        assert len(ids) == n * len(vad_splits)
        rows = []
        for s in vad_splits:
            with h5py.File(f"{BANKED_VAD}/feats.{s}.shard0.hdf") as f:
                rows.append(np.asarray(f["inputs"][: int(f["seqLengths"][:n, 0].sum())]))
        assert np.array_equal(np.concatenate(rows), p_npy)  # every stream frame kept (none beyond mfcc)
        b_ids = open(f"{BANKED_DATA}/{name}.ids").read().split()
        row = {u: i for i, u in enumerate(b_ids)}
        b_len = np.loadtxt(f"{BANKED_DATA}/{name}.lengths", dtype=np.int64)
        b_off = np.concatenate([[0], np.cumsum(b_len)])
        b_km = open(f"{BANKED_DATA}/{name}.km").read().splitlines()
        b_npy = np.load(f"{BANKED_DATA}/{name}.npy", mmap_mode="r")
        p_off = np.concatenate([[0], np.cumsum(p_len)])
        max_abs = 0.0
        for i, u in enumerate(ids):
            r = row[u]
            assert p_len[i] == b_len[r], u
            assert p_km[i] == b_km[r], u
            a = p_npy[p_off[i]:p_off[i + 1]].astype(np.float32)
            b = np.asarray(b_npy[b_off[r]:b_off[r + 1]], dtype=np.float32)
            max_abs = max(max_abs, float(np.abs(a - b).max()))
        print(f"{name}: {len(ids)} utts, lengths and km equal; max |feature - source feature| = {max_abs}",
              file=sys.stderr)
