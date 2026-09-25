"""Graph-build tests of ``config/w2vu2.py`` (the wav2vec-U 2.0 reproduction reference): ``py()`` builds on
CPU without running a job; the job types and counts, the seeds, the wiring, the rqmt, the ``w2vu2/``
scope of aliases and outputs, and the import allowlist."""

import ast
import os
import subprocess
import sys
import textwrap
from collections import Counter

import pytest
from sisyphus import gs, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.config import w2vu2

PKG = "i6_experiments.users.wu.experiments.unsupervised_asr"
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PKG_DIR, "config", "w2vu2.py")
DUMMY_W2VU_PYTHON = "/pinned/by/the/test/w2vu-python"

#: the job types this entry adds on top of the port's inputs, and how many of each
EXPECTED_COUNTS = {
    # 1c inputs (report A)
    "MfccKmeansJob": 1,
    "W2vu2FeatureDataJob": 1,
    "FairseqTextDataJob": 1,
    "TextToPhonemeJob": 1,
    "KenLMplzJob": 1,
    "CreateBinaryLMJob": 1,
    # 5 GAN seeds + the 1d CTC student
    "FairseqHydraTrainingJob": 6,
    "W2vu2GanSelectJob": 1,
    # report B: 5 seeds + the selected generator; 5 seeds x 2 dev splits + the train forward;
    # plus the intermediate eval: 5 seeds x 30 updates (5000 .. 150000), each on 2 dev splits
    "W2vu2GeneratorCheckpointJob": 6 + 150,
    "ReturnnForwardJobV2": 11 + 300,
    "W2vu2GanPerJob": 10 + 300,
    "W2vu2GanPseudoLabelJob": 1,
    # report C
    "FairseqAudioManifestJob": 3,
    "FairseqCtcDataJob": 1,
    "CtcPhoneDecodeJob": 1,
    "FlashlightLexiconJob": 1,
    "OggZipWordRefsJob": 1,
    "CtcWordDecodeJob": 1,
}

#: production's rqmt (banked ``job.save``), with the i6 changes (training gpu_mem 80 -> 32; GAN mem 60 -> 100)
GAN_RQMT = {"gpu": 1, "gpu_mem": 32, "mem": 100, "time": 11.5, "cpu": 8}
CTC_RQMT = {"gpu": 4, "gpu_mem": 32, "mem": 60, "time": 11.5, "cpu": 16}
FORWARD_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
#: the dev-clean / dev-other generator forwards (checkpoint_best and intermediate): FORWARD_RQMT on CPU
DEV_FORWARD_RQMT = {"gpu": 0, "mem": 24, "time": 2, "cpu": 4}
PHONE_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}
WORD_DECODE_RQMT = {"gpu": 1, "gpu_mem": 40, "mem": 64, "time": 11.5, "cpu": 8}
#: the intermediate-eval updates: every 5000, the epoch ends 45000 / 90000 / 135000 (180 updates per epoch)
#: moved one save (1000 updates) earlier
INTERMEDIATE_UPDATES = sorted({45000: 44000, 90000: 89000, 135000: 134000}.get(u, u)
                              for u in range(5000, 150_001, 5000))


@pytest.fixture
def built(monkeypatch):
    """``w2vu2.py()`` in a private job registry; returns ``(result, new jobs, registered outputs)``.

    The shared port inputs are built first, so ``new`` holds only the jobs this entry adds.  The
    settings define no training partition here (JUPITER; see :func:`built_i6`)."""
    monkeypatch.delattr(gs, w2vu2.I6_TRAIN_PARTITION_SETTING, raising=False)
    return _build(monkeypatch)


@pytest.fixture
def built_i6(monkeypatch):
    """:func:`built` under i6's settings constant ``GPU_ROUTE_TRAIN = "gpu_32gb"``."""
    monkeypatch.setattr(gs, w2vu2.I6_TRAIN_PARTITION_SETTING, "gpu_32gb", raising=False)
    return _build(monkeypatch)


def _build(monkeypatch):
    import sisyphus.job

    from i6_experiments.users.wu.experiments.unsupervised_asr.inputs import get_inputs

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})
    monkeypatch.setattr(gs, "FFMPEG_BINARY", "/pinned/by/the/test/ffmpeg", raising=False)
    monkeypatch.setattr(gs, "W2VU_PYTHON", DUMMY_W2VU_PYTHON, raising=False)
    monkeypatch.setattr(gs, "W2VU_FAIRSEQ_ROOT", None, raising=False)
    out = {}

    def _reg(name, value, export_graph=False):
        assert name not in out or out[name][1] is value, f"output {name} registered with two values"
        out[name] = (gs.ALIAS_AND_OUTPUT_SUBDIR, value)

    monkeypatch.setattr(tk, "register_output", _reg)
    monkeypatch.setattr(tk, "register_report", lambda *a, **k: None)
    get_inputs()
    before = set(sisyphus.job.created_jobs)
    out.clear()
    subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    res = w2vu2.py()
    assert gs.ALIAS_AND_OUTPUT_SUBDIR == subdir, "py() must restore ALIAS_AND_OUTPUT_SUBDIR"
    new = [j for k, j in sisyphus.job.created_jobs.items() if k not in before]
    return res, new, out


def _overwrite(path):
    ho = path.hash_overwrite
    return ho[-1] if isinstance(ho, tuple) else ho


def _same(a, b) -> bool:
    """The same sisyphus path (job arguments are deep-copied, so identity does not survive)."""
    return a.creator is b.creator and a.get_path() == b.get_path()


def _of(jobs, name):
    return [j for j in jobs if type(j).__name__ == name]


def test_job_types_and_counts(built):
    _, new, _ = built
    counts = Counter(type(j).__name__ for j in new)
    for name, n in EXPECTED_COUNTS.items():
        assert counts[name] == n, (name, counts[name], n)
    # nothing is built twice under different hashes
    ids = [j.job_id() for j in new]
    assert len(ids) == len(set(ids))


def test_production_seeds(built):
    res, _, _ = built
    assert w2vu2.PRODUCTION_SEEDS == (0, 1, 2, 3, 4)
    trainings = res["1c"]["gan"].trainings
    assert sorted(trainings) == [0, 1, 2, 3, 4]
    for seed, job in trainings.items():
        assert job.fairseq_hydra_config.config_dict["common"]["seed"] == seed
    assert len({j.job_id() for j in trainings.values()}) == 5
    select = res["1c"]["gan"].selection
    assert sorted(select.checkpoint_dirs) == [0, 1, 2, 3, 4]


def test_every_seed_is_evaluated_on_both_dev_splits(built):
    res, _, out = built
    evals = res["1c"]["eval"]
    trainings = res["1c"]["gan"].trainings
    vad = res["inputs"].vad
    assert sorted(evals) == list(w2vu2.PRODUCTION_SEEDS)
    for seed, ev in evals.items():
        conv = ev["generator"]
        assert conv.fairseq_checkpoint.creator is trainings[seed]
        assert conv.fairseq_checkpoint.path.endswith("checkpoint_best.pt")
        assert _same(conv.text_dict, res["1c"]["gan"].text_data.out_dict)
        for split in w2vu2.EVAL_SPLITS:
            fwd, per = ev[split]["forward"], ev[split]["per"]
            assert fwd.model_checkpoint.path.creator is conv
            assert _same(per.hyps, fwd.out_files["hyps.json"]) and per.split == split
            assert _same(per.gold, res["inputs"].gold)
            feats = fwd.returnn_config.config["forward_data"]["files"]
            assert feats == list(vad.out_feature_hdfs[split])
            for f in ("per.json", "per.txt"):
                assert f"{w2vu2.GAN_ARM}/s{seed}/{split}/{f}" in out


def test_pseudo_labels_come_from_the_selected_generator(built):
    res, _, _ = built
    sel = res["1c"]["select"]
    gan = res["1c"]["gan"]
    assert _same(sel["generator"].fairseq_checkpoint, gan.selection.out_checkpoint)
    assert sel["forward"].model_checkpoint.path.creator is sel["generator"]
    assert sel["forward"].returnn_config.config["forward_data"]["files"] == list(
        res["inputs"].vad.out_feature_hdfs["train"])
    labels = sel["labels"]
    assert labels.hyps == [sel["forward"].out_files["hyps.json"]] and labels.expected_num_seqs == 28539


def test_dev_forwards_of_best_and_intermediate_chains_run_on_cpu(built):
    """Every dev forward, of ``checkpoint_best.pt`` and of every intermediate checkpoint, uses device cpu."""
    res, _, _ = built
    evals, inter = res["1c"]["eval"], res["1c"]["intermediate"]
    assert sorted(inter) == list(w2vu2.PRODUCTION_SEEDS)
    forwards = [evals[s][split]["forward"] for s in evals for split in w2vu2.EVAL_SPLITS]
    for seed, points in inter.items():
        assert sorted(points) == INTERMEDIATE_UPDATES, (seed, sorted(points))
        for u in points:
            conv = points[u]["generator"]
            assert conv.fairseq_checkpoint.creator is res["1c"]["gan"].trainings[seed]
            assert conv.fairseq_checkpoint.path == f"checkpoints/{w2vu2.gan_update_checkpoint_name(u)}"
        forwards += [points[u][split]["forward"] for u in points for split in w2vu2.EVAL_SPLITS]
    assert len(forwards) == 10 + 300
    for fwd in forwards:
        assert fwd.device == "cpu" and fwd.rqmt["gpu"] == 0, (fwd.device, fwd.rqmt)


def test_intermediate_checkpoint_names_follow_fairseq():
    """180 updates per epoch (batch_by_size: 178 x 160 + 56 + 3); production's names; no epoch end."""
    assert w2vu2.GAN_UPDATES_PER_EPOCH == 180
    assert w2vu2.gan_update_checkpoint_name(148000) == "checkpoint_823_148000.pt"  # production s0 best
    assert w2vu2.gan_update_checkpoint_name(7000) == "checkpoint_39_7000.pt"
    assert list(w2vu2.intermediate_eval_updates(150_000)) == INTERMEDIATE_UPDATES
    assert all(u % 180 for u in INTERMEDIATE_UPDATES)
    with pytest.raises(AssertionError):
        w2vu2.gan_update_checkpoint_name(45000)  # an epoch end: fairseq writes no update-named file


def test_pseudo_label_forward_stays_on_gpu(built):
    res, _, _ = built
    fwd = res["1c"]["select"]["forward"]
    assert fwd.device == "gpu" and fwd.rqmt["gpu"] == 1 and fwd.rqmt["gpu_mem"] == 40, (fwd.device, fwd.rqmt)


def test_selftrain_wiring(built):
    res, _, out = built
    from i6_experiments.users.wu.experiments.unsupervised_asr.lm.word_lm import official_4gram_arpa, official_lexicon
    from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2_tools import (
        FAIRSEQ_ROOT_HASH_OVERWRITE,
        W2VU_PYTHON_HASH_OVERWRITE,
    )

    d = res["1d"]
    inputs = res["inputs"]
    assert {n: m.ogg_zip for n, m in d["manifests"].items()} == {
        "train": inputs.ogg_zips["train-clean-100"], "dev-clean": inputs.ogg_zips["dev-clean"],
        "dev-other": inputs.ogg_zips["dev-other"]}
    assert _same(d["data"].labels, res["1c"]["select"]["labels"].out_labels)
    assert _same(d["data"].manifest_dir, d["manifests"]["train"].out_dir)
    train = d["train"]
    assert _same(train.fairseq_hydra_config.config_dict["task"]["data"], d["data"].out_data_dir)
    assert train.fairseq_python_exe.get_path() == DUMMY_W2VU_PYTHON
    # sisyphus stores a str overwrite given to Path() as (None, str)
    assert _overwrite(train.fairseq_python_exe) == W2VU_PYTHON_HASH_OVERWRITE
    assert _overwrite(train.fairseq_root) == FAIRSEQ_ROOT_HASH_OVERWRITE
    for job in (d["phone"], d["word"]):
        assert sorted(job.manifests) == sorted(w2vu2.EVAL_SPLITS)
        assert job.checkpoint.creator is train and job.checkpoint.path.endswith("checkpoint_last.pt")
        assert _same(job.dict_phn, d["data"].out_dict_phn)
        assert _same(job.fairseq_python_exe, train.fairseq_python_exe)
    assert d["word"].lm.get_path() == official_4gram_arpa().get_path()
    assert d["lexicon"].lexicon.get_path() == official_lexicon().get_path()
    assert _same(d["word"].word_refs, d["refs"].out_refs)
    assert (d["word"].beam, d["word"].lm_weight, d["word"].word_score) == (500, 2.0, -1.0)
    assert sorted(d["refs"].ogg_zips) == sorted(w2vu2.EVAL_SPLITS)
    for name in ("per.json", "hyps.json", "word_wer.json", "word_hyps.json", "dict.phn.txt",
                 "pseudo_labels.json", "word_refs.json", "lm/lexicon.phn.txt"):
        assert f"{w2vu2.SELFTRAIN_PREFIX}/{name}" in out, name


def test_rqmt_is_production_except_training_gpu_mem(built):
    res, new, _ = built
    for job in res["1c"]["gan"].trainings.values():
        assert job.rqmt == GAN_RQMT, job.rqmt
    assert res["1d"]["train"].rqmt == CTC_RQMT, res["1d"]["train"].rqmt
    forwards = _of(new, "ReturnnForwardJobV2")
    select_forward = res["1c"]["select"]["forward"]
    dev_forwards = [j for j in forwards if j is not select_forward]
    assert len(forwards) == 311 and len(dev_forwards) == 310
    assert all(j.rqmt == DEV_FORWARD_RQMT for j in dev_forwards), [j.rqmt for j in dev_forwards]
    assert select_forward.rqmt == FORWARD_RQMT, select_forward.rqmt
    assert res["1d"]["phone"].rqmt == PHONE_DECODE_RQMT
    assert res["1d"]["word"].rqmt == WORD_DECODE_RQMT


def test_i6_trainings_carry_the_settings_partition_and_nothing_else_does(built, built_i6):
    """With i6's ``GPU_ROUTE_TRAIN`` the 6 trainings get ``-p gpu_32gb`` and nothing else changes: the
    other rqmt, and every job id (rqmt is not hashed)."""
    res, new, _ = built_i6
    trains = list(res["1c"]["gan"].trainings.values()) + [res["1d"]["train"]]
    for job, base in zip(trains, [GAN_RQMT] * 5 + [CTC_RQMT]):
        assert job.rqmt == dict(base, sbatch_args=["-p", "gpu_32gb"]), job.rqmt
    others = [j for j in new if j not in trains]
    assert not [j for j in others if "sbatch_args" in (getattr(j, "rqmt", None) or {})]
    assert sorted(j.job_id() for j in new) == sorted(j.job_id() for j in built[1])


def test_everything_is_scoped_under_w2vu2(built):
    _, new, out = built
    assert out, "no output registered"
    for name, (subdir, _) in out.items():
        assert subdir == w2vu2.ALIAS_AND_OUTPUT_PREFIX, name
        assert name.startswith(("sae/1c/", "sae/1d/")), name
    for job in new:
        assert w2vu2.ALIAS_AND_OUTPUT_PREFIX in job._sis_alias_prefixes, job.job_id()
    for name in ("FairseqHydraTrainingJob", "W2vu2GanSelectJob", "W2vu2GanPerJob", "CtcWordDecodeJob"):
        for job in _of(new, name):
            assert job.get_aliases() and all(a.startswith(("sae/1c/", "sae/1d/")) for a in job.get_aliases())


def test_config_imports_only_allowed_modules():
    """Static: every import of ``config/w2vu2.py`` is stdlib, sisyphus, or inside ``unsupervised_asr``."""
    with open(CONFIG_FILE) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level <= 2, f"relative import leaves the package: {ast.dump(node)}"
                continue
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top in sys.stdlib_module_names or top in ("sisyphus", "__future__"), name


def test_graph_build_imports_only_the_allowlist(tmp_path):
    """Dynamic, in a fresh interpreter: building the graph imports only stdlib, installed third-party
    packages, i6_core, i6_experiments.common, returnn, sisyphus and ``unsupervised_asr``."""
    script = textwrap.dedent(f"""
        import os, sys, sysconfig
        from sisyphus import gs, tk
        gs.FFMPEG_BINARY = "/pinned/by/the/test/ffmpeg"
        gs.W2VU_PYTHON = {DUMMY_W2VU_PYTHON!r}
        tk.register_output = lambda *a, **k: None
        from {PKG}.config import w2vu2
        w2vu2.py()
        allowed_prefixes = ("i6_core", "i6_experiments.common", "returnn", "sisyphus", {PKG!r})
        parents = set()
        for p in allowed_prefixes:
            parts = p.split(".")
            parents.update(".".join(parts[:i]) for i in range(1, len(parts)))
        # Classify by where the code lives: legacy RETURNN aliases (``Config``, ``TFUtil``, ...) and
        # similar re-exports carry foreign names but files inside an allowed package.
        import i6_core, i6_experiments.common, returnn, sisyphus, {PKG} as pkg
        roots = {{os.path.realpath(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib", "purelib", "platlib")}}
        for m in (i6_core, i6_experiments.common, returnn, sisyphus, pkg):
            roots.update(os.path.realpath(d) for d in m.__path__)
        roots = tuple(r + os.sep for r in roots)
        bad = []
        for name, mod in sorted(sys.modules.items()):
            if mod is None or name in parents:
                continue
            f = getattr(mod, "__file__", None)
            if not f or not os.path.isabs(f) or not os.path.exists(f):
                continue  # builtin / extension-internal / synthetic module without code on disk
            f = os.path.realpath(f)
            if f.startswith(roots) or os.sep + "site-packages" + os.sep in f:
                continue
            bad.append((name, f))
        print("BAD", bad)
    """)
    proc = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("BAD ")][-1]
    assert line == "BAD []", line
