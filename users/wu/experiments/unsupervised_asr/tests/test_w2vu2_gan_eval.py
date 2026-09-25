"""Tests of the §1c wav2vec-U 2.0 generator forward in RETURNN (``model.w2vu2_generator``,
``analysis.w2vu2_gan_eval``).

Fast tests use tiny synthetic inputs.  The ``artefact`` tests read the banked source outputs (the s0
GAN checkpoint ``FairseqW2vu2TrainJob.HOb2GgtYT7Bc``, its text ``dict.txt``, the dev gold, the PER
record ``W2vu2PerEvalJob.ptwMk3TuPPYb`` and the pseudo-labels ``GanPseudoLabelJob.xjn6QnNqwEEH``)
and the VAD-trimmed L15 HDFs; they are skipped unless ``SAE_ARTEFACT_DIR`` is set (``conftest.py``)
and skip when a banked path is absent.  The fairseq reference forward runs in the source's ``w2vu``
python in a subprocess; nothing in the package imports fairseq.

Run (CPU, login node)::

    PYTHONPATH=<recipe>:<recipe>/returnn:<sisyphus> SAE_ARTEFACT_DIR=1 \\
        python -m pytest -p no:cacheprovider tests/test_w2vu2_gan_eval.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import w2vu2_gan_eval as G
from i6_experiments.users.wu.experiments.unsupervised_asr.model import w2vu2_generator as M
from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer import ConvRecognizer
from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer_only import get_model
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import PHONES

# the generator vocabulary of the source's text side: fairseq specials, <SIL>, then 39 phones
VOCAB44 = list(M.FAIRSEQ_SPECIAL_SYMBOLS) + ["<SIL>"] + [p for p in PHONES if p != "SIL"]
SMALL_NET = dict(M.W2VU2_NET_ARGS, in_dim=16)


def _outputs(job, tmp, names):
    for attr, fn in names.items():
        setattr(job, attr, tk.Path(os.path.join(tmp, fn)))


def _returnn_root() -> str:
    import returnn

    return os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__)))  # the dir holding rnn.py


def _returnn_hdf(path, rows, dim, dtype):
    """``{tag: [T, dim]}`` through RETURNN's own writer (what HDFDataset reads)."""
    from returnn.datasets.hdf import SimpleHDFWriter

    writer = SimpleHDFWriter(filename=path, dim=dim, ndim=2)
    for tag, x in rows.items():
        x = np.asarray(x, dtype=dtype)
        writer.insert_batch(x[None], [x.shape[0]], [tag])
    writer.close()
    return tk.Path(path)


def _eval_bn_model(net_args, seed=0):
    """A ConvRecognizer whose eval forward differs from its train forward (non-default BN stats)."""
    torch.manual_seed(seed)
    model = ConvRecognizer(**net_args)
    g = torch.Generator().manual_seed(seed + 1)
    d = net_args["in_dim"]
    with torch.no_grad():
        model.bn.running_mean.copy_(torch.randn(d, generator=g))
        model.bn.running_var.copy_(torch.rand(d, generator=g) + 0.5)
        model.bn.weight.normal_(generator=g)
        model.conv.weight.mul_(20.0)
    return model.eval()


def _raw_conv_out(model, x, lens):
    """The output conv's output ``[B, T_out, V]`` at the model's own dtype (no cast)."""
    got = []
    h = model.conv.register_forward_hook(lambda _m, _i, o: got.append(o))
    try:
        with torch.no_grad():
            model(x, lens)
    finally:
        h.remove()
    return got[0].transpose(1, 2)


def _run_forward_job(job, tmp, monkeypatch, *, threads=None, batch_size_frames=None):
    """``ReturnnForwardJobV2.create_files`` + ``run`` in-process in ``tmp`` (the job's own code path)."""
    names = {"out_returnn_config_file": "returnn.config"}
    for k in job.out_files:
        job.out_files[k] = tk.Path(os.path.join(tmp, k))
    _outputs(job, tmp, names)
    if batch_size_frames:
        job.returnn_config.config["batch_size"] = int(batch_size_frames)
    if threads:
        job.rqmt["cpu"] = int(threads)
    monkeypatch.chdir(tmp)
    job.create_files()
    job.run()
    return {k: v.get_path() for k, v in job.out_files.items()}


# =================================================================================================
# fast unit tests
# =================================================================================================
def test_greedy_decode_convention():
    """argmax (first index on ties), collapse repeats, drop <SIL>, keep fairseq specials."""
    V = VOCAB44
    sil, ah, n, unk = V.index("<SIL>"), V.index("AH"), V.index("N"), V.index("<unk>")
    path = [sil, ah, ah, sil, ah, n, n, unk, sil, sil]
    x = np.full((len(path), len(V)), -5.0, dtype=np.float32)
    x[np.arange(len(path)), path] = 1.0
    assert M.greedy_decode(x, V) == ["AH", "AH", "N", "<unk>"]
    tie = np.zeros((1, len(V)), dtype=np.float32)  # all equal -> index 0 (<s>)
    assert M.greedy_decode(tie, V) == ["<s>"]
    x_sil = np.full((3, len(V)), -1.0, dtype=np.float32)
    x_sil[:, sil] = 0.0
    assert M.greedy_decode(x_sil, V) == []


def test_read_fairseq_dict(tmp_path):
    p = tmp_path / "dict.txt"
    p.write_text("<SIL> 10\nAH 5\nN 3\n")
    assert G.read_fairseq_dict(str(p)) == ["<s>", "<pad>", "</s>", "<unk>", "<SIL>", "AH", "N"]
    for bad in ("AH 5 #fairseq:overwrite\n", "AH 5\nAH 3\n", "AH\n", "AH x\n"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            G.read_fairseq_dict(str(p))


def test_generator_logits_are_the_pre_softmax_output_and_batch_independent():
    model = _eval_bn_model(SMALL_NET)
    rng = np.random.RandomState(0)
    lens = [7, 20, 1, 13]
    xs = [rng.randn(t, 16).astype(np.float16) for t in lens]
    batch = np.zeros((len(xs), max(lens), 16), dtype=np.float16)
    for b, x in enumerate(xs):
        batch[b, : len(x)] = x
        batch[b, len(x):] = 100.0  # garbage in the padding must not leak
    feats, L = torch.from_numpy(batch), torch.tensor(lens)
    with torch.no_grad():
        logits = M.generator_logits(model, feats, L)
        logp = model(feats, L)
    assert logits.dtype == torch.float32 and logits.shape == (4, 7, 44)
    torch.testing.assert_close(torch.log_softmax(logits, -1), logp, rtol=0, atol=1e-5)
    out_lens = model.output_lengths(L).tolist()
    assert out_lens == [3, 7, 1, 5]
    for b, x in enumerate(xs):
        with torch.no_grad():
            single = M.generator_logits(model, torch.from_numpy(x)[None], torch.tensor([len(x)]))[0]
        torch.testing.assert_close(logits[b, : out_lens[b]], single, rtol=0, atol=1e-5)


def _fake_fairseq_checkpoint(path, model, cfg_overrides=None):
    sd = model.state_dict()
    fs = {}
    for k, v in sd.items():
        fs["generator." + (k.replace("conv.", "proj.1.", 1) if k.startswith("conv.") else k)] = v.clone()
    fs["discriminator.net.0.weight"] = torch.zeros(3, 3)
    fs["decoder.weight"] = torch.zeros(2)
    cfg = {"_name": "wav2vec_u", "input_dim": model.in_dim, "generator_kernel": 9, "generator_stride": 3,
           "generator_dilation": 1, "generator_pad": -1, "generator_bias": False, "generator_dropout": 0.1,
           "generator_batch_norm": 30, "generator_residual": True, "segmentation": {"type": "JOIN"}}
    cfg.update(cfg_overrides or {})
    torch.save({"args": None, "cfg": {"model": cfg, "task": {"_name": "unpaired_audio_text"}}, "model": fs,
                "optimizer_history": [{"num_updates": 1234}],
                "extra_state": {"train_iterator": {"epoch": 7}, "best": 1.5, "val_loss": 1.5}}, path)


def test_checkpoint_converter_on_a_synthetic_fairseq_checkpoint(tmp_path):
    tmp = str(tmp_path)
    model = _eval_bn_model(SMALL_NET, seed=3)
    _fake_fairseq_checkpoint(os.path.join(tmp, "fs.pt"), model)
    with open(os.path.join(tmp, "dict.txt"), "w") as fh:
        fh.writelines(f"{s} {100 - k}\n" for k, s in enumerate(VOCAB44[4:]))

    job = G.W2vu2GeneratorCheckpointJob(fairseq_checkpoint=tk.Path(os.path.join(tmp, "fs.pt")),
                                        text_dict=tk.Path(os.path.join(tmp, "dict.txt")), net_args=SMALL_NET)
    _outputs(job, tmp, {"out_checkpoint": "generator.pt", "out_vocab": "vocab.json", "out_stats": "stats.txt"})
    job.run()
    ck = torch.load(os.path.join(tmp, "generator.pt"), weights_only=True)
    assert (ck["epoch"], ck["step"]) == (7, 1234)
    assert set(ck["model"]) == set(model.state_dict())
    for k, v in model.state_dict().items():
        assert torch.equal(ck["model"][k], v), k
    assert json.load(open(os.path.join(tmp, "vocab.json"))) == VOCAB44

    # a config the forward cannot reproduce is refused
    _fake_fairseq_checkpoint(os.path.join(tmp, "fs_bad.pt"), model, {"generator_stride": 2})
    job.fairseq_checkpoint = tk.Path(os.path.join(tmp, "fs_bad.pt"))
    with pytest.raises(ValueError, match="generator_stride"):
        job.run()
    # a pickled foreign class is refused by the restricted unpickler (no arbitrary imports)
    import argparse

    torch.save({"args": argparse.Namespace(a=1), "cfg": {}, "model": {}}, os.path.join(tmp, "fs_obj.pt"))
    job.fairseq_checkpoint = tk.Path(os.path.join(tmp, "fs_obj.pt"))
    with pytest.raises(Exception, match="(?i)weights_only|unsupported global"):
        job.run()


def test_per_job_on_toy_decodes(tmp_path):
    tmp = str(tmp_path)
    gold = {"a": ["AH", "N", "T"], "b": ["S"], "c": ["AH"]}
    hyps = {"a": ["AH", "T"], "b": ["S", "S"], "c": ["AH"]}
    json.dump({"dev-other": gold}, open(os.path.join(tmp, "gold.json"), "w"))
    json.dump(hyps, open(os.path.join(tmp, "hyps.json"), "w"))
    job = G.W2vu2GanPerJob(hyps=tk.Path(os.path.join(tmp, "hyps.json")),
                           gold=tk.Path(os.path.join(tmp, "gold.json")), split="dev-other")
    _outputs(job, tmp, {"out_per": "per.json", "out_report": "per.txt"})
    job.run()
    rec = json.load(open(os.path.join(tmp, "per.json")))
    assert (rec["errors"], rec["del"], rec["ins"], rec["reference_phones"]) == (2, 1, 1, 5)
    assert rec["per"] == pytest.approx(2 / 5)
    json.dump({"a": ["AH"]}, open(os.path.join(tmp, "hyps.json"), "w"))
    with pytest.raises(AssertionError):
        job.run()


def test_pseudo_label_job_format_feeds_phone_targets(tmp_path):
    from i6_experiments.users.wu.experiments.unsupervised_asr.data.gold import PhoneTargetHdfJob

    tmp = str(tmp_path)
    json.dump({"u2": ["AH", "N"], "u1": []}, open(os.path.join(tmp, "h0.json"), "w"))
    json.dump({"u3": ["ZH"]}, open(os.path.join(tmp, "h1.json"), "w"))
    job = G.W2vu2GanPseudoLabelJob(hyps=[tk.Path(os.path.join(tmp, f"h{k}.json")) for k in (0, 1)],
                                   expected_num_seqs=3)
    _outputs(job, tmp, {"out_labels": "labels.json"})
    job.run()
    rec = json.load(open(os.path.join(tmp, "labels.json")))
    assert rec == {"labels": {"u1": "", "u2": "AH N", "u3": "ZH"}, "utts": 3, "empty": 1}
    tj = PhoneTargetHdfJob(labels_json=tk.Path(os.path.join(tmp, "labels.json")), label_key="labels")
    _outputs(tj, tmp, {"out_hdf": "targets.hdf", "out_stats": "targets.stats.txt"})
    tj.run()
    with h5py.File(os.path.join(tmp, "targets.hdf"), "r") as fh:
        assert [t.decode() for t in fh["seqTags"][:]] == ["u1", "u2", "u3"]
    json.dump({"u4": ["AH"], "u2": ["N"]}, open(os.path.join(tmp, "h1.json"), "w"))
    with pytest.raises(AssertionError, match="decoded twice"):
        job.run()


# =================================================================================================
# (d) the ReturnnForwardJobV2 of the builder: config serializes, loads and runs in RETURNN on CPU
# =================================================================================================
def test_d_forward_job_runs_in_returnn_on_cpu(tmp_path, monkeypatch):
    """The job's own ``create_files`` + ``run`` (RETURNN ``rnn.py``, CPU) on 57 tiny utterances: every
    ``hyps.json`` entry = the direct ``greedy_decode`` of ``ConvRecognizer.eval()``'s raw logits."""
    tmp = str(tmp_path)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    model = _eval_bn_model(SMALL_NET, seed=5)
    ckpt = os.path.join(tmp, "generator.pt")
    torch.save({"model": model.state_dict(), "epoch": 1, "step": 0}, ckpt)
    json.dump(VOCAB44, open(os.path.join(tmp, "vocab.json"), "w"))
    rng = np.random.RandomState(4)
    feats = {f"{100 + k}-1-{k:04d}": rng.randn(int(rng.randint(1, 40)), 16).astype(np.float16) for k in range(57)}
    hdf = _returnn_hdf(os.path.join(tmp, "feats.hdf"), feats, 16, "float16")

    from i6_core.returnn.training import PtCheckpoint

    job = G.w2vu2_forward_job(name="test", checkpoint=PtCheckpoint(tk.Path(ckpt)),
                              vocab=tk.Path(os.path.join(tmp, "vocab.json")), feature_hdfs=[hdf],
                              expected_num_seqs=len(feats), net_args=SMALL_NET, device="cpu",
                              returnn_exe=tk.Path(sys.executable), returnn_root=tk.Path(_returnn_root()))
    assert job.device == "cpu" and set(job.out_files) == set(G.W2VU2_FORWARD_OUTPUT_FILES)
    out = _run_forward_job(job, tmp, monkeypatch, threads=2, batch_size_frames=300)
    cfg_text = open(os.path.join(tmp, "returnn.config")).read()
    assert "w2vu2_generator" in cfg_text and os.path.join(tmp, "vocab.json") in cfg_text
    hyps = json.load(open(out["hyps.json"]))
    assert set(hyps) == set(feats)
    for tag, x in feats.items():
        with torch.no_grad():
            lg = M.generator_logits(model, torch.from_numpy(x)[None], torch.tensor([len(x)]))[0].numpy()
        assert hyps[tag] == M.greedy_decode(lg, VOCAB44), tag
    assert open(out["hyps.stats.txt"]).read().count(f"utts = {len(feats)}") == 1


def test_d_default_forward_config_serializes(tmp_path):
    """The production-shaped config (default net args, GPU device) writes and its python part runs."""
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_core.returnn.training import PtCheckpoint

    tmp = str(tmp_path)
    cfg = G.build_w2vu2_forward_config(feature_hdfs=[tk.Path("/nonexistent/feats.dev-other.shard0.hdf")],
                                       vocab=tk.Path("/nonexistent/vocab.json"), expected_num_seqs=2864)
    cfg = ReturnnForwardJobV2.create_returnn_config(
        model_checkpoint=PtCheckpoint(tk.Path("/nonexistent/generator.pt")), returnn_config=cfg,
        log_verbosity=3, device="gpu")
    cfg.black_formatting = False
    cfg.write(os.path.join(tmp, "returnn.config"))
    ns = {}
    exec(compile(open(os.path.join(tmp, "returnn.config")).read(), "returnn.config", "exec"), ns)
    assert ns["extern_data"]["data"]["dim"] == 1024 and ns["forward_data"]["files"] == [
        "/nonexistent/feats.dev-other.shard0.hdf"]
    m = ns["get_model"](epoch=0, step=0)
    assert m.in_dim == 1024 and m.n_out == 44 and m.stride == 3
    assert ns["forward_step"] is M.w2vu2_forward_step


# =================================================================================================
# artefact tests against the banked source outputs: (a) equivalence, (b) PER, (c) pseudo-labels
# =================================================================================================
_W = "/e/project1/spell/wu24/2026-07-13_unsupervised/work"
_U = f"{_W}/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2"
BANKED = {
    "checkpoint": f"{_U}/gan/FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train/checkpoint_best.pt",
    "text_data": f"{_U}/text/FairseqPreprocessTextJob.bi2B89fES77z/output/text_data",
    "gan_data": f"{_U}/features/MergeW2vu2DataJob.wxBxCIaJpqS2/output/data",
    "per": f"{_U}/eval/W2vu2PerEvalJob.ptwMk3TuPPYb/output/per.json",
    "gold": f"{_U}/eval/GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json",
    "labels": f"{_U}/selftrain/GanPseudoLabelJob.xjn6QnNqwEEH/output/labels.json",
    "feats": f"{_W}/speech_llm/sae/emc/blankfree_data_jobs/BlankfreeVadHdfJob.SAjz8y1cT06g/output",
    "mfa": "/e/project1/spell/common_hf_home/hub/datasets--gilkeyio--librispeech-alignments/snapshots/"
           "0daa1eb43dda38ee6ce752e785555380e5628f5c",
    "w2vu_python": "/e/project1/spell/wu24/env/conda/envs/w2vu/bin/python",
    "w2vu_shim": "/e/project1/spell/wu24/env/conda/envs/w2vu/fairseq_shim",
}
N_DEV = {"dev-clean": 2703, "dev-other": 2864}
FORWARD_THREADS = 16
CPU_BATCH_FRAMES = 200_000


def _banked(*keys):
    for k in keys:
        if not os.path.exists(BANKED[k]):
            pytest.skip(f"banked {k} not found: {BANKED[k]}")
    return [BANKED[k] for k in keys]


def _hdf_rows(path, pick=None):
    with h5py.File(path, "r") as fh:
        tags = [t.decode() for t in fh["seqTags"][:]]
        lens = fh["seqLengths"][:, 0]
        off = np.concatenate([[0], np.cumsum(lens)])
        idx = range(len(tags)) if pick is None else pick(len(tags))
        return {tags[k]: fh["inputs"][off[k]:off[k + 1]] for k in idx}


def _convert(tmp):
    ck, td = _banked("checkpoint", "text_data")
    job = G.W2vu2GeneratorCheckpointJob(fairseq_checkpoint=tk.Path(ck), text_dict=tk.Path(f"{td}/dict.txt"))
    _outputs(job, tmp, {"out_checkpoint": "generator.pt", "out_vocab": "vocab.json", "out_stats": "convert.stats.txt"})
    job.run()
    return job


_FAIRSEQ_REF = r'''
import argparse, os, sys
import numpy as np, torch
ckpt, data, text_data, inp, out, dtype = sys.argv[1:7]
import fairseq
from fairseq import checkpoint_utils, utils
utils.import_user_module(argparse.Namespace(
    user_dir=os.path.join(os.path.dirname(fairseq.__file__), "examples", "wav2vec", "unsupervised")))
# the source's eval overrides (w2vu2/eval_per.py _load_model)
overrides = {"task": {"data": data, "text_data": text_data},
             "model": {"segmentation": {"type": "NONE"}, "no_softmax": True}}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt], arg_overrides=overrides)
model = models[0].to("cpu").eval()
if dtype == "float64":
    model = model.double()
x_all = np.load(inp)
res = {}
for tag in x_all.files:
    x = torch.from_numpy(np.asarray(x_all[tag], dtype=dtype)).unsqueeze(0)
    with torch.no_grad():
        r = model(x, padding_mask=torch.zeros(x.shape[:2], dtype=torch.bool), dense_x_only=True, segment=True)
    res[tag] = r["logits"][0].numpy()
np.savez(out, **res)
print("VOCAB " + " ".join(task.target_dictionary.symbols))
'''


def _fairseq_logits(tmp, feats, dtype):
    ck, td, data, py, shim = _banked("checkpoint", "text_data", "gan_data", "w2vu_python", "w2vu_shim")
    script, inp, out = (os.path.join(tmp, n) for n in ("fs_ref.py", "inp.npz", f"ref_{dtype}.npz"))
    open(script, "w").write(_FAIRSEQ_REF)
    np.savez(inp, **feats)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = shim  # fairseq's `examples` as a top-level package, as the source's jobs ran
    run = subprocess.run([py, script, ck, data, td, inp, out, dtype], env=env, cwd=tmp, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:]
    vocab = [ln for ln in run.stdout.splitlines() if ln.startswith("VOCAB ")][-1].split()[1:]
    ref = np.load(out)
    return {t: ref[t] for t in ref.files}, vocab


@pytest.mark.artefact
@pytest.mark.slow
def test_a_generator_matches_fairseq_forward(tmp_path, capsys):
    """(a) 20 dev-other utterances (every 143rd of the port HDF): the converted s0 generator vs the
    fairseq model forward (w2vu python, CPU, eval, the source's eval overrides).

    Asserted: the vocabulary equals fairseq's ``target_dictionary``; in float64 on both sides the raw
    logits agree within 1e-5 (the implementation equivalence); in float32 (the production dtype) the
    per-frame argmax is identical on every frame.  The float32 max |diff| is printed, next to
    fairseq's own float32-vs-float64 deviation (accumulation-order rounding of the 1024-d in_proj and
    the 9 x 1024 conv, which differs between the two torch builds)."""
    tmp = str(tmp_path)
    (feats_dir,) = _banked("feats")
    job = _convert(tmp)
    vocab = json.load(open(job.out_vocab.get_path()))
    feats = _hdf_rows(f"{feats_dir}/feats.dev-other.shard0.hdf", lambda n: list(range(0, n, 143))[:20])
    assert len(feats) == 20
    ref32, fs_vocab = _fairseq_logits(tmp, feats, "float32")
    ref64, _ = _fairseq_logits(tmp, feats, "float64")
    assert vocab == fs_vocab

    sd = torch.load(job.out_checkpoint.get_path(), weights_only=True)["model"]
    m32 = get_model(epoch=0, step=0, **M.W2VU2_NET_ARGS)
    m32.load_state_dict(sd)
    m32.eval()
    m64 = get_model(epoch=0, step=0, **M.W2VU2_NET_ARGS)
    m64.load_state_dict(sd)
    m64.double().eval()
    d32 = d64 = fs_self = 0.0
    frames = 0
    for tag, x in feats.items():
        xt, lt = torch.from_numpy(x)[None], torch.tensor([len(x)])
        with torch.no_grad():
            a32 = M.generator_logits(m32, xt, lt)[0].numpy()
        a64 = _raw_conv_out(m64, xt, lt)[0].numpy()
        assert a64.dtype == np.float64 and a32.shape == ref32[tag].shape == a64.shape == ref64[tag].shape
        d32 = max(d32, float(np.abs(a32 - ref32[tag]).max()))
        d64 = max(d64, float(np.abs(a64 - ref64[tag]).max()))
        fs_self = max(fs_self, float(np.abs(ref32[tag] - ref64[tag]).max()))
        assert np.array_equal(a32.argmax(1), ref32[tag].argmax(1)), tag
        assert np.array_equal(a64.argmax(1), ref64[tag].argmax(1)), tag
        assert M.greedy_decode(a32, vocab) == M.greedy_decode(ref32[tag], fs_vocab), tag
        frames += a32.shape[0]
    with capsys.disabled():
        print(f"\n(a) 20 utts, {frames} output frames: float64 max|diff| = {d64:.3e}; float32 max|diff| = "
              f"{d32:.3e} (fairseq float32 vs its own float64: {fs_self:.3e}); max|logit| = "
              f"{max(float(np.abs(r).max()) for r in ref32.values()):.2f}; argmax identical on every frame")
    assert d64 <= 1e-5


def _forward_split(tmp, monkeypatch, conv_job, hdfs, n, name):
    from i6_core.returnn.training import PtCheckpoint

    sub = os.path.join(tmp, name)
    os.makedirs(sub)
    job = G.w2vu2_forward_job(name=name, checkpoint=PtCheckpoint(conv_job.out_checkpoint), vocab=conv_job.out_vocab,
                              feature_hdfs=[tk.Path(h) for h in hdfs], expected_num_seqs=n, device="cpu",
                              returnn_exe=tk.Path(sys.executable), returnn_root=tk.Path(_returnn_root()))
    return _run_forward_job(job, sub, monkeypatch, threads=FORWARD_THREADS, batch_size_frames=CPU_BATCH_FRAMES)


@pytest.mark.artefact
@pytest.mark.slow
def test_b_per_reproduces_banked(tmp_path, monkeypatch, capsys):
    """(b) the full chain through RETURNN on CPU (converter -> ReturnnForwardJobV2 -> W2vu2GanPerJob)
    over the port's dev HDFs, gold from the port's GoldPhonesJob on the pinned MFA snapshot:
    PER equals the banked record to its reported three digits (0.173 / 0.214); the error counts and
    the banked counts are printed."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.data.gold import GoldPhonesJob

    tmp = str(tmp_path)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    feats_dir, per_path, mfa, banked_gold = _banked("feats", "per", "mfa", "gold")
    banked = json.load(open(per_path))
    conv = _convert(tmp)
    gold_job = GoldPhonesJob(mfa_dir=tk.Path(mfa))
    _outputs(gold_job, tmp, {"out_gold": "gold.json"})
    gold_job.run()
    assert json.load(open(gold_job.out_gold.get_path())) == json.load(open(banked_gold))
    lines = []
    for split in ("dev-clean", "dev-other"):
        out = _forward_split(tmp, monkeypatch, conv, [f"{feats_dir}/feats.{split}.shard0.hdf"], N_DEV[split], split)
        pj = G.W2vu2GanPerJob(hyps=tk.Path(out["hyps.json"]), gold=gold_job.out_gold, split=split)
        _outputs(pj, os.path.join(tmp, split), {"out_per": "per.json", "out_report": "per.txt"})
        pj.run()
        rec = json.load(open(pj.out_per.get_path()))
        ref = banked[split]  # {"PER", "errors", "ref_phones", "utts"}
        lines.append(f"{split}: PER {rec['per']:.6f} errors {rec['errors']} ref {rec['reference_phones']} "
                     f"utts {rec['utterances']} | banked PER {ref['PER']:.6f} errors {ref['errors']} "
                     f"ref {ref['ref_phones']} utts {ref['utts']}")
        assert (rec["utterances"], rec["reference_phones"]) == (ref["utts"], ref["ref_phones"]), lines[-1]
        assert round(rec["per"], 3) == round(ref["PER"], 3), lines[-1]
    with capsys.disabled():
        print("\n(b) " + "\n(b) ".join(lines))


@pytest.mark.artefact
@pytest.mark.slow
def test_c_pseudo_labels_equal_banked(tmp_path, monkeypatch, capsys):
    """(c) the first 200 utterances of train shard 0 through the RETURNN forward (CPU) and
    W2vu2GanPseudoLabelJob: every label string equals the banked GanPseudoLabelJob.xjn6QnNqwEEH one;
    the labels load through PhoneTargetHdfJob."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.data.gold import PhoneTargetHdfJob

    tmp = str(tmp_path)
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    feats_dir, labels_path = _banked("feats", "labels")
    banked = json.load(open(labels_path))["labels"]
    conv = _convert(tmp)
    feats = _hdf_rows(f"{feats_dir}/feats.train.shard0.hdf", lambda n: range(200))
    hdf = _returnn_hdf(os.path.join(tmp, "train200.hdf"), feats, 1024, "float16")
    out = _forward_split(tmp, monkeypatch, conv, [hdf.get_path()], 200, "train200")
    lj = G.W2vu2GanPseudoLabelJob(hyps=[tk.Path(out["hyps.json"])], expected_num_seqs=200)
    _outputs(lj, tmp, {"out_labels": "labels.json"})
    lj.run()
    ours = json.load(open(lj.out_labels.get_path()))["labels"]
    assert set(ours) <= set(banked)
    diff = sorted(t for t in ours if ours[t] != banked[t])
    with capsys.disabled():
        print(f"\n(c) {len(ours) - len(diff)} / {len(ours)} label strings equal the banked ones"
              + (f"; differing: {diff[:5]}" if diff else ""))
    tj = PhoneTargetHdfJob(labels_json=lj.out_labels, label_key="labels")
    _outputs(tj, tmp, {"out_hdf": "targets.hdf", "out_stats": "targets.stats.txt"})
    tj.run()
    assert not diff
