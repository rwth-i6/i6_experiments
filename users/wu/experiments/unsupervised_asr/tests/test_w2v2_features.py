"""CPU tests of w2v2.features / w2v2.forward without the wav2vec2 weights: the seq-order job on a tiny
ogg zip, the RETURNN config, the statistics helpers and the feature callback."""

from __future__ import annotations

import os
import pickle
import types

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import features as F
from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import forward as FW

from test_data_ogg_zip import UTTS, make_zip  # sibling test module (pytest prepends this dir)


def run_seq_order(tmp_path, **kw):
    job = F.L15ForwardSeqOrderJob(**kw)
    job.out_seq_list = tk.Path(str(tmp_path / "seq_list.txt"))
    job.out_seq_order = tk.Path(str(tmp_path / "seq_order.py"))
    job.out_stats = tk.Path(str(tmp_path / "stats.txt"))
    job.run()
    seq_list = open(tmp_path / "seq_list.txt").read().split()
    order = eval(open(tmp_path / "seq_order.py").read())
    return seq_list, order


def test_seq_order_job(tmp_path):
    path, _ = make_zip(tmp_path)
    z = tk.Path(path)
    full = lambda u: f"dev-other/{u}/{u}"  # noqa: E731
    seq_list, order = run_seq_order(tmp_path, ogg_zip=z, shard=(0, 1))
    assert seq_list == [full(u) for u in sorted(UTTS)] and order == {t: i for i, t in enumerate(seq_list)}
    seq_list, _ = run_seq_order(tmp_path, ogg_zip=z, shard=(1, 2))
    assert seq_list == [full(sorted(UTTS)[1])]
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join([UTTS[2], UTTS[0]]) + "\n")
    seq_list, order = run_seq_order(tmp_path, ogg_zip=z, ids=tk.Path(str(ids)), order="given")
    assert seq_list == [full(UTTS[2]), full(UTTS[0])] and order[full(UTTS[2])] == 0
    ids.write_text("nope-1-0000\n")
    with pytest.raises(AssertionError, match="not in the zip"):
        run_seq_order(tmp_path, ogg_zip=z, ids=tk.Path(str(ids)))
    for bad in ({}, {"shard": (0, 1), "ids": tk.Path(str(ids))}, {"shard": (2, 2)}, {"shard": (0, 1), "order": "x"}):
        with pytest.raises(ValueError):
            F.L15ForwardSeqOrderJob(ogg_zip=z, **bad)


def test_forward_config(tmp_path):
    cfg = F.build_l15_forward_config(ogg_zip=tk.Path("/x/out.ogg.zip"), seq_list=tk.Path("/x/l.txt"),
                                     seq_order=tk.Path("/x/o.py"), hf_model_dir=tk.Path("/x/m"), expected_num_seqs=3)
    data = cfg.config["forward_data"]
    assert data["class"] == "MetaDataset" and data["data_map"] == {FW.AUDIO_KEY: ("zip_dataset", "data")}
    zd = data["datasets"]["zip_dataset"]
    assert zd["class"] == "OggZipDataset" and zd["seq_ordering"] == "sorted"
    assert zd["audio"] == {"features": "raw", "peak_normalization": False, "preemphasis": None}
    assert cfg.config["max_seqs"] == 1
    cfg.black_formatting, cfg._black_path = False, None
    out = str(tmp_path / "returnn.config")
    cfg.write(out)
    text = open(out).read()
    assert "w2v2.forward\", fromlist=[\"get_model\"]" in text and "'expected_num_seqs': 3" in text


def test_stats_helpers():
    rng = np.random.RandomState(0)
    xs = [rng.randn(n, 6).astype(np.float32) for n in (3, 7, 1)]
    rs = FW.RunningStats(6)
    for x in xs:
        rs.add(x)
    allx = np.concatenate(xs).astype(np.float64)
    np.testing.assert_allclose(rs.stats(), np.stack([allx.mean(0), allx.std(0)]), rtol=1e-5, atol=1e-6)
    p = FW.per_utt_stats(xs[2])
    assert p.dtype == np.float32 and p.shape == (2, 6) and np.all(p[1] == np.float32(1e-5))


def test_callback_writes_bare_ids(tmp_path):
    pytest.importorskip("returnn")
    import h5py

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cb = FW.L15FeatureCallback(feature_dim=4, expected_num_seqs=2)
        cb.init()
        rng = np.random.RandomState(0)
        xs = {"dev-other/1-2-0003/1-2-0003": rng.randn(5, 4).astype(np.float32) * 100,
              "dev-other/1-2-0001/1-2-0001": rng.randn(3, 4).astype(np.float32)}
        for tag, x in xs.items():
            cb.process_seq(seq_tag=tag, outputs={"states": types.SimpleNamespace(raw_tensor=x)})
        cb.finish()
        with h5py.File("feats.hdf", "r") as fh:
            assert [t.decode() for t in fh["seqTags"][:]] == ["1-2-0003", "1-2-0001"]
            assert fh["inputs"].dtype == np.float16
            np.testing.assert_array_equal(fh["inputs"][:5], xs["dev-other/1-2-0003/1-2-0003"].astype(np.float16))
        per = pickle.load(open("perutt_stats.pkl", "rb"))
        np.testing.assert_array_equal(per["1-2-0001"], FW.per_utt_stats(xs["dev-other/1-2-0001/1-2-0001"]))
        assert np.load("global_stats.npy").shape == (2, 4)
        with pytest.raises(ValueError):  # a bare id is not a segment name
            cb.process_seq(seq_tag="1-2-0001", outputs={"states": types.SimpleNamespace(raw_tensor=xs["dev-other/1-2-0001/1-2-0001"])})
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------------------------------
# T3.2 (test plan 2026-09-24, S11): the layer tap of a tiny random wav2vec2 (stable layer norm, 4
# layers, hidden 16) saved to tmp_path -- no download.  Dropout and layerdrop are 0 and the saved
# config asks for heavy time masking, so the only thing that can move a train-mode forward is the
# SpecAugment mask ``_build_l15_module`` switches on (forward.py ``apply_spec_augment = True``).
# ---------------------------------------------------------------------------------------------------
def _conv_out_len(n):
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        n = (n - k) // s + 1
    return n


@pytest.fixture(scope="module")
def tiny_w2v2_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=4, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True, feat_extract_norm="layer",
        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
        final_dropout=0.0, layerdrop=0.0,
        apply_spec_augment=False, mask_time_prob=0.5, mask_time_length=2, mask_feature_prob=0.0,
    )
    torch.manual_seed(0)
    model = transformers.Wav2Vec2Model(cfg)
    d = tmp_path_factory.mktemp("tiny_w2v2")
    model.save_pretrained(str(d))
    return str(d)


def _wavs():
    import torch

    g = torch.Generator().manual_seed(0)
    lens = torch.tensor([16000, 12345])
    wav = torch.zeros(2, int(lens.max()))
    for b, n in enumerate(lens.tolist()):
        wav[b, :n] = 0.3 * torch.randn(n, generator=g) + 0.05  # nonzero mean, non-unit variance
    return wav, lens


def test_l15_tap_is_hidden_state_k(tiny_w2v2_dir):
    """``_build_l15_module(encoder_layer=2)`` returns hidden_states[2] of the FULL model (the dropped
    blocks and the final stable layer norm never reach it), and never applies SpecAugment in eval."""
    import torch
    from transformers import Wav2Vec2Model

    module = FW._build_l15_module(hf_model_dir=tiny_w2v2_dir, encoder_layer=2)
    assert not module.training and not module.model.training
    assert len(module.model.encoder.layers) == 3 and module.model.config.apply_spec_augment is True
    assert not any(p.requires_grad for p in module.parameters())

    ref = Wav2Vec2Model.from_pretrained(tiny_w2v2_dir)
    ref.config.apply_spec_augment = False
    ref.eval()
    assert ref.config.num_hidden_layers == 4

    wav, lens = _wavs()
    with torch.no_grad():
        states, out_lens = module(wav, lens)
        states2, _ = module(wav, lens)
        normed = module._zero_mean_unit_var(wav, lens)
        att = (torch.arange(wav.shape[1])[None, :] < lens[:, None]).long()
        ref_out = ref(normed, attention_mask=att, output_hidden_states=True)
    assert len(ref_out.hidden_states) == 5
    torch.testing.assert_close(states, states2, rtol=0, atol=0)  # two eval forwards: identical
    torch.testing.assert_close(states, ref_out.hidden_states[2], rtol=0, atol=1e-6)
    # the tap is not the last block's output, nor the final-normed output
    assert (states - ref_out.hidden_states[3]).abs().max() > 1e-3
    assert out_lens.tolist() == [_conv_out_len(n) for n in lens.tolist()]
    assert out_lens.dtype == torch.long and states.shape[:2] == (2, int(out_lens.max()))
    assert states.shape[2] == 16

    # S11 power check: in train mode the SAME module masks (dropout/layerdrop are 0 in the config),
    # so eval() is what keeps SpecAugment off the frozen tap
    module.train()
    assert not module.model.feature_extractor.training
    torch.manual_seed(0)
    with torch.no_grad():
        states_train, _ = module(wav, lens)
    assert (states_train - states).abs().max() > 1e-3
    module.eval()


def test_l15_encoder_layer_range(tiny_w2v2_dir):
    with pytest.raises(ValueError):
        FW._build_l15_module(hf_model_dir=tiny_w2v2_dir, encoder_layer=5)
    full = FW._build_l15_module(hf_model_dir=tiny_w2v2_dir, encoder_layer=4)
    assert len(full.model.encoder.layers) == 4


def test_zero_mean_unit_var_matches_feature_extractor(tiny_w2v2_dir):
    """``_zero_mean_unit_var`` == ``Wav2Vec2FeatureExtractor(do_normalize=True)`` per utterance over
    the valid samples (padding comes out as 0.0, the extractor's padding value)."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    module = FW._build_l15_module(hf_model_dir=tiny_w2v2_dir, encoder_layer=2)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    wav, lens = _wavs()
    ours = module._zero_mean_unit_var(wav, lens).numpy()
    for b, n in enumerate(lens.tolist()):
        single = fe(wav[b, :n].numpy(), sampling_rate=16000, return_tensors="np")["input_values"][0]
        np.testing.assert_allclose(ours[b, :n], single, rtol=1e-5, atol=1e-5)
        assert np.all(ours[b, n:] == 0.0)
    batched = fe([wav[b, :n].numpy() for b, n in enumerate(lens.tolist())], sampling_rate=16000,
                 padding=True, return_tensors="np")["input_values"]
    np.testing.assert_allclose(ours, batched, rtol=1e-5, atol=1e-5)
