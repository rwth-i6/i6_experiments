import os
import tempfile

import json

from i6_experiments.users.sheremeta.returnn.metrics import (
    count_params,
    parse_benchmark,
    parse_peak_mem_gb,
    parse_training_time,
)


_LR = (
    "{\n"
    "1: EpochData(learningRate=0.001, error={'train_loss_ce': 2.3, 'dev_loss_ce': 2.4, "
    "':meta:epoch_train_time_secs': 120, ':meta:global_train_step_end': 1000, ':meta:device': 'NVIDIA A100'}),\n"
    "2: EpochData(learningRate=0.0009, error={'dev_loss_ce': 2.1, "
    "':meta:epoch_train_time_secs': 118, ':meta:global_train_step_end': 2000, ':meta:device': 'NVIDIA A100'}),\n"
    "}\n"
)

_LOG = (
    "Memory usage (cuda:0): alloc cur 12.5GB, alloc peak 35.8GB, reserved cur 36.0GB, reserved peak 36.0GB\n"
    "ep 1 forward, mem_usage:cuda 2.34GB, 1.2 sec/step\n"
    "Memory usage (cuda:0): alloc cur 13.0GB, alloc peak 37.1GB, reserved cur 38.0GB, reserved peak 38.0GB\n"
)


def test_parse_training_time():
    t = parse_training_time(_LR)
    assert t["train_time_s"] == 238.0, t
    assert t["epochs_done"] == 2, t
    assert t["total_steps"] == 2000, t
    assert t["device"] == "NVIDIA A100", t


def test_parse_peak_mem():
    m = parse_peak_mem_gb(_LOG)
    assert abs(m["peak_alloc_gb"] - 37.1) < 1e-6, m
    assert abs(m["peak_reserved_gb"] - 38.0) < 1e-6, m


def test_parse_peak_mem_only_memusage():
    m = parse_peak_mem_gb("ep 1, mem_usage:cuda 5.0GB, step\n")
    assert abs(m["peak_usage_gb"] - 5.0) < 1e-6, m
    assert m["peak_alloc_gb"] is None and m["peak_reserved_gb"] is None, m


def test_memusage_outranks_the_start_up_counters_on_a_captured_row():
    """
    A captured row's start-up counters miss the graph pool entirely, so reporting them as the peak is
    an order of magnitude out. The per-step mem_usage already carries the pool and has to win.
    """
    log = (
        "Memory usage (cuda:0): alloc cur 8.4GB alloc peak 8.4GB reserved cur 8.4GB reserved peak 8.4GB\n"
        "ep 7 train, step 3, mem_usage:cuda:0 88.6GB, mem_graph_pool:cuda:0 76.98GB, 0.515 sec/step\n"
    )
    m = parse_peak_mem_gb(log)

    assert abs(m["peak_usage_gb"] - 88.6) < 1e-6, m
    assert abs(m["graph_pool_gb"] - 76.98) < 1e-6, m
    assert abs(m["peak_alloc_gb"] - 8.4) < 1e-6, m


def test_count_params():
    import torch

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "epoch.001.pt")
        torch.save({"model": {"encoder.a": torch.zeros(3, 4), "decoder.b": torch.zeros(5)}}, path)
        total, by_comp = count_params(path)
        assert total == 17, total
        assert by_comp == {"encoder": 12, "decoder": 5}, by_comp


def test_count_params_no_model_wrapper():
    import torch

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "flat.pt")
        torch.save({"encoder.w": torch.zeros(2, 2)}, path)
        total, by_comp = count_params(path)
        assert total == 4 and by_comp == {"encoder": 4}, (total, by_comp)


def test_parse_benchmark():
    txt = json.dumps({
        "rtf": 0.123456, "am_rtf": 0.08, "search_rtf": 0.0434, "utts_per_s": 12.3456,
        "infer_peak_reserved_GB": 4.5678, "infer_peak_alloc_GB": 3.1, "mem_driver_audio_s": 33.44,
    })
    b = parse_benchmark(txt)
    assert b["rtf"] == 0.1235, b
    assert b["am_rtf"] == 0.08 and b["search_rtf"] == 0.0434, b
    assert b["utts_per_s"] == 12.35, b
    assert b["infer_mem_GB"] == 4.57, b
    assert b["mem_at_s"] == 33.4, b


def test_parse_benchmark_alloc_fallback():
    b = parse_benchmark(json.dumps({"rtf": 0.2, "infer_peak_alloc_GB": 3.14}))
    assert b["infer_mem_GB"] == 3.14, b
    assert b["am_rtf"] is None, b


def test_best_wer_by_group():
    from i6_experiments.users.sheremeta.returnn.metrics import best_wer_by_group

    wers = {
        "best_cpkt": 6.5, "ep100": 5.1, "avg3_best_dev_ce": 5.0,
        "bestgreedy_beam8": 4.9, "bestgreedy_beam32": 4.8,
        "bestgreedy_jointctc_rescoring": 4.6, "bestgreedy_jointctc_prefix": 4.7,
        "ep100_beam32_jointctc_pad_ctc0.3": 4.65, "ep100_beam8_4gram_lm0.3": 4.5,
    }
    best = best_wer_by_group(wers)
    assert best["greedy"] == ("avg3_best_dev_ce", 5.0), best
    assert best["beam"] == ("bestgreedy_beam32", 4.8), best
    assert best["joint_ctc"] == ("bestgreedy_jointctc_rescoring", 4.6), best
    assert best["lm"] == ("ep100_beam8_4gram_lm0.3", 4.5), best
    assert best_wer_by_group({"ep005": 9.0}) == {"greedy": ("ep005", 9.0)}


def test_resolve_labels():
    from i6_experiments.users.sheremeta.returnn.checkpoint import resolve_names
    from i6_experiments.users.sheremeta.returnn.metrics import resolve_wer_labels

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    names = resolve_names(
        {"best_cpkt": _Var(94), "ep100": "ep100", "avg3_best_dev_ce": [_Var(94), _Var(95), 100], "bestgreedy": _Var("best_cpkt")}
    )
    assert names == {"best_cpkt": "ep094", "ep100": "ep100", "avg3_best_dev_ce": "avg3_ep094_ep095_ep100", "bestgreedy": "ep094"}, names

    wers = {"bestgreedy_beam8": 4.9, "bestgreedy_jointctc_rescoring": 4.6, "ep100": 5.1, "best_cpkt": 5.0, "ep005": 9.0}
    got = resolve_wer_labels(wers, names)
    assert got == {"ep094_beam8": 4.9, "ep094_jointctc_rescoring": 4.6, "ep100": 5.1, "ep094": 5.0, "ep005": 9.0}, got
    assert resolve_wer_labels({"best_cpkt": 5.1, "ep100": 5.1}, {"best_cpkt": "ep100"}) == {"ep100": 5.1}


def test_format_markdown():
    from i6_experiments.users.sheremeta.returnn.metrics import format_metrics_markdown

    m = {
        "name": "aed_ls_variant", "corpus": "librispeech", "wer": 5.5, "wer_label": "avg3_best_dev_ce",
        "wer_beam": 5.1, "wer_beam_label": "bestgreedy_beam32",
        "wer_joint_ctc": 4.9, "wer_joint_ctc_label": "bestgreedy_jointctc_prefix",
        "wer_all": {"bestgreedy_jointctc_prefix": 4.9, "bestgreedy_beam32": 5.1, "avg3_best_dev_ce": 5.5},
        "sub": 3.8, "del": 0.9, "ins": 0.8,
        "rtf": 0.052, "am_rtf": 0.031, "search_rtf": 0.021, "utts_per_s": 15.2,
        "infer_mem_GB": 2.47, "mem_at_s": 34.9, "params_M": 118.4,
        "params_by_component": {"encoder": 92.1, "decoder": 21.8}, "ckpt_MB": 1358.4,
        "train_time_h": 41.6, "gpu_h": 41.6, "peak_train_mem_GB": 71.8, "epochs_done": 100,
        "total_steps": 246800, "train_device": "NVIDIA H100", "frame_rate": 16.67, "delay_s": 0.0, "spm": "10k",
    }
    md = format_metrics_markdown(m)
    assert md.startswith("# aed_ls_variant (librispeech)"), md
    assert "| WER (greedy) | 5.5 % (avg3_best_dev_ce) |" in md
    assert "| WER (beam) | 5.1 % (bestgreedy_beam32) |" in md
    assert "| WER (joint CTC) | 4.9 % (bestgreedy_jointctc_prefix) |" in md
    assert md.index("| bestgreedy_jointctc_prefix | 4.9 |") < md.index("| avg3_best_dev_ce | 5.5 |"), md
    assert "| RTF (single-stream) | 0.052 |" in md
    assert "| Inference peak memory | 2.47 GB (at 34.9 s utterance) |" in md
    assert "| Epochs | 100 / 246,800 steps |" in md
    md2 = format_metrics_markdown({"name": "aed_loq", "corpus": "loquacious", "wer": 7.1, "params_M": 118.4})
    assert "Sub / Del / Ins" not in md2 and "RTF" not in md2, md2
    assert "WER (beam)" not in md2, md2
    assert "| WER (greedy) | 7.1 % |" in md2


def test_peak_mem_is_read_from_an_archived_training_log(tmp_path):
    """
    The cleanup packs a finished training's logs into finished.tar.gz and leaves log.run.1 empty, so a
    reader that only globs the live logs silently reports no peak memory for every finished row.
    """
    import tarfile

    from i6_experiments.users.sheremeta.returnn.throughput import rank_zero_log_text

    step = (
        "[default0]:ep 7 train, step 3, ce 1.2, num_seqs 26, mem_usage:cuda:0 88.6GB,"
        " mem_graph_pool:cuda:0 76.98GB, 0.515 sec/step\n"
    )
    other_rank = "[default1]:ep 7 train, step 3, ce 1.2, num_seqs 26, mem_usage:cuda:1 99.9GB, 0.515 sec/step\n"

    live = tmp_path / "live"
    live.mkdir()
    (live / "log.run.1").write_text(step + other_rank)
    assert parse_peak_mem_gb(rank_zero_log_text(str(live)))["peak_usage_gb"] == 88.6

    archived = tmp_path / "archived"
    archived.mkdir()
    (archived / "log.run.1").write_text("")
    with tarfile.open(archived / "finished.tar.gz", "w:gz") as tar:
        tar.add(live / "log.run.1", arcname="engine/job.run.1790230.1")

    assert parse_peak_mem_gb(rank_zero_log_text(str(archived)))["peak_usage_gb"] == 88.6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
    print("all ok")
