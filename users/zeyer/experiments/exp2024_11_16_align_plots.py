"""
Generate plots

Actually currently not really for Sisyphus, but standalone script...

To run many of the things here:

Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"

Then: python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import ... as f; f()"
For example:
    python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_11_16_align_plots import plot_all as f; f()"

Similar as :func:`i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores`.
"""

from __future__ import annotations
from typing import Optional, Union, Callable, Tuple, List
import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

from sisyphus import Path
from i6_experiments.users.schmitt.hdf import load_hdf_data

# See i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.visualize_grad_scores

# The funcs depend on these global vars,
# which are then being changed to iterate through different seqs / models.
# Not so nice, but I want to be pragmatic here.

# seq_list = Path(
#     "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
# )
# seq_list = open(seq_list.get_path()).read().splitlines()
seq_tag = "train-clean-100/103-1240-0000/103-1240-0000"

# See i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.py for names.
model_name_short = "base"
model_name_short_ext = ""
model_name = (
    "v6-relPosAttDef"
    "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm10k-bpeSample001"
)
model_title = "CTC baseline"
vocab = "spm10k"  # currently never changed
model_time_downsampling = 6  # currently never changed

# These are globals, not changed.
# See i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.py for names.
#
# * Forced align, selected blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# File: base-spm512-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 58.2/47.8, blank ratio 13.5%, sil ref ratio 18.0%,
# File: base-spm512-blankSep-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 58.7/50.5, blank ratio 16.8%, sil ref ratio 18.0%,
# File: base-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 68.2/52.0, blank ratio 21.8%, sil ref ratio 18.0%,
# File: blankSep-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 84.4/75.4, blank ratio 26.6%, sil ref ratio 18.0%,
# File: lpNormedGradC05_11P1-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 78.9/66.7, blank ratio 22.7%, sil ref ratio 18.0%,
# File: base-bpe10k-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 66.2/56.3, blank ratio 22.5%, sil ref ratio 18.0%,
# File: base-bpe10k-blankSep-ep-1-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0
# Dataset duration 9:36:16, TSE LR/center 72.5/65.3, blank ratio 26.7%, sil ref ratio 18.0%,
#
# * Grad align, selected -shift0-am1.0-prior1.0, except blankSep, where we take baselines
# File: base-spm512-shift0-am1.0-prior1.0
# Dataset duration 9:36:16, TSE LR/center 77.6/61.0, blank ratio 14.0%, sil ref ratio 18.0%,
# File: base-spm512-blankSep
# Dataset duration 9:36:16, TSE LR/center 75.9/60.5, blank ratio 13.6%, sil ref ratio 18.0%,
# File: base-shift0-am1.0-prior1.0
# Dataset duration 9:36:16, TSE LR/center 67.9/50.2, blank ratio 15.9%, sil ref ratio 18.0%,
# File: blankSep
# Dataset duration 9:36:16, TSE LR/center 72.5/55.7, blank ratio 15.3%, sil ref ratio 18.0%,
# File: lpNormedGradC05_11P1-shift0-am1.0-prior1.0
# Dataset duration 9:36:16, TSE LR/center 70.1/54.1, blank ratio 15.5%, sil ref ratio 18.0%,
# File: base-bpe10k-shift0-am1.0-prior1.0
# Dataset duration 9:36:16, TSE LR/center 71.3/54.7, blank ratio 16.2%, sil ref ratio 18.0%,
# File: base-bpe10k-blankSep
# Dataset duration 9:36:16, TSE LR/center 67.3/51.1, blank ratio 15.2%, sil ref ratio 18.0%,
#
models = [
    # model_title, model_name_short, model_name_short_ext, model_name, use_for_part
    (
        "CTC baseline, no blank penalty, no prior",
        "base",
        "",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001",
        {"model_probs", "grad_scores"},
    ),
    (
        "CTC baseline, with blank penalty and prior",
        "base",
        "-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001",
        {"model_probs"},
    ),
    (
        "CTC baseline, no blank penalty and with prior",
        "base",
        "-shift0-am1.0-prior1.0",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001",
        {"grad_scores"},
    ),
    (
        "CTC blank sep, no blank penalty, no prior",
        "blankSep",
        "-fix-blank_logit_shift0",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
        "-blankSep",
        {"model_probs"},
    ),
    (
        "CTC blank sep, no blank penalty, no prior",
        "blankSep",
        "",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
        "-blankSep",
        {"grad_scores"},
    ),
    (
        "CTC blank sep, with blank penalty and prior",
        "blankSep",
        "-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
        "-blankSep",
        {"model_probs"},
    ),
    (
        "CTC normed grad, with blank penalty and prior",
        "lpNormedGradC05_11P1",
        "-fix-blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0",
        "v6-relPosAttDef"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
        "-lpNormedGradC05_11P1",
        {"model_probs"},
    ),
]
grad_type_base = "blankStopGrad-inclBlankState-p0.1"
grad_type = (
    grad_type_base + "-smTimeTrue-bScorecalc-bScore_estflipped_after_softmax_over_time"
    "-non_blank_score_reducelog_mean_exp-bScore_flipped_percentile60-smLabelsTrue"
)
include_overlap_win_in_word_boundaries = False
out_prefix = "output/exp2024_11_16_grad_align/"


def plot_all():
    global seq_tag, model_title, model_name_short, model_name_short_ext, model_name
    print("seq_tag:", seq_tag)
    print("ref:", get_ref_words())
    plotter = Plotter(out_filename=out_prefix + seq_tag + "/combined.pdf")
    plot_audio_features(plotter=plotter)
    for model_title, model_name_short, model_name_short_ext, model_name, use_for_part in models:
        if use_for_part is None:
            use_for_part = {"model_probs", "grad_scores"}
        assert use_for_part.issubset({"model_probs", "grad_scores"})
        if "model_probs" in use_for_part:
            plot_model_probs(plotter=plotter)
        if "grad_scores" in use_for_part:
            plot_grad_scores(plotter=plotter)
    plotter.make()


def report_relevant():
    from glob import glob

    _cat_remove_prefixes = [
        "output/exp2024_09_16_grad_align/ctc_forced_align/",
        "output/exp2024_09_16_grad_align/ctc-grad-align/",
    ]
    _cat_remove_postfixes = [
        "-blankStopGrad-inclBlankState-p0.1"
        "-smTimeTrue-bScorecalc-bScore_estflipped_after_softmax_over_time"
        "-non_blank_score_reducelog_mean_exp-bScore_flipped_percentile60-smLabelsTrue"
        "/align-metrics_short_report.txt",
        "/align-metrics.short_report.txt",
    ]

    # noinspection PyShadowingNames
    def _cat(fn: str):
        fn_ = fn
        for prefix in _cat_remove_prefixes:
            if fn.startswith(prefix):
                fn = fn[len(prefix) :]
        for postfix in _cat_remove_postfixes:
            if fn.endswith(postfix):
                fn = fn[: -len(postfix)]
        print(f"File: {fn}")
        with open(fn_, "r") as f:
            print(f.read().strip())

    base_models = [
        "base-spm512",
        "base-spm512-blankSep",
        "base",
        "blankSep",
        "lpNormedGradC05_11P1",
        "base-bpe10k",
        "base-bpe10k-blankSep",
    ]

    forced_align_prefix = "output/exp2024_09_16_grad_align/ctc_forced_align/"
    # TSE align with forced align, multiple variants:
    # without
    # with prior and shift
    print(f"\n* Forced align, variants for shift/prior")
    forced_align_postfix = "/align-metrics.short_report.txt"
    for base in base_models:
        print(f"\n** Base models {base}")
        for fn in sorted(glob(f"{forced_align_prefix}{base}-ep-1-fix*{forced_align_postfix}")):
            _cat(fn)
        print("--")
    # -> I think always shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0?
    # is always reasonably good.
    forced_align_sel_set = "blank_logit_shift-10-ctc_prior_typestatic-ctc_am_scale1.0-ctc_prior_scale1.0"

    print(f"\n* Forced align, selected {forced_align_sel_set}")
    for base in base_models:
        _cat(f"{forced_align_prefix}{base}-ep-1-fix-{forced_align_sel_set}{forced_align_postfix}")
    print("--")

    # TSE blank sep wrong
    print(f"\n* Forced align, blankSep wrong vs fix")
    for base in base_models:
        if "blankSep" not in base:
            continue
        _cat(f"{forced_align_prefix}{base}-ep-1{forced_align_postfix}")
        _cat(f"{forced_align_prefix}{base}-ep-1-fix-blank_logit_shift0{forced_align_postfix}")
        _cat(f"{forced_align_prefix}{base}-ep-1-fix-{forced_align_sel_set}{forced_align_postfix}")
    print("--")

    # TSE align with grad align, multiple variants:
    grad_align_prefix = "output/exp2024_09_16_grad_align/ctc-grad-align/"
    # GradScore vs GradScoreExt
    # without prior
    # with prior

    grad_align_grad_score_variant_fn = (  # GradScore
        "-blankStopGrad-inclBlankState-p0.1-smTimeTrue-bScore-6/align-metrics_short_report.txt"
    )

    grad_align_grad_score_ext_variant_fn = (  # GradScoreExt
        "-blankStopGrad-inclBlankState-p0.1"
        "-smTimeTrue-bScorecalc-bScore_estflipped_after_softmax_over_time"
        "-non_blank_score_reducelog_mean_exp-bScore_flipped_percentile60-smLabelsTrue"
        "/align-metrics_short_report.txt"
    )

    print(f"\n* Grad align, variants for shift/prior")
    for base in base_models:
        print(f"\n** Base models {base}")
        _cat(f"{grad_align_prefix}{base}{grad_align_grad_score_ext_variant_fn}")
        for fn in sorted(glob(f"{grad_align_prefix}{base}-shift*-am*-prior*{grad_align_grad_score_ext_variant_fn}")):
            _cat(fn)
        print("--")
    # -> shift0, am1, prior1 seems best for most cases, except for all blankSep, where prior0 is better.
    sel = "-shift0-am1.0-prior1.0"
    print(f"\n* Grad align, selected {sel}, except blankSep, where we take baselines")
    for base in base_models:
        if "blankSep" in base:
            _cat(f"{grad_align_prefix}{base}{grad_align_grad_score_ext_variant_fn}")
        else:
            _cat(f"{grad_align_prefix}{base}{sel}{grad_align_grad_score_ext_variant_fn}")
    print("--")

    # TSE normed grad, with normed grad in grad align
    print("\n*** Grad align, effect of lpNormedGradUsed")
    for fn in sorted(glob(f"{grad_align_prefix}*lpNormedGradUsed*{grad_align_grad_score_ext_variant_fn}")):
        assert "lpNormedGradUsed-" in fn
        fn_ = fn.replace("lpNormedGradUsed-", "")
        _cat(fn_)
        _cat(fn)
        print("--")
    # -> almost no effect at all?


def get_audio_features() -> np.array:
    """
    :return: log mel filterbank features, [T, D].
    """
    feat_type = "log_mel_filterbank"
    # dim = 120  # D=120 here, but that's arbitrary, just what looks nice.
    dim = 80
    out_fn_npz = out_prefix + seq_tag + f"/audio_features_{feat_type}_{dim}.npz"
    if os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        return np.load(out_fn_npz)["audio_features"]

    from returnn.datasets.audio import OggZipDataset

    dataset = OggZipDataset(
        os.readlink("output/librispeech/dataset/train-clean-100"),
        targets=None,
        audio={"features": feat_type, "num_feature_filters": dim},
        # audio={"features": "mfcc", "num_feature_filters": 80},
    )
    dataset.init_seq_order(epoch=1, seq_list=[seq_tag])
    dataset.load_seqs(0, 1)
    audio_features = dataset.get_data(0, "data")  # [T, D]
    print(f"audio_features.shape: {audio_features.shape}")

    print("save to:", out_fn_npz)
    os.makedirs(os.path.dirname(out_fn_npz), exist_ok=True)
    np.savez(out_fn_npz, audio_features=audio_features)
    return audio_features


def get_audio_features_rf() -> np.array:
    """
    :return: log mel filterbank features, [T, D].
    """
    dim = 80
    out_fn_npz = out_prefix + seq_tag + f"/rf_audio_log_mel_filterbank_from_raw_{dim}.npz"
    if os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        return np.load(out_fn_npz)["audio_features"]

    from returnn.datasets.audio import OggZipDataset

    dataset = OggZipDataset(
        os.readlink("output/librispeech/dataset/train-clean-100"),
        targets=None,
        audio={"features": "raw", "num_feature_filters": 1},
    )
    dataset.init_seq_order(epoch=1, seq_list=[seq_tag])
    dataset.load_seqs(0, 1)
    samples_np = dataset.get_data(0, "data")  # [T, 1]
    print(f"samples_np.shape: {samples_np.shape}")
    assert samples_np.shape[1] == 1
    samples_np = samples_np[:, 0]  # [T]

    import torch
    import returnn.frontend as rf
    from returnn.tensor import Dim

    time_dim = Dim(rf.convert_to_tensor(torch.tensor(len(samples_np))), name="time")
    samples = rf.convert_to_tensor(torch.tensor(samples_np), dims=[time_dim])
    feat_dim = Dim(dim, name="feature")
    audio_features, feat_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        samples,
        in_spatial_dim=time_dim,
        out_dim=feat_dim,
        sampling_rate=16_000,
    )
    audio_features_np = audio_features.copy_compatible_to_dims_raw([feat_spatial_dim, feat_dim]).numpy()

    print("save to:", out_fn_npz)
    os.makedirs(os.path.dirname(out_fn_npz), exist_ok=True)
    np.savez(out_fn_npz, audio_features=audio_features_np)
    return audio_features_np


def get_grad_scores():
    out_fn_npz = (
        out_prefix
        + seq_tag
        + f"/visualize_grad_scores/"
        + f"{model_name_short}{model_name_short_ext}-{grad_type_base}/grads.npz"
    )

    if os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        data = np.load(out_fn_npz)
        return data["score_matrix"]

    score_matrix_hdf = Path(
        f"output/exp2024_09_16_grad_align/ctc-grad-align/"
        f"{model_name_short}{model_name_short_ext}-{grad_type_base}/input_grads.hdf"
    )
    print("load grad scores HDF:", score_matrix_hdf)
    score_matrix_data_dict = load_hdf_data(score_matrix_hdf, num_dims=2)
    basename_tags = {os.path.basename(tag): tag for tag in score_matrix_data_dict.keys()}

    seq_tag_ = seq_tag
    if seq_tag_ not in score_matrix_data_dict:
        if os.path.basename(seq_tag_) in basename_tags:
            seq_tag_ = basename_tags[os.path.basename(seq_tag_)]

    score_matrix = score_matrix_data_dict[seq_tag_]  # [S, T]
    print(f"load {score_matrix_hdf}: {seq_tag_}, shape {score_matrix.shape}")
    print(f"save to:", out_fn_npz)
    os.makedirs(os.path.dirname(out_fn_npz), exist_ok=True)
    np.savez(out_fn_npz, seq_tag=seq_tag_, score_matrix=score_matrix)
    return score_matrix


def get_ref_word_boundaries() -> List[Tuple[float, float, str]]:
    out_fn_pickle = out_prefix + seq_tag + f"/ref_word_boundaries_overlap{include_overlap_win_in_word_boundaries}.pkl"
    if os.path.exists(out_fn_pickle):
        print(f"Already exists: {out_fn_pickle}")
        return pickle.load(open(out_fn_pickle, "rb"))

    ref_words = get_ref_words()

    # adopted from i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align.CalcAlignmentMetrics
    from returnn.sprint.cache import open_file_archive

    ref_alignment_allophones = Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    ref_alignment_sprint_cache = Path(
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
    )
    features_sprint_cache = Path(  # for exact timings
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
    )

    def _cf(fn: Path) -> str:
        return fn.get_path()

    print("Loading ref alignment Sprint cache...")
    ref_align_sprint_cache = open_file_archive(_cf(ref_alignment_sprint_cache))
    print("Loading ref alignment allophones...")
    ref_align_sprint_cache.set_allophones(_cf(ref_alignment_allophones))
    allophones = ref_align_sprint_cache.get_allophones_list()

    print("Loading features Sprint cache...")
    features_sprint_cache = open_file_archive(_cf(features_sprint_cache))

    def _ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    key = seq_tag
    key_ref = "train-other-960/" + "/".join(seq_tag.split("/")[1:])

    print("seq tag:", key)
    feat_times, _ = features_sprint_cache.read(key_ref, typ="feat")  # list of (start_time: float, end_time: float) (s)
    # Note: Sprint does not add padding. The first feature frame covers (0., 0.025).
    print("feat times:", feat_times)
    ref_align = ref_align_sprint_cache.read(key_ref, typ="align")  # list of (time_index: int, allophone, state, weight)
    print("ref align:", ref_align)
    assert len(feat_times) == len(ref_align), f"feat len {len(feat_times)} vs ref align len {len(ref_align)}"
    print(f"  start time: {feat_times[0][0]} sec")
    print(f"  end time: {feat_times[-1][1]} sec")

    ref_start_time = feat_times[0][0]  # should be 0.0
    print("ref start time:", ref_start_time)
    duration_sec = feat_times[-1][1] - ref_start_time
    sampling_rate = 16_000
    len_samples = round(duration_sec * sampling_rate)  # 16 kHz
    print(f"  num samples: {len_samples} (rounded from {duration_sec * sampling_rate})")
    # RETURNN uses log mel filterbank features, 10ms frame shift, via stft (valid padding)
    window_len = 0.025  # 25 ms
    step_len = 0.010  # 10 ms
    window_num_frames = int(window_len * sampling_rate)
    step_num_frames = int(step_len * sampling_rate)
    len_feat = _ceil_div(len_samples - (window_num_frames - 1), step_num_frames)
    print(f"  num features (10ms): {len_feat} (window {window_num_frames} step {step_num_frames})")
    # ref_alignment_len_factor =
    # len_feat_downsampled = _ceil_div(len_feat, self.ref_alignment_len_factor)
    # print(f"  downsampled num features: {len_feat_downsampled} (factor {self.ref_alignment_len_factor})")

    ref_word_boundaries: List[Tuple[float, float]] = []
    cur_word_start_frame = None
    prev_allophone_idx = None
    ref_words_phones = []
    num_sil_frames_ref = 0
    for t, (t_, allophone_idx, hmm_state_idx, _) in enumerate(ref_align):
        assert t == t_
        if "[SILENCE]" in allophones[allophone_idx]:
            num_sil_frames_ref += 1
            continue
        if cur_word_start_frame is None:
            cur_word_start_frame = t  # new word starts here
            ref_words_phones.append([])
        if prev_allophone_idx != allophone_idx:
            ref_words_phones[-1].append(allophones[allophone_idx])
        if "@f" in allophones[allophone_idx] and (
            t == len(ref_align) - 1 or ref_align[t + 1][1] != allophone_idx or ref_align[t + 1][2] < hmm_state_idx
        ):
            # end of word
            start_time = feat_times[cur_word_start_frame][0] - ref_start_time
            end_time = feat_times[t][1] - ref_start_time
            # take center 10ms of the 25ms window
            # (or not, as we also don't do for the alignment)
            if not include_overlap_win_in_word_boundaries:
                start_time += (window_len - step_len) / 2
                end_time -= (window_len - step_len) / 2
            ref_word_boundaries.append((start_time, end_time))
            cur_word_start_frame = None
        prev_allophone_idx = allophone_idx
    assert cur_word_start_frame is None  # word should have ended
    assert len(ref_words_phones) == len(ref_word_boundaries) == len(ref_words)

    # If we would want the phones: " ".join(p[: p.index("{")] for p in phones)
    # But we use the words.
    ref_word_boundaries_ = [
        (float(start_time), float(end_time), word)
        for (start_time, end_time), word in zip(ref_word_boundaries, ref_words)
    ]

    print(f"  num words: {len(ref_word_boundaries)}")
    print(f"  num silence frames: {num_sil_frames_ref}")
    print("save to:", out_fn_pickle)
    os.makedirs(os.path.dirname(out_fn_pickle), exist_ok=True)
    pickle.dump(ref_word_boundaries_, open(out_fn_pickle, "wb"))

    return ref_word_boundaries_


def get_ref_words() -> List[str]:
    out_fn_pkl = out_prefix + seq_tag + "/ref_words.pkl"
    if os.path.exists(out_fn_pkl):
        print(f"Already exists: {out_fn_pkl}")
        return pickle.load(open(out_fn_pkl, "rb"))

    from returnn.datasets.audio import OggZipDataset

    dataset = OggZipDataset(
        os.readlink("output/librispeech/dataset/train-clean-100"), targets={"class": "Utf8ByteTargets"}, audio=None
    )
    dataset.init_seq_order(epoch=1, seq_list=[seq_tag])
    dataset.load_seqs(0, 1)
    ref_bytes = dataset.get_data(0, "classes")  # [T, D]
    print(f"ref_bytes.shape: {ref_bytes.shape}")
    print(f"ref_bytes: {ref_bytes!r}")

    ref_str = dataset.targets.get_seq_labels(ref_bytes)
    print(f"ref_str: {ref_str}")
    print(f"ref_str repr: {ref_str!r} type {type(ref_str).__name__}")

    # Split by space.
    # Lowercase all, as Librispeech does not have casing.
    ref_words = [w.lower() for w in ref_str.split()]
    print(f"ref_words: {ref_words}")

    print("save to:", out_fn_pkl)
    os.makedirs(os.path.dirname(out_fn_pkl), exist_ok=True)
    pickle.dump(ref_words, open(out_fn_pkl, "wb"))
    return ref_words


def get_ref_label_seq() -> List[Tuple[int, str]]:
    out_fn_pkl = out_prefix + seq_tag + f"/ref_label_seq_{vocab}.pkl"
    if os.path.exists(out_fn_pkl):
        print(f"Already exists: {out_fn_pkl}")
        return pickle.load(open(out_fn_pkl, "rb"))

    from returnn.config import Config
    from returnn.datasets.basic import init_dataset
    from returnn.datasets.audio import OggZipDataset

    config = Config()
    train_config_fn = f"alias/ctc/{model_name}/train/output/returnn.config"
    print("load RETURNN config:", train_config_fn)
    config.load_file(train_config_fn)

    # Take the eval dataset dict as base, to get targets without augmentation.
    ds_dict = config.typed_dict["eval_datasets"]["dev"].copy()
    if ds_dict["class"] == "MultiProcDataset":
        ds_dict = ds_dict["dataset"]
    assert ds_dict["class"] == "OggZipDataset", f"dataset dict: {ds_dict}"
    train_ds_dict = config.typed_dict["train"]
    if train_ds_dict["class"] == "MultiProcDataset":
        train_ds_dict = train_ds_dict["dataset"]
    assert train_ds_dict["class"] == "OggZipDataset", f"train dataset dict: {train_ds_dict}"
    ds_dict["path"] = train_ds_dict["path"]  # get train data
    del ds_dict["fixed_random_subset"]
    ds_dict["audio"] = None
    print("dataset dict:", ds_dict)

    ds = init_dataset(ds_dict)
    print("dataset:", ds)
    assert isinstance(ds, OggZipDataset)
    ds.init_seq_order(epoch=1, seq_list=[seq_tag])
    ds.load_seqs(0, 1)
    seq = ds.get_data(0, "classes")  # [T]
    print(f"seq.shape: {seq.shape}")
    print("seq:", seq)
    print("seq str:", ds.targets.get_seq_labels(seq))
    seq_labels = [ds.targets.labels[i] for i in seq]
    print("seq labels:", seq_labels)
    seq_ = list(zip(seq, seq_labels))

    print("save to:", out_fn_pkl)
    os.makedirs(os.path.dirname(out_fn_pkl), exist_ok=True)
    pickle.dump(seq_, open(out_fn_pkl, "wb"))
    return seq_


def get_model_log_prob_ref_label_seq_incl_blank_direct(*, force: bool = False) -> np.array:
    """
    :return: [T,S+1] log probs, S is the target length, first entry is blank (thus +1)
    """
    # We want to load the model, and then forward the seq, and get the log probs for the ref label seq.
    out_fn_npz = out_prefix + seq_tag + f"/model_log_probs_ref_label_seq_incl_blank_{model_name_short}.npz"
    if not force and os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        return np.load(out_fn_npz)["model_log_probs_ref_label_seq_incl_blank"]

    # To find the right model, we can reuse the RETURNN config from the input grads,
    # which has everything already well prepared (model def, forward dataset).
    input_grads_hdf = f"output/exp2024_09_16_grad_align/ctc-grad-align/{model_name_short}/input_grads.hdf"
    input_grads_hdf_ = os.readlink(input_grads_hdf)
    work_out_dir = os.path.dirname(input_grads_hdf_)
    returnn_config_fn = f"{work_out_dir}/returnn.config"
    assert work_out_dir.endswith("/output") and os.path.exists(returnn_config_fn)

    from returnn.util import BehaviorVersion
    from returnn.config import Config, global_config_ctx
    from returnn.torch.engine import Engine, ForwardCallbackIface
    from returnn.datasets.basic import init_dataset
    from returnn.tensor import batch_dim, Tensor, TensorDict
    import returnn.frontend as rf
    from .exp2024_04_23_baselines.ctc import Model

    config = Config()
    print("load RETURNN config:", returnn_config_fn)
    config.load_file(returnn_config_fn)

    # We could also more directly load the model, via get_model, then torch.load, etc.
    # But using the engine does all that work for us.

    def _forward_step(*, model: Model, extern_data: TensorDict, **_kwargs):
        print("forward_step", model, extern_data)
        print("Behavior version:", BehaviorVersion.get_if_set(), BehaviorVersion.get())

        default_input_key = config.typed_value("default_input")
        default_target_key = config.typed_value("target")
        data = extern_data[default_input_key]
        targets = extern_data[default_target_key]
        assert model.blank_idx == targets.sparse_dim.dimension  # blank idx at end. not implemented otherwise
        # Add blank as first ref label for plotting.
        targets_, (target_ext_spatial_dim,) = rf.pad(
            targets, axes=[targets.get_time_dim_tag()], padding=[(1, 0)], value=model.blank_idx
        )
        targets_.sparse_dim = model.wb_target_dim

        # Call: logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
        in_spatial_dim = data.get_time_dim_tag()

        if data.feature_dim and data.feature_dim.dimension == 1:
            data = rf.squeeze(data, axis=data.feature_dim)
        assert not data.feature_dim  # raw audio
        logits, enc, enc_spatial_dim = model(data, in_spatial_dim=in_spatial_dim)
        log_probs_wb = model.log_probs_wb_from_logits(logits)
        log_probs_ref_seq = rf.gather(log_probs_wb, indices=targets_, axis=model.wb_target_dim)  # [B,T,S+1]
        log_probs_ref_seq.mark_as_default_output(shape=[batch_dim, enc_spatial_dim, target_ext_spatial_dim])

    config.typed_dict["forward_step"] = _forward_step
    config.typed_dict["device"] = None  # allow any, also CPU
    del config.typed_dict["model_outputs"]  # not the same here
    # We specifically don't want multi-processing for the dataloader:
    # - It's unnecessary overhead here.
    # - It breaks our custom init_seq_order.
    config.typed_dict["torch_dataloader_opts"] = {"num_workers": 0}

    engine = Engine(config)
    with global_config_ctx(config):  # the forward get_model needs a global config
        engine.init_network_from_config(config)

    ds = init_dataset(config.typed_dict["forward_data"])
    ds.init_seq_order(epoch=1, seq_list=[seq_tag])

    out: Optional[Tensor] = None

    class _ForwardCallback(ForwardCallbackIface):
        seq_tag = seq_tag  # keep a ref

        # noinspection PyShadowingNames
        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            print("process_seq", seq_tag, outputs)
            assert self.seq_tag == seq_tag
            nonlocal out
            assert out is None  # not called multiple times
            out = outputs["output"]

    callback = _ForwardCallback()
    engine.forward_with_callback(dataset=ds, callback=callback, dataset_init_epoch=False)

    print("got output:", out, out.raw_tensor.shape)
    os.makedirs(os.path.dirname(out_fn_npz), exist_ok=True)
    np.savez(out_fn_npz, model_log_probs_ref_label_seq_incl_blank=out.raw_tensor)
    return out.raw_tensor  # [T,S+1]


def get_model_log_prob_ref_label_seq_incl_blank(*, force: bool = False) -> np.array:
    """
    :return: [T,S+1] log probs, S is the target length, first entry is blank (thus +1)
    """
    # We want to load the model, and then forward the seq, and get the log probs for the ref label seq.

    out_fn_npz = (
        out_prefix
        + seq_tag
        + f"/model_log_probs_ref_label_seq_incl_blank_{model_name_short}{model_name_short_ext}_via_hdf.npz"
    )
    if not force and os.path.exists(out_fn_npz):
        print(f"Already exists: {out_fn_npz}")
        return np.load(out_fn_npz)["model_log_probs_ref_label_seq_incl_blank"]

    from returnn.datasets.hdf import HDFDataset

    hdf_fn = (
        f"output/exp2024_09_16_grad_align/ctc_ref_log_probs/"
        f"{model_name_short}-ep-1{model_name_short_ext}/log_probs.hdf"
    )
    print("load HDF:", hdf_fn)
    dataset = HDFDataset([hdf_fn])
    dataset.initialize()
    dataset.init_seq_order(epoch=1, seq_list=[seq_tag])

    model_log_probs = dataset.get_data(0, "data")  # [T * (S+1)]
    sizes = dataset.get_data(0, "sizes")  # [2] (T, S+1)
    size_time, size_targets_ext = sizes
    assert size_time * size_targets_ext == model_log_probs.shape[0], f"sizes {sizes}, shape {model_log_probs.shape}"
    model_log_probs = model_log_probs.reshape(size_time, size_targets_ext)  # [T, S+1]
    print("got output:", model_log_probs.shape)
    os.makedirs(os.path.dirname(out_fn_npz), exist_ok=True)
    np.savez(out_fn_npz, model_log_probs_ref_label_seq_incl_blank=model_log_probs)
    return model_log_probs  # [T,S+1]


def get_word_boundaries_from_hdf_alignment(
    *, align_type: str = "probs_best_path", force: bool = False
) -> List[Tuple[float, float, str]]:
    out_fn_pickle = (
        out_prefix
        + seq_tag
        + f"/word_boundaries_{model_name_short}{model_name_short_ext}"
        + f"_{align_type}_overlap{include_overlap_win_in_word_boundaries}.pkl"
    )
    if not force and os.path.exists(out_fn_pickle):
        print(f"Already exists: {out_fn_pickle}")
        return pickle.load(open(out_fn_pickle, "rb"))

    # Copied and adapted from i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align.CalcAlignmentMetrics.

    ref_words = get_ref_words()

    print("align type:", align_type)
    if align_type == "probs_best_path":  # aka forced align, using the model probs
        alignment_label_topology = "ctc"
        ref_alignment_len_factor = model_time_downsampling
        hdf_fn = (
            f"output/exp2024_09_16_grad_align/ctc_forced_align/"
            f"{model_name_short}-ep-1{model_name_short_ext}/align.hdf"
        )
    elif align_type == "grad":
        alignment_label_topology = "explicit"
        ref_alignment_len_factor = 1
        hdf_fn = (
            f"output/exp2024_09_16_grad_align/ctc-grad-align/"
            f"{model_name_short}{model_name_short_ext}-{grad_type}/align.hdf"
        )
    else:
        raise ValueError(f"align_type {align_type!r} not supported")
    print("alignment label topology:", alignment_label_topology)
    print("ref alignment len factor:", ref_alignment_len_factor)
    print("hdf filename:", hdf_fn)
    assert os.path.exists(hdf_fn), f"alignment HDF not found: {hdf_fn}"

    from returnn.datasets.hdf import HDFDataset
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
    from returnn.datasets.util.vocabulary import Vocabulary

    vocabs = {
        "spm10k": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm10k").model_file).out_vocab, 10_240),
        "spm512": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm512").model_file).out_vocab, 512),
        "bpe10k": ("bpe", get_vocab_by_str("bpe10k").vocab, 10_025),
    }
    alignment_bpe_style, vocab_file, vocab_size = vocabs[vocab]
    alignment_blank_idx = vocab_size  # assumption...
    bpe_vocab = Vocabulary(vocab_file.get_path(), unknown_label=None)  # note: without blank

    alignments_ds = HDFDataset([hdf_fn])
    alignments_ds.initialize()
    alignments_ds.init_seq_order(epoch=1, seq_list=[seq_tag])

    alignment = alignments_ds.get_data(0, "data")  # [T]
    print("alignment:", alignment)
    print("alignment str:", " ".join("ε" if l == alignment_blank_idx else bpe_vocab.labels[l] for l in alignment))

    if align_type == "probs_best_path":
        align_score = alignments_ds.get_data(0, "scores")  # [1]
        print("align score:", align_score, "prob:", np.exp(align_score))

        ref_label_seq = [alignment_blank_idx] + [l for l, _ in get_ref_label_seq()]  # [S+1]
        model_log_probs = get_model_log_prob_ref_label_seq_incl_blank()  # [T,S+1]
        assert model_log_probs.shape == (len(alignment), len(ref_label_seq)), f"got {model_log_probs.shape}"
        model_log_probs_seq = [model_log_probs[t, ref_label_seq.index(int(l))] for t, l in enumerate(alignment)]  # [T]
        print("model prob seq frames:", [np.exp(p) for p in model_log_probs_seq])
        model_log_prob_seq_sum = sum(model_log_probs_seq)
        print("model log prob seq sum:", model_log_prob_seq_sum, "prob:", np.exp(model_log_prob_seq_sum))
        # assert abs(model_log_prob_seq_sum - align_score.item()) < 1e-5  # ???
    else:
        assert align_type == "grad"

    if alignment_label_topology == "explicit":
        align_states = alignments_ds.get_data(0, "states")
    elif alignment_label_topology == "ctc":
        align_states = []
        s = 0
        prev_label_idx = alignment_blank_idx
        for label_idx in alignment:
            if label_idx == prev_label_idx:
                align_states.append(s)
            elif label_idx == alignment_blank_idx:  # and label_idx != prev_label_idx
                # Was in label, went into blank.
                s += 1
                assert s % 2 == 0
                align_states.append(s)
            else:  # label_idx != blank_idx and label_idx != prev_label_idx
                # Went into new label.
                if prev_label_idx == alignment_blank_idx:
                    assert s % 2 == 0
                    s += 1
                else:  # was in other label before
                    assert s % 2 == 1
                    s += 2  # skip over blank state
                align_states.append(s)
            prev_label_idx = label_idx
        align_states = np.array(align_states)  # [T]
    else:
        raise ValueError(f"alignment_label_topology {alignment_label_topology!r} not supported")
    print(f"  actual align len: {len(alignment)}")
    assert len(alignment) == len(align_states)

    # noinspection PyShadowingNames
    def _start_end_time_for_align_frame_idx(t: int) -> Tuple[float, float]:
        """in seconds"""
        # For the downsampling, assume same padding, thus pad:
        stride = win_size = ref_alignment_len_factor
        pad_total = win_size - 1
        pad_left = pad_total // 2
        t0 = t * stride - pad_left  # inclusive
        t1 = t0 + win_size - 1  # inclusive
        # Now about the log mel features.
        window_len = 0.025  # 25 ms
        step_len = 0.010  # 10 ms
        sampling_rate = 16_000
        window_num_frames = int(window_len * sampling_rate)
        step_num_frames = int(step_len * sampling_rate)
        # Note: The feature extraction (e.g. RF log_mel_filterbank_from_raw) does not add padding,
        # thus starts at raw sample frame 0.
        t0 *= step_num_frames
        t1 *= step_num_frames
        t1 += window_num_frames  # exclusive
        if not include_overlap_win_in_word_boundaries:
            # take center 10ms of the 25ms window
            t0 += (window_len - step_len) / 2
            t1 -= (window_len - step_len) / 2
        return max(0.0, t0 / sampling_rate), t1 / sampling_rate

    # noinspection PyShadowingNames
    def _is_word_end(t: int) -> bool:
        label_idx = alignment[t]
        state_idx = align_states[t]
        if label_idx == alignment_blank_idx:
            return False
        if alignment_bpe_style == "bpe" and bpe_vocab.labels[label_idx].endswith("@@"):
            return False
        if t == len(alignment) - 1:
            return True
        if state_idx == align_states[t + 1]:
            return False
        if alignment_bpe_style == "spm":
            for t_ in range(t + 1, len(alignment)):
                if alignment[t_] == alignment_blank_idx:
                    continue
                if bpe_vocab.labels[alignment[t_]].startswith("▁"):
                    return True
                return False
            return True  # reached end
        assert alignment_bpe_style == "bpe"
        return True

    cur_word_start_frame = None
    word_boundaries: List[Tuple[float, float]] = []
    words_bpe: List[List[str]] = []
    prev_state_idx = 0
    for t, (label_idx, state_idx) in enumerate(zip(alignment, align_states)):
        if label_idx == alignment_blank_idx:
            continue
        if cur_word_start_frame is None:
            cur_word_start_frame = t  # new word starts here
            words_bpe.append([])
            if alignment_bpe_style == "spm":
                assert bpe_vocab.labels[label_idx].startswith("▁"), bpe_vocab.labels[label_idx]  # sanity check
        if state_idx != prev_state_idx:
            words_bpe[-1].append(bpe_vocab.labels[label_idx])
        if _is_word_end(t):
            assert cur_word_start_frame is not None
            word_frame_start, _ = _start_end_time_for_align_frame_idx(cur_word_start_frame)
            _, word_frame_end = _start_end_time_for_align_frame_idx(t)
            word_boundaries.append((word_frame_start, word_frame_end))
            cur_word_start_frame = None
        prev_state_idx = state_idx
    assert cur_word_start_frame is None  # word should have ended
    num_words = len(word_boundaries)
    print("  num words:", num_words)
    assert num_words == len(word_boundaries) == len(words_bpe) == len(ref_words)
    res = [
        (float(start_time), float(end_time), word) for (start_time, end_time), word in zip(word_boundaries, ref_words)
    ]
    print("result:", res)
    print("save to:", out_fn_pickle)
    os.makedirs(os.path.dirname(out_fn_pickle), exist_ok=True)
    pickle.dump(res, open(out_fn_pickle, "wb"))
    return res


def plot_audio_features(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + "/audio_features.pdf"
    audio_features = get_audio_features_rf()
    ref_word_boundaries = get_ref_word_boundaries()

    if not plotter:
        plotter = Plotter(plot_at_del=True, out_filename=out_fn_pdf)

    def _plot(ax):
        # audio_features is [T,D]
        # Define a custom colormap, based on Blues
        color_white = np.array((0.96862745098039216, 0.98431372549019602, 1.0))
        color_blue = np.array((0.03137254901960784, 0.18823529411764706, 0.41960784313725491))
        i = np.linspace(0.0, 1.0, 100)[:, np.newaxis] ** 4.0  # not linear, keep more white
        colors = i * color_blue[None, :] + (1 - i) * color_white[None, :]
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_blues", colors, N=100)
        mat_ = ax.matshow(audio_features.T, cmap=custom_cmap, aspect="auto")
        ax.tick_params(direction="out", length=20, width=2)

        ax.set_ylabel("Features")
        ax.set_ylim(ax.get_ylim()[::-1])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plotter.fig.colorbar(mat_, cax=cax, orientation="vertical")

    plotter.add_plot("Audio log mel filterbank features", _plot, ref_word_boundaries, rate=100, rate_rounding=None)


def plot_grad_scores(*, plotter: Optional[Plotter] = None):
    out_fn_pdf = out_prefix + seq_tag + f"/visualize_grad_scores/{model_name_short}{model_name_short_ext}/grads.pdf"

    ref_labels = get_ref_label_seq()

    score_matrix = get_grad_scores()
    print(f"{model_name_short}{model_name_short_ext}, seq {seq_tag}, shape (SxT) {score_matrix.shape}")
    assert score_matrix.shape[0] == len(ref_labels)
    score_matrix = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

    word_boundaries = get_word_boundaries_from_hdf_alignment(align_type="grad")

    if not plotter:
        plotter = Plotter(plot_at_del=True, out_filename=out_fn_pdf)

    def _plot(ax):
        # score_matrix is [S,T]
        mat_ = ax.matshow(score_matrix, cmap="Blues", aspect="auto")
        ax.tick_params(direction="out", length=20, width=2)
        ax.set_ylabel("Labels")
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_yticks(range(len(ref_labels)), [l.lower() for _, l in ref_labels], fontsize=6)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plotter.fig.colorbar(mat_, cax=cax, orientation="vertical")

    plotter.add_plot(f"{model_title}, input grads", _plot, word_boundaries, rate=100, rate_rounding=1)


def plot_model_probs(*, plotter: Optional[Plotter] = None):
    print("model probs:", model_name_short, model_name_short_ext, model_name)
    out_fn_pdf = out_prefix + seq_tag + f"/visualize_model_probs/{model_name_short}{model_name_short_ext}/probs.pdf"

    ref_labels = get_ref_label_seq()

    score_matrix = get_model_log_prob_ref_label_seq_incl_blank()  # [T,S+1]
    print(f"{model_name_short}{model_name_short_ext}, seq {seq_tag}, shape (Tx(S+1)) {score_matrix.shape}")
    assert score_matrix.shape[1] == len(ref_labels) + 1  # blank + labels
    # score_matrix = np.exp(score_matrix)

    ref_audio = get_audio_features_rf()  # [T,D]
    if model_time_downsampling > 1:
        # Transform the score matrix into time downsampling 1 (i.e. 100 Hz),
        # such that we match the features directly in the plot.
        score_matrix = np.repeat(score_matrix, model_time_downsampling, axis=0)

        # Also see get_word_boundaries_from_hdf_alignment._start_end_time_for_align_frame_idx()
        win_size = model_time_downsampling
        pad_total = win_size - 1
        pad_left = pad_total // 2
        score_matrix = score_matrix[pad_left:]  # cut off the padded frames

        score_matrix = score_matrix[: ref_audio.shape[0]]  # cut off the end
    assert score_matrix.shape[0] == ref_audio.shape[0], (
        f"probs {score_matrix.shape} vs audio {ref_audio.shape},"
        f" tag {seq_tag}, downsampling {model_time_downsampling} undone"
    )

    word_boundaries = get_word_boundaries_from_hdf_alignment(align_type="probs_best_path")

    if not plotter:
        plotter = Plotter(plot_at_del=True, out_filename=out_fn_pdf)

    def _plot(ax):
        # score_matrix is [T,S+1]
        mat_ = ax.matshow(score_matrix.T, cmap="Blues", aspect="auto")
        ax.tick_params(direction="out", length=20, width=2)
        ax.set_ylabel("Blank+labels")
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_yticks(range(len(ref_labels) + 1), ["<blank>"] + [l.lower() for _, l in ref_labels], fontsize=6)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plotter.fig.colorbar(mat_, cax=cax, orientation="vertical")

    # Note: rate=100 here because we transformed the score matrix to 100 Hz above.
    plotter.add_plot(
        f"{model_title}, model ref label probs", _plot, word_boundaries, rate=100, rate_rounding=model_time_downsampling
    )


def _log_softmax(x: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
    max_score = np.max(x, axis=axis, keepdims=True)
    x = x - max_score
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


class Plotter:
    def __init__(self, *, plot_at_del: bool = False, out_filename: str):
        self.plot_at_del = plot_at_del
        assert out_filename.endswith(".pdf")
        self.out_filename = out_filename

        self.num_figs = 0
        self.plot_titles: List[str] = []
        self.plot_callbacks: List[Callable] = []
        self.plot_word_boundaries: List[Optional[List[Tuple[float, float, str]]]] = []
        self.plot_rates: List[Union[int, float]] = []
        self.plot_rate_rounding = []

        self.fig = None
        self.ax = None

    def add_plot(
        self,
        title: str,
        callback: Callable,
        word_boundaries: Optional[List[Tuple[float, float, str]]] = None,
        *,
        rate: Union[int, float],
        rate_rounding: Optional[int] = None,
    ):
        self.plot_titles.append(title)
        self.plot_callbacks.append(callback)
        self.plot_word_boundaries.append(word_boundaries)
        self.plot_rates.append(rate)
        self.plot_rate_rounding.append(rate_rounding)
        self.num_figs += 1

    def make(self):
        self.fig, self.ax = plt.subplots(nrows=self.num_figs, ncols=1, figsize=(20, 5 * self.num_figs))
        if self.num_figs == 1:
            self.ax = [self.ax]

        for i, (title, callback, word_boundaries, rate, rate_rounding) in enumerate(
            zip(
                self.plot_titles,
                self.plot_callbacks,
                self.plot_word_boundaries,
                self.plot_rates,
                self.plot_rate_rounding,
            )
        ):
            ax = self.ax[i]

            callback(ax)

            if i in (0, self.num_figs - 1):
                ax.set_xlabel("time [sec]")
            ticks = np.arange(0, int(ax.get_xlim()[1] / rate) + 1, 1)
            # ticks = ax.get_xticks() / rate
            # ticks = [round(t, 2) for t in ticks]
            # ticks = [int(t) if t == int(t) else t for t in ticks]
            # ax.set_xticklabels(ticks)
            ax.set_xticks(ticks * rate, ticks)

            if i == self.num_figs - 1:
                ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
                ax.xaxis.set_label_position("bottom")
            elif i == 0:
                ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
                ax.xaxis.set_label_position("top")
            else:
                ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False)

            if word_boundaries:
                # print(f"{title} word boundaries:", word_boundaries)
                for start, end, word in word_boundaries:
                    start *= rate
                    end *= rate
                    if rate_rounding:
                        win_size = rate_rounding
                        pad_total = win_size - 1
                        pad_left = pad_total // 2
                        start = round((start + pad_left) / rate_rounding) * rate_rounding - pad_left
                        end = round((end + pad_left - win_size) / rate_rounding) * rate_rounding - pad_left + win_size
                    ax.axvline(start, color="black", linestyle="--", alpha=0.4)
                    ax.axvline(end, color="black", linestyle="--", alpha=0.4)
                    ax.axvspan(start, end, color="gray", alpha=0.2)
                    ax.text(
                        (start + end) / 2,
                        ax.get_ylim()[1] - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # Add padding
                        word,
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="center",
                        fontsize=12,
                    )

            if self.num_figs > 1:
                # ax.set_title(title, fontweight="bold", x=0, y=1)
                ax.set_title(title)
                # ax.set_title(title, x=1.025, y=-0.48, fontsize=18, fontweight="bold")

        # plt.gca().xaxis.tick_bottom()
        plt.tight_layout()

        os.makedirs(os.path.dirname(self.out_filename), exist_ok=True)
        print("save to:", self.out_filename)
        plt.savefig(self.out_filename)

    def __del__(self):
        if self.plot_at_del:
            self.make()


def _setup():
    import i6_core.util as util

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    font_size = 22
    matplotlib.rcParams.update(
        {"font.size": font_size, "xtick.labelsize": font_size * 0.8, "ytick.labelsize": font_size * 0.8}
    )


_setup()
