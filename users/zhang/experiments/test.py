from i6_experiments.users.zhang.experiments.WER_PPL.util import WER_ppl_PlotAndSummaryJob, GnuPlotJob
from typing import Tuple, Dict, Set, List, Optional, Union
from sisyphus import Job, Task, tk, gs
from i6_experiments.users.zhang.datasets.librispeech import get_vocab_by_str
import i6_core.util as util
from i6_experiments.users.zhang.experiments.exp_wer_ppl import EVAL_DATASET_KEYS
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import _get_lm_model
from collections import namedtuple

_Lm = namedtuple("Lm", ["name", "train_version", "setup"])

_lms = {
    "n24-d512": _Lm("trafo-n24-d512-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b100_5k", "v3", "lm"),
    "n96-d512": _Lm("trafo-n96-d512-gelu-drop0-b32_1k", "v3", "lm"),
    "n32-d1024": _Lm("trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b32_1k", "v3", "lm"),
    "n32-d1024-claix2023": _Lm(
        "trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k", "v4", "lm_claix2023"
    ),
    "n32-d1280-claix2023": _Lm(
        "trafo-n32-d1280-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-spm10k", "v4", "lm_claix2023"
    ),
}



def py():
    #recog_ext_with_lm(ctc_model_name="L16-D1280-spm10k-auxAED-b100k", lm_name="n32-d1280-claix2023")  # 3.88 (!!)
    lm = _get_lm_model(_lms["n32-d1024"])
    vocab = "spm10k"
    from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_dataset
    from i6_experiments.users.zhang.experiments.lm_getter import GetSubwordRatioJob
    from i6_experiments.users.zhang.datasets.librispeech import get_librispeech_lm_combined_txt
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
    from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
    lm_dataset = get_librispeech_lm_dataset(vocab=vocab)
    ratio = GetSubwordRatioJob(get_librispeech_lm_combined_txt(), vocab, get_returnn_subword_nmt(), apply_job=ApplySentencepieceToTextJob).out_ratio
    #ratio = 1
    # tk.register_output(f"LBS_{vocab}_ratio", ratio)
    from i6_experiments.users.zhang.experiments.lm.lm_ppl import compute_ppl_single_epoch
    ppls = compute_ppl_single_epoch(
        prefix_name="n32-d1280-claix2023_trafo_spm10k",
        model_with_checkpoint=lm,
        epoch="epoch_unk",
        dataset=lm_dataset,
        dataset_keys=["transcriptions-test-other", "transcriptions-dev-other"],
        exponent=ratio,
        same_seq=True,
        batch_size=10_000,
    )
