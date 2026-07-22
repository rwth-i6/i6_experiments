"""
Fair-comparison baseline for the speech-LLM comparison:
continue-train ("fine-tune") the 20-full-epoch large Loquacious AED
for 2 more full epochs.

The prefix-LLM is initialized from this AED's encoder
and then fine-tuned for a further 1-2 full epochs on Loquacious,
so the AED baseline by itself trains for fewer total epochs.
This mirrors that continuation (preload + a small fine-tune learning rate),
sweeping the peak learning rate.

Isolated from exp2025_10_04_loquacious so that only these jobs run
(the parent recipe has many unrelated runnable jobs).
"""

from __future__ import annotations

from sisyphus import Path
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_task_raw_v2

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

__setup_root_prefix__ = "exp2026_06_13_loq_aed_ft"


# The 20-full-epoch (500-subepoch) large Loquacious AED:
# the paper's AED baseline, and the encoder init of the prefix-LLM.
_large_aed_nep500_ckpt = Path(
    "/rwthfs/rz/cluster/hpcwork/p0023999/az668407/setups/2025-08-aed-large/work"
    "/i6_core/returnn/training/ReturnnTrainingJob.yzDpbrwHpvqB/output/models/epoch.500.pt",
    hash_overwrite="loq-aed-large-nEp500-epoch500",
)


def py():
    prefix = get_setup_prefix_for_module(__name__)
    task_spm10k = get_loquacious_task_raw_v2(vocab="spm10k")

    for ft_peak_lr in (5e-6, 1e-5, 2e-5):
        ft_name = f"large-nEp500-ft2ep-peakLr{ft_peak_lr}"
        ft_exp = aed_train_exp(
            ft_name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name="large", train_epoch_split=25),
            model_config={
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(
                    50, base_lr=1.0, peak_lr=ft_peak_lr, low_lr=1e-7, lowest_lr=1e-7, step_peak_fraction=0.05
                ),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "aux_loss_layers": [4, 10, 16],
                "dec_aux_loss_layers": [3],
                "max_seq_length_default_target": None,
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "preload_from_files": {
                    "base": {"filename": _large_aed_nep500_ckpt, "init_for_train": True},
                },
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + ft_name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=ft_exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
        )
