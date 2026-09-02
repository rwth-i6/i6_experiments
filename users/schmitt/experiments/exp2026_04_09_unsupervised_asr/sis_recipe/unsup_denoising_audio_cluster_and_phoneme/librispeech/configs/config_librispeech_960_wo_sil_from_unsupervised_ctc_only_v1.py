import copy
from typing import List, Dict

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ....lm_fused_recog import run_lm_fused_recog
from ....phoneme_lm.librispeech.configs.config_librispeech_960_wo_sil_v1 import (
    lm_model_args as phoneme_lm_model_args,
    lm_training_name as phoneme_lm_training_name,
)
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_w_sil_in_input_v1 import (
    base_config,
    alternate_batching,
    build_training_datasets_w_silence_in_input,
    settings,
)
from .config_librispeech_960_wo_sil_v1 import test_data_dict_wo_sil


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 100:
        return [5, 10, 20, 40, 60, 100]

    raise ValueError("Unsupported num_epochs")


base_config_ = dict_update_deep(
    copy.deepcopy(base_config),
    {
        "__network_module": "definitions.conformer_ctc_discrete_shared_v1.Model",
        "__train_step_module": "train_steps.aed_denoising_discrete_shared.train_step",
        "__forward_step_module": "recognition.discrete_audio_ctc.forward_step.forward_step",
        "train_args": {
            "aux_loss_scales": (1.0,),
            "text_ce_loss_scale": 0.0,
            "text_masked_ce_loss_scale": 0.0,
            "audio_ce_loss_scale": 0.0,
            "audio_masked_ce_loss_scale": 0.0,
            "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
            "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
        },
        "model_args.text_aux_loss_layers": (3,),
        "model_args.audio_aux_loss_layers": (3,),
        "training.__num_gpus": 1,
    },
    ["train_args"],
)

train_data = build_training_datasets_w_silence_in_input(
    sil_prob=0.25,
    surround_w_sil=True,
    settings=settings,
    max_num_sil=7,
    max_num_surround_sil=1,
)

base_num_epochs = 100


def py(checkpoints: Dict):
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    def frozen_proj_config(num_epochs, low_lr, peak_lr, freeze_emb_layers):
        """Init from the unsupervised checkpoint, freeze everything but the CTC projection.

        The supervised counterpart of exactly this setup reaches ~41% PER, i.e. a single linear
        projection is enough to read phonemes off the frozen shared-encoder states. The open question
        is how to find that projection without labels.
        """
        return dict_update_deep(
            copy.deepcopy(base_config_),
            {
                "training.preload_from_files": {
                    "pretrained_model": {
                        "ignore_missing": True,  # CTC layer is missing in checkpoint
                        "init_for_train": True,
                        "checkpoint_key": "model",
                        "filename": checkpoints[
                            "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1"
                        ][500],
                    }
                },
                "training.__lr_opts": {
                    "piecewise_epochs": [
                        0,
                        0.45 * num_epochs,
                        0.9 * num_epochs,
                        num_epochs,
                    ],
                    "piecewise_values": [low_lr, peak_lr, 1e-5, 1e-6],
                },
                "model_args.freeze_params_list": [
                    rf"encoder\.",
                    *([r"text_embedding", r"audio_embedding"] if freeze_emb_layers else []),
                ],
            },
        )

    for num_epochs, low_lr, peak_lr, freeze_emb_layers in [
        (100, 1e-4, 1e-4, True),
        # (1e-4, 1e-4, False),
    ]:
        run_experiment(
            training_name=f"{prefix_name}/baseline_low-lr-{low_lr}_peak-lr-{peak_lr}_freeze-enc{'_freeze-emb' if freeze_emb_layers else ''}",
            config=frozen_proj_config(num_epochs, low_lr, peak_lr, freeze_emb_layers),
            train_data=train_data,
            test_data_dict=test_data_dict_wo_sil,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=False,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )

    # --- corpus-level output-statistics matching -------------------------------------------------
    # The same-modality CTC losses never train the path recognition uses (the TEXT head on AUDIO
    # states), and the adversarial loss only matches the two modalities' state *marginals*, which
    # leaves the labeling unidentified -- measured on the trained model, the text-trained projection
    # has mean row-wise cosine 0.03 to the label-trained one and only 2/42 rows have the right
    # phoneme as nearest neighbour. These variants add a label-dependent signal instead: make the
    # corpus statistics of what the text head emits on audio match those of the unpaired phoneme
    # text (see models/train_steps/output_stats.py).
    for out_stats_name, out_stats_opts in [
        # unigram + expected length: the cheap diagnostic. Breaks the gross pathologies (13 of 41
        # phonemes unreachable, 1.86x too many emitted tokens) but leaves frequency-matched
        # phonemes ambiguous.
        ("uni", {"unigram_scale": 1.0, "length_scale": 1.0}),
        # + phonotactics, which is what actually resolves the residual permutation ambiguity.
        ("uni-bi", {"unigram_scale": 1.0, "length_scale": 1.0, "bigram_scale": 1.0}),
    ]:
        out_stats_config = dict_update_deep(
            frozen_proj_config(100, 1e-4, 1e-4, True),
            {"train_args": {"output_stats_opts": out_stats_opts}},
        )
        out_stats_name_full = (
            f"{prefix_name}/baseline_low-lr-{1e-4}_peak-lr-{1e-4}_freeze-enc_freeze-emb_out-stats-{out_stats_name}"
        )
        train_job = run_experiment(
            training_name=out_stats_name_full,
            config=copy.deepcopy(out_stats_config),
            train_data=train_data,
            test_data_dict=test_data_dict_wo_sil,
            keep_epochs=get_keep_epochs(100),
            skip_eval=False,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )

        # Label-synchronous CTC prefix search + phoneme-LM shallow fusion. Stage A showed the output
        # is not merely mislabeled (relabeling it does not help), so the only remaining route is to
        # generate *better hypotheses* -- which is what the LM can do during decoding. lm_scale=0.0
        # is the control: it isolates "beam search instead of greedy" from "the LM".
        if out_stats_name == "uni":
            # missing LM checkpoints mean main() built the configs in the wrong order -- fail loudly
            # rather than silently skipping the fused recognition.
            assert phoneme_lm_training_name in checkpoints, (
                f"phoneme LM checkpoints not registered under {phoneme_lm_training_name!r};"
                " its config must run before this one in main()"
            )
            lm_ckpts = checkpoints[phoneme_lm_training_name]
            run_lm_fused_recog(
                training_name=out_stats_name_full,
                config={
                    **out_stats_config["general"],
                    **out_stats_config.get("recog", {}),
                    "__callback_module": out_stats_config["__callback_module"],
                },
                train_job=train_job,
                train_args={"net_args": out_stats_config["model_args"]},
                train_data=train_data,
                test_data_dict=test_data_dict_wo_sil,
                checkpoints=[20],
                lm_checkpoint=lm_ckpts[1000],
                lm_args=phoneme_lm_model_args,
                beam_size=12,
                lm_scales=(0.0, 1.0),
                length_rewards=(0.0,),
                rqmt={"time": 2, "gpu_mem": 24},
            )
            # Viterbi (best-path) prefix scoring instead of the CTC marginal. The marginal sums over
            # *all* alignments of y, and the alignment count grows with |y|, so on this (diffuse)
            # model the combinatorial term dominates the acoustic evidence and it over-generates:
            # 111 labels/utt against a 62-label reference -> 154% PER, versus greedy's 109%. The
            # model's own frame-level non-blank rate implies ~63 labels/utt, i.e. the calibration is
            # fine and it is the marginalization that is wrong. "max" has no such bias: with a wide
            # enough beam and lm_scale=0 it *is* best-path (greedy) decoding, which is the sane
            # baseline for the LM to build on. Two runs: the control and the LM on top of it.
            # NB if the lm=0.0 control does not land near greedy's 108.7, suspect beam pruning
            # first -- measured on synthetic worst-case (random) posteriors, max-mode recovers the
            # best path on 0/5 seqs at beam 4, 1/5 at beam 40 and 5/5 at beam 200.
            run_lm_fused_recog(
                training_name=out_stats_name_full,
                config={
                    **out_stats_config["general"],
                    **out_stats_config.get("recog", {}),
                    "__callback_module": out_stats_config["__callback_module"],
                },
                train_job=train_job,
                train_args={"net_args": out_stats_config["model_args"]},
                train_data=train_data,
                test_data_dict=test_data_dict_wo_sil,
                checkpoints=[20],
                lm_checkpoint=lm_ckpts[1000],
                lm_args=phoneme_lm_model_args,
                beam_size=12,
                lm_scales=(0.0, 1.0),
                length_rewards=(0.0,),
                score_type="max",
                rqmt={"time": 2},
            )
            # Time-synchronous CTC prefix beam search (the classic algorithm) + Viterbi scoring.
            # The label-synchronous search above is only valid when the CTC prefix score is an
            # auxiliary term next to a dominant decoder: standalone it compares hypotheses that have
            # consumed different numbers of *frames*, so the beam fills with prefixes that crammed
            # their labels into the first frames and then keep extending -- 13 nats below a verified
            # optimum, insensitive to beam width. Time-synchronous decoding advances one frame at a
            # time, so every hypothesis has consumed the same audio and the scores are comparable.
            # Verified: with score_type="max" it reproduces greedy decoding *exactly* already at
            # beam 4 (labels/seq and score both match the best path), so the lm=0.0 control should
            # land on greedy's 108.7 and the LM builds on top of that rather than on 154/182.
            run_lm_fused_recog(
                training_name=out_stats_name_full,
                config={
                    **out_stats_config["general"],
                    **out_stats_config.get("recog", {}),
                    "__callback_module": out_stats_config["__callback_module"],
                },
                train_job=train_job,
                train_args={"net_args": out_stats_config["model_args"]},
                train_data=train_data,
                test_data_dict=test_data_dict_wo_sil,
                checkpoints=[20],
                lm_checkpoint=lm_ckpts[1000],
                lm_args=phoneme_lm_model_args,
                beam_size=12,
                lm_scales=(0.0, 0.1, 0.2, 0.3, 0.6, 1.0),
                length_rewards=(0.0,),
                score_type="max",
                search_type="time_sync",
                rqmt={"time": 2},
            )
            # run_lm_fused_recog(
            #     training_name=out_stats_name_full,
            #     config={
            #         **out_stats_config["general"],
            #         **out_stats_config.get("recog", {}),
            #         "__callback_module": out_stats_config["__callback_module"],
            #     },
            #     train_job=train_job,
            #     train_args={"net_args": out_stats_config["model_args"]},
            #     train_data=train_data,
            #     test_data_dict=test_data_dict_wo_sil,
            #     checkpoints=[20],
            #     lm_checkpoint=lm_ckpts[1000],
            #     lm_args=phoneme_lm_model_args,
            #     beam_size=12,
            #     lm_scales=(0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            #     length_rewards=(0.0,),
            #     rqmt={"time": 2},
            # )

    codebook_config = dict_update_deep(
        copy.deepcopy(base_config_),
        {
            "training.preload_from_files": {
                "pretrained_model": {
                    "ignore_missing": True,  # CTC layer is missing in checkpoint
                    "init_for_train": True,
                    "checkpoint_key": "model",
                    "filename": checkpoints[
                        "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1"
                    ][500],
                }
            },
            "training.__lr_opts": {
                "piecewise_epochs": [
                    0,
                    0.45 * base_num_epochs,
                    0.9 * base_num_epochs,
                    base_num_epochs,
                ],
                "piecewise_values": [1e-4, 1e-4, 1e-5, 1e-6],
            },
            "model_args.freeze_params_list": [
                rf"encoder\.",
                r"text_embedding",
                r"audio_embedding",
            ],
            "model_args.codebook_opts": {},  # {} -> enable codebook with default settings
            "train_args": {"codebook_diversity_loss_scale": 0.1},
        },
    )
    run_experiment(
        training_name=f"{prefix_name}/baseline_low-lr-{1e-4}_peak-lr-{1e-4}_freeze-enc_freeze-emb_codebook",
        config=codebook_config,
        train_data=train_data,
        test_data_dict=test_data_dict_wo_sil,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        recog_model_args=dict_update_deep(codebook_config["model_args"], {"codebook_opts.codebook_prob": 1.0}),
        codebook_analysis_opts={"checkpoints": get_keep_epochs(base_num_epochs)},
    )

    # for num_epochs, batch_size, latent_groups, latent_vars, codebook_prob in [
    #     (100, 15_000, 1, 320, 0.5),
    #     (100, 15_000, 1, 100, 0.5),
    # ]:
    #     run_experiment(
    #         training_name=f"{prefix_name}/baseline_bs-{batch_size}_low-lr-{1e-4}_peak-lr-{1e-4}_freeze-enc_freeze-emb_codebook-g-{latent_groups}-v-{latent_vars}-p-{codebook_prob}",
    #         config=dict_update_deep(
    #             copy.deepcopy(base_config_),
    #             {
    #                 "training.preload_from_files": {
    #                     "pretrained_model": {
    #                         "ignore_missing": True,  # CTC layer is missing in checkpoint
    #                         "init_for_train": True,
    #                         "checkpoint_key": "model",
    #                         "filename": checkpoints[
    #                             "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1"
    #                         ][500],
    #                     }
    #                 },
    #                 "training.batch_size": batch_size,
    #                 "training.__lr_opts": {
    #                     "piecewise_epochs": [
    #                         0,
    #                         0.45 * num_epochs,
    #                         0.9 * num_epochs,
    #                         num_epochs,
    #                     ],
    #                     "piecewise_values": [1e-4, 1e-4, 1e-5, 1e-6],
    #                 },
    #                 "model_args.freeze_params_list": [
    #                     rf"encoder\.",
    #                     r"text_embedding",
    #                     r"audio_embedding",
    #                 ],
    #                 "model_args.codebook_opts": {
    #                     "codebook_prob": 0.5,
    #                     "latent_vars": 320,
    #                     "latent_groups": 2,
    #                 },  # {} -> enable codebook with default settings
    #                 "train_args": {"codebook_diversity_loss_scale": 0.1},
    #             },
    #         ),
    #         train_data=train_data,
    #         test_data_dict=test_data_dict_wo_sil,
    #         keep_epochs=get_keep_epochs(num_epochs),
    #         skip_eval=False,
    #         additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    #     )
