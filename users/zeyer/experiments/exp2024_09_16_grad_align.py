"""
More on grad align
"""


from __future__ import annotations

from typing import Optional, Any, List, Sequence, Dict
from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoint

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    train_exp as ctc_train_exp,
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    speed_pert_librosa_config,
    Model as CtcModel,
    ctc_model_def,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    _get_cfg_lrlin_oclr_by_bs_nep,
    _log_mel_feature_dim,
    _batch_size_factor,
    _get_cfg_lrlin_oclr_by_bs_nep,
    _get_bos_idx,
    _get_eos_idx,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    Model as AedModel,
    aed_training,
)
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoder
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ForwardRFDef
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from sisyphus import tk, Path


def py():
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        seq_list_960_to_split_100_360_500,
        seq_list_split_100_360_500_to_single_960,
    )
    from .exp2024_09_09_grad_align import CalcAlignmentMetrics, ForcedAlignOnScoreMatrixJob
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    prefix = "exp2024_09_16_grad_align/"

    # gmm_alignment_hdf = Path(
    #     "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_core/returnn/hdf/ReturnnDumpHDFJob.nQ1YkjerObMO/output/data.hdf"
    # )
    gmm_alignment_allophones = Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    gmm_alignment_sprint_cache = Path(
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
    )
    features_sprint_cache = Path(  # for exact timings
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
    )
    seq_list_ref = Path(
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
    )
    seq_list = seq_list_960_to_split_100_360_500(seq_list_ref)
    vocabs = {
        "spm10k": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm10k").model_file).out_vocab, 10_240),
        "spm512": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm512").model_file).out_vocab, 512),
        "bpe10k": ("bpe", get_vocab_by_str("bpe10k").vocab, 10_025),
    }

    # WERs: dev-other/test-other
    for shortname, fullname, vocab in [
        (  # ctc forced align: 110.7/43.7ms
            "noBias",  # 5.65/5.94, better baseline
            "v6-relPosAttDef-noBias"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            "spm10k",
        ),
        (  # ctc forced align: 111.5/52.9ms
            "base",  # 5.77/6.03
            # output/ctc/v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001/recog_results_best
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            "spm10k",
        ),
        (  # ctc forced align: 116.8/74.4ms
            "lpNormedGradC05_11P1",  # 5.71/5.87
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC05_11P1",
            "spm10k",
        ),
        (
            "lpNormedGradC05_11P1Seq",  # 5.83/5.91
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC05_11P1Seq",
            "spm10k",
        ),
        (
            "lpNormedGradC01_11P1",  # 6.21/6.55
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC01_11P1",
            "spm10k",
        ),
        (  # ctc forced align: 98.5/77.6ms
            "blankSep",  # 5.73/6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-blankSep",
            "spm10k",
        ),
        (  # ctc forced align: 75.4/42.7ms
            "base-spm512",  # 5.97/6.21
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm512-bpeSample001",
            "spm512",
        ),
        (  # ctc forced align: 59.6/48.5ms
            "base-spm512-blankSep",  # 6.02/6.04
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm512-bpeSample001"
            "-blankSep",
            "spm512",
        ),
        (  # ctc forced align: 113.9/68.1ms
            "base-bpe10k",  # 6.18/6.35
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-bpe10k-bpeSample001",
            "bpe10k",
        ),
        (  # ctc forced align: 84.9/64.2ms
            "base-bpe10k-blankSep",  # 5.98/6.13
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-bpe10k-bpeSample001"
            "-blankSep",
            "bpe10k",
        ),
        (
            "ebranchformer",  # 5.54/5.69
            # output/ctc/v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001/recog_results_best
            "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
            "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
            "spm10k",
        ),
    ]:
        # Note: task hardcoded... (and also not needed, I just need the train dataset...)
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        # train_dataset.main_dataset["fixed_random_subset"] = 1000  # for debugging...
        train_dataset.main_dataset["seq_list_filter_file"] = seq_list

        for epoch in [20, 40, 80, 160, 320, 500, -1]:
            ctc_model = sis_get_ctc_model(fullname, epoch=epoch)

            alignment = ctc_forced_align(ctc_model, train_dataset)
            alignment.creator.add_alias(f"{prefix}ctc_forced_align/{shortname}-ep{epoch}/align")
            tk.register_output(f"{prefix}ctc_forced_align/{shortname}-ep{epoch}/align.hdf", alignment)

            name = f"ctc_forced_align/{shortname}-ep{epoch}/align-metrics"
            job = CalcAlignmentMetrics(
                seq_list=seq_list,
                seq_list_ref=seq_list_ref,
                alignment_hdf=alignment,
                alignment_label_topology="ctc",
                alignment_bpe_vocab=vocabs[vocab][1],
                alignment_bpe_style=vocabs[vocab][0],
                alignment_blank_idx=vocabs[vocab][2],
                features_sprint_cache=features_sprint_cache,
                ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                ref_alignment_allophones=gmm_alignment_allophones,
                ref_alignment_len_factor=6,
            )
            job.add_alias(prefix + name)
            tk.register_output(prefix + name + ".json", job.out_scores)
            tk.register_output(prefix + name + ".short_report.txt", job.out_short_report_str)

        for extra_name, grad_opts in [
            ("", {}),
            # *([("-bs1", {"max_seqs": 1})] if shortname == "base" else []),  # test influence of batching
            (
                "-blankStopGrad-inclBlankState-p0.1",
                {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True, "grad_norm_p": 0.1},
            ),
            (
                "-inclBlankState-p0.1",
                {"ctc_partial_scores_include_next_blank": True, "grad_norm_p": 0.1},
            ),
            (
                "-inclBlankStateBoth-p0.1",
                {"ctc_partial_scores_include_next_blank": "both", "grad_norm_p": 0.1},
            ),
            (
                "-blankStopGrad-inclBlankStateBoth-p0.1",
                {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": "both", "grad_norm_p": 0.1},
            ),
            (
                "-blankStopGrad-inclBlankStateBothPrev-p0.1",
                {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": "both_prev", "grad_norm_p": 0.1},
            ),
        ]:
            grad_opts = grad_opts.copy()
            # base model
            epoch = grad_opts.pop("epoch", -1)
            ctc_model = sis_get_ctc_model(fullname, epoch=epoch)

            if "blankSep" in shortname:
                assert ctc_model.definition.config["out_blank_separated"]  # sanity check

            # Now grad based align
            grads = get_ctc_input_grads(
                ctc_model,
                train_dataset,
                config={
                    **({"fixed_blank_sep_v1": True} if "blankSep" in shortname else {}),
                    **grad_opts,
                    **({"batch_size": 5_000 * _batch_size_factor} if shortname == "ebranchformer" else {}),
                },
            )
            if shortname == "blankSep":
                # I get some strange CUDA error: an illegal memory access was encountered,
                # maybe this fixes it:
                grads.creator.set_env("CUDA_LAUNCH_BLOCKING", "1")

            grad_name = f"ctc-grad-align/{shortname}{extra_name}"
            tk.register_output(f"{prefix}{grad_name}/input_grads.hdf", grads)
            grads.creator.add_alias(f"{prefix}{grad_name}/input_grads")

            # see also exp2024_09_09_grad_align.py
            for align_opts in [
                # {"apply_softmax_over_time": True, "blank_score": -4},
                {"apply_softmax_over_time": True, "blank_score": -6},
                # {"apply_softmax_over_time": True, "blank_score": -7},
                # {
                #     "apply_softmax_over_time": True,
                #     "blank_score": "calc",
                #     "blank_score_est": "flipped_after_softmax_over_time",
                #     "non_blank_score_reduce": "log_mean_exp",
                #     "blank_score_flipped_percentile": 80,
                #     "apply_softmax_over_labels": True,
                # },
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 60,
                    "apply_softmax_over_labels": True,
                },
                # {
                #     "apply_softmax_over_time": True,
                #     "blank_score": "calc",
                #     "blank_score_est": "flipped_after_softmax_over_time",
                #     "non_blank_score_reduce": "log_mean_exp",
                #     "blank_score_flipped_percentile": 40,
                #     "apply_softmax_over_labels": True,
                # },
            ]:
                # factor, grad_hdf = grads[grad_name]
                factor = 1
                grad_hdf = grads

                # The dumped grads cover about 9.6h audio from train.
                name = grad_name
                for k, v in align_opts.items():
                    # Shorten the name a bit. We also might run into `File name too long` errors otherwise.
                    if k.startswith("blank_score"):
                        k = "bScore" + k[len("blank_score") :]
                    k = {"apply_softmax_over_time": "smTime", "apply_softmax_over_labels": "smLabels"}.get(k, k)
                    name += f"-{k}{v}"
                job = ForcedAlignOnScoreMatrixJob(
                    score_matrix_hdf=grad_hdf,
                    cut_off_eos=False,
                    # Need to know blank idx for the generated output alignment.
                    num_labels=vocabs[vocab][2] + 1,
                    blank_idx=vocabs[vocab][2],
                    returnn_dataset=train_dataset.get_main_dataset(),
                    **align_opts,
                )
                job.add_alias(prefix + name + "/align")
                tk.register_output(prefix + name + "/align.hdf", job.out_align)
                alignment_hdf = job.out_align

                name += "/align-metrics"
                job = CalcAlignmentMetrics(
                    seq_list=seq_list,
                    seq_list_ref=seq_list_ref,
                    alignment_hdf=alignment_hdf,
                    alignment_bpe_vocab=vocabs[vocab][1],
                    alignment_bpe_style=vocabs[vocab][0],
                    alignment_blank_idx=vocabs[vocab][2],
                    features_sprint_cache=features_sprint_cache,
                    ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                    ref_alignment_allophones=gmm_alignment_allophones,
                    ref_alignment_len_factor=factor,
                )
                job.add_alias(prefix + name)
                tk.register_output(prefix + name + ".json", job.out_scores)
                tk.register_output(prefix + name + "_short_report.txt", job.out_short_report_str)

    # Grad align debug
    for name, grad_opts in [
        ("base", {}),  # 98.0/74.6
        ("base-inclBlankState", {"ctc_partial_scores_include_next_blank": True}),  # 94.4/69.6
        ("base", {"epoch": 80}),  # 113.4/93.9
        ("base-p1", {"grad_norm_p": 1}),  # 97.4/74.5
        ("base-p0.1", {"grad_norm_p": 0.1}),  # 98.5/75.9
        ("base-multSource", {"source_grad_mult_with_source": True}),  # 102.3/77.2
        ("base-blankStopGrad", {"stop_grad_blank": True}),  # 97.9/76.2
        (  # 87.6/65.1
            "base-blankStopGrad-inclBlankState-p0.1",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True, "grad_norm_p": 0.1},
        ),
        (  # 87.7/64.9
            "base-blankStopGrad-inclBlankState-p0.5",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True, "grad_norm_p": 0.5},
        ),
        (  # 88.4/65.3
            "base-blankStopGrad-inclBlankState-p1",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True, "grad_norm_p": 1},
        ),
        (  # 91.7/67.2
            "base-blankStopGrad-inclBlankState-p3",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True, "grad_norm_p": 3},
        ),
        (  # 90.0/65.9
            "base-blankStopGrad-inclBlankState",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True},
        ),
        (
            "base-blankStopGrad-inclBlankState-p0.1-multSource",
            {
                "stop_grad_blank": True,
                "ctc_partial_scores_include_next_blank": True,
                "grad_norm_p": 0.1,
                "source_grad_mult_with_source": True,
            },
        ),
    ]:
        grad_opts = grad_opts.copy()
        # base model
        epoch = grad_opts.pop("epoch", -1)
        ctc_model = sis_get_ctc_model(
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            epoch=epoch,
        )
        vocab = "spm10k"
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        train_dataset.main_dataset["fixed_random_subset"] = 100  # for debugging...
        # train_dataset.main_dataset["seq_list_filter_file"] = seq_list
        grads = get_ctc_input_grads(ctc_model, train_dataset, grad_opts)
        tk.register_output(f"{prefix}debug/ctc_grad_align/{name}-ep{epoch}/input_grads.hdf", grads)
        grads.creator.add_alias(f"{prefix}debug/ctc_grad_align/{name}-ep{epoch}/input_grads")

        # see also exp2024_09_09_grad_align.py
        for opts in [
            {"sm": True, "blank_score": -6},
            # Always a bit better (e.g. 84.1/63.1 vs 87.6/65.1), but more heuristic:
            {
                "sm": True,
                "blank_score": "calc",
                "blank_score_est": "flipped_after_softmax_over_time",
                "non_blank_score_reduce": "log_mean_exp",
                "blank_score_flipped_percentile": 60,
                "apply_softmax_over_labels": True,
            },
        ]:
            opts = opts.copy()
            apply_softmax_over_time = opts.pop("sm", False)
            # factor, grad_hdf = grads[grad_name]
            factor = 1
            grad_hdf = grads

            align_name = f"debug/ctc_grad_align/{name}-ep{epoch}/grad-align"
            for k, v in opts.items():
                # Shorten the name a bit. We also might run into `File name too long` errors otherwise.
                if k.startswith("blank_score"):
                    k = "bScore" + k[len("blank_score") :]
                k = {"apply_softmax_over_time": "smTime", "apply_softmax_over_labels": "smLabels"}.get(k, k)
                align_name += f"-{k}{v}"
            job = ForcedAlignOnScoreMatrixJob(
                score_matrix_hdf=grad_hdf,
                cut_off_eos=False,
                apply_softmax_over_time=apply_softmax_over_time,
                # Need to know blank idx for the generated output alignment.
                num_labels=vocabs[vocab][2] + 1,
                blank_idx=vocabs[vocab][2],
                returnn_dataset=train_dataset.get_main_dataset(),
                **opts,
            )
            job.add_alias(prefix + align_name + "/align")
            tk.register_output(prefix + align_name + "/align.hdf", job.out_align)
            alignment_hdf = job.out_align

            from i6_experiments.users.zeyer.datasets.utils.extract_seq_list import ExtractSeqListJob

            ds = train_dataset.get_main_dataset().copy()
            ds["audio"] = None
            ds["targets"] = None

            seq_list_debug = ExtractSeqListJob(returnn_dataset=ds).out_seq_list
            seq_list_debug_ref = seq_list_split_100_360_500_to_single_960(seq_list_debug)

            align_name += "/metrics"
            job = CalcAlignmentMetrics(
                seq_list=seq_list_debug,
                seq_list_ref=seq_list_debug_ref,
                alignment_hdf=alignment_hdf,
                alignment_bpe_vocab=vocabs[vocab][1],
                alignment_bpe_style=vocabs[vocab][0],
                alignment_blank_idx=vocabs[vocab][2],
                features_sprint_cache=features_sprint_cache,
                ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                ref_alignment_allophones=gmm_alignment_allophones,
                ref_alignment_len_factor=factor,
            )
            job.add_alias(prefix + align_name)
            tk.register_output(prefix + align_name + ".json", job.out_scores)
            tk.register_output(prefix + align_name + "_short_report.txt", job.out_short_report_str)

    # ---- AED models ----
    # WERs: dev-other/test-other
    for shortname, fullname, vocab in [
        (
            "base",  # 4.98/5.49
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm10k-spmSample07",
            "spm10k",
        )
    ]:
        # Note: task hardcoded... (and also not needed, I just need the train dataset...)
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        # train_dataset.main_dataset["fixed_random_subset"] = 1000  # for debugging...
        train_dataset.main_dataset["seq_list_filter_file"] = seq_list

        for extra_name, grad_opts in [
            ("", {}),
            # *([("-bs1", {"max_seqs": 1})] if shortname == "base" else []),  # test influence of batching
            ("-p1", {"grad_norm_p": 1}),
            ("-p0.1", {"grad_norm_p": 0.1}),
        ]:
            grad_opts = grad_opts.copy()
            # base model
            epoch = grad_opts.pop("epoch", -1)
            aed_model = sis_get_aed_model(fullname, epoch=epoch)

            # Now grad based align
            grads = get_aed_input_grads(
                aed_model,
                train_dataset,
                config={
                    **grad_opts,
                    **({"batch_size": 5_000 * _batch_size_factor} if shortname == "ebranchformer" else {}),
                },
            )

            grad_name = f"aed-grad-align/{shortname}{extra_name}"
            tk.register_output(f"{prefix}{grad_name}/input_grads.hdf", grads)
            grads.creator.add_alias(f"{prefix}{grad_name}/input_grads")

            # see also exp2024_09_09_grad_align.py
            for align_opts in [
                {"apply_softmax_over_time": True, "blank_score": -6},
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 80,
                    "apply_softmax_over_labels": True,
                },
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 60,
                    "apply_softmax_over_labels": True,
                },
                {
                    "apply_softmax_over_time": True,
                    "blank_score": "calc",
                    "blank_score_est": "flipped_after_softmax_over_time",
                    "non_blank_score_reduce": "log_mean_exp",
                    "blank_score_flipped_percentile": 40,
                    "apply_softmax_over_labels": True,
                },
            ]:
                # factor, grad_hdf = grads[grad_name]
                factor = 1
                grad_hdf = grads

                # The dumped grads cover about 9.6h audio from train.
                name = grad_name
                for k, v in align_opts.items():
                    # Shorten the name a bit. We also might run into `File name too long` errors otherwise.
                    if k.startswith("blank_score"):
                        k = "bScore" + k[len("blank_score") :]
                    k = {"apply_softmax_over_time": "smTime", "apply_softmax_over_labels": "smLabels"}.get(k, k)
                    name += f"-{k}{v}"
                job = ForcedAlignOnScoreMatrixJob(
                    score_matrix_hdf=grad_hdf,
                    cut_off_eos=False,
                    # Need to know blank idx for the generated output alignment.
                    num_labels=vocabs[vocab][2] + 1,
                    blank_idx=vocabs[vocab][2],
                    returnn_dataset=train_dataset.get_main_dataset(),
                    **align_opts,
                )
                job.add_alias(prefix + name + "/align")
                tk.register_output(prefix + name + "/align.hdf", job.out_align)
                alignment_hdf = job.out_align

                name += "/align-metrics"
                job = CalcAlignmentMetrics(
                    seq_list=seq_list,
                    seq_list_ref=seq_list_ref,
                    alignment_hdf=alignment_hdf,
                    alignment_bpe_vocab=vocabs[vocab][1],
                    alignment_bpe_style=vocabs[vocab][0],
                    alignment_blank_idx=vocabs[vocab][2],
                    features_sprint_cache=features_sprint_cache,
                    ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                    ref_alignment_allophones=gmm_alignment_allophones,
                    ref_alignment_len_factor=factor,
                )
                job.add_alias(prefix + name)
                tk.register_output(prefix + name + ".json", job.out_scores)
                tk.register_output(prefix + name + "_short_report.txt", job.out_short_report_str)

    # TODO job to dump grads, diff variants:
    #  - using prob entropy instead of ground truth log prob
    pass

    # TODO align using att weights


_called_ctc_py_once = False


def sis_get_ctc_model(name: str, *, epoch: int = -1) -> ModelWithCheckpoint:
    if (
        name == "v6-relPosAttDef-noBias"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
    ):
        exp = ctc_train_exp(  # 5.65 (!!!)
            "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
            "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
            config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            model_config={
                "enc_conformer_layer": rf.build_dict(
                    rf.encoder.conformer.ConformerEncoderLayer,
                    ff=rf.build_dict(
                        rf.encoder.conformer.ConformerPositionwiseFeedForward,
                        activation=rf.build_dict(rf.relu_square),
                        with_bias=False,
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            },
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        )

    else:

        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import py as ctc_py, _train_experiments

        global _called_ctc_py_once

        if not _called_ctc_py_once:
            from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

            with disable_register_output():
                ctc_py()
            _called_ctc_py_once = True

        exp = _train_experiments[name]

    if epoch < 0:
        return exp.get_last_fixed_epoch()
    return exp.get_epoch(epoch)


_called_aed_py_once = False


def sis_get_aed_model(name: str, *, epoch: int = -1) -> ModelWithCheckpoint:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import py as aed_py, _train_experiments

    global _called_aed_py_once

    if not _called_aed_py_once:
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        with disable_register_output():
            aed_py()
        _called_aed_py_once = True

    exp = _train_experiments[name]

    if epoch < 0:
        return exp.get_last_fixed_epoch()
    return exp.get_epoch(epoch)


def ctc_forced_align(model: ModelWithCheckpoint, dataset: DatasetConfig) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_dict = dataset.get_extern_data()
    default_input_dict = extern_data_dict[dataset.get_default_input()]
    input_dims: Sequence[Dim] = (
        default_input_dict["dims"] if "dims" in default_input_dict else default_input_dict["dim_tags"]
    )
    assert isinstance(input_dims, (tuple, list)) and all(isinstance(dim, Dim) for dim in input_dims)
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    classes_dim = default_target_dict["sparse_dim"]
    assert isinstance(classes_dim, Dim)
    classes_with_blank_dim = classes_dim + 1

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_ctc_model_forced_align_step,
        config={
            "model_outputs": {
                "output": {"shape": (None,), "sparse_dim": classes_with_blank_dim},
                "scores": {"shape": ()},
            }
        },
        forward_rqmt={"time": 12},
    )


def _ctc_model_forced_align_step(*, model: CtcModel, extern_data: TensorDict, **_kwargs):
    """
    :param model: model with batch size 1
    """
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.fsa import best_path_ctc

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    out_spatial_dim = expected_output.dims[-1]

    logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
    path, score = best_path_ctc(
        logits=logits,
        input_spatial_dim=enc_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets.get_time_dim_tag(),
        blank_index=model.blank_idx,
    )
    out_spatial_dim.declare_same_as(enc_spatial_dim)
    path.mark_as_default_output(shape=[batch_dim, enc_spatial_dim])
    score.mark_as_output("scores", shape=[batch_dim])


def get_ctc_input_grads(
    model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_dict = dataset.get_extern_data()
    default_input_dict = extern_data_dict[dataset.get_default_input()]
    input_dims: Sequence[Dim] = (
        default_input_dict["dims"] if "dims" in default_input_dict else default_input_dict["dim_tags"]
    )
    assert isinstance(input_dims, (tuple, list)) and all(isinstance(dim, Dim) for dim in input_dims)
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    batch_dim, out_spatial_dim = default_target_dict["dim_tags"]
    feat_spatial_dim = Dim(None, name="feat_spatial_dim")  # it's not the input raw audio spatial dim...

    if config:
        config = config.copy()
    else:
        config = {}
    config.setdefault("__batch_size_dependent", True)
    config.setdefault("batch_size", 10_000 * _batch_size_factor)  # grads need more mem

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_ctc_model_get_input_grads_step,
        config={
            "model_outputs": {
                "output": {"dims": [batch_dim, out_spatial_dim, feat_spatial_dim], "dtype": "float32"},
                "feat_size": {"dims": [batch_dim], "dtype": "int32"},
                "targets_size": {"dims": [batch_dim], "dtype": "int32"},
                "partial_scores": {"dims": [batch_dim, out_spatial_dim], "dtype": "float32"},
            },
            **config,
        },
        forward_rqmt={"time": 24},
    )


def _ctc_model_get_input_grads_step(*, model: CtcModel, extern_data: TensorDict, **_kwargs):
    import torch
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.fsa import ctc_partial_scores
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    feat_spatial_dim_ = expected_output.dims[-1]

    # Call: logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
    in_spatial_dim = source.get_time_dim_tag()

    # log mel filterbank features
    source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        source,
        in_spatial_dim=in_spatial_dim,
        out_dim=model.in_dim,
        sampling_rate=16_000,
    )
    in_spatial_dim.get_size_tensor().mark_as_output("feat_size")
    feat_spatial_dim_.declare_same_as(in_spatial_dim)
    if model.feature_batch_norm:
        source = model.feature_batch_norm(source)
    if model.feature_norm:
        source = rf.normalize(source, axis=in_spatial_dim)
    if model.feature_stats:
        source = (source - model.feature_stats.mean) / model.feature_stats.std_dev

    with torch.enable_grad():
        # Just that we have well-defined dim order, for the grad logic below.
        source = source.copy_transpose((batch_dim, in_spatial_dim, model.in_dim))  # [B,T_in,D]
        source.raw_tensor.requires_grad = True

        # (No Mixup, no SpecAugment)

        # Encoder including convolutional frontend
        enc, enc_spatial_dim = model.encoder(source, in_spatial_dim=in_spatial_dim)
        logits = model.enc_logits(enc)
        if config.bool("stop_grad_blank", False):
            # Just that we have well-defined dim order, for the grad logic.
            logits = logits.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))

            # noinspection PyShadowingNames
            def _zero_grad_blank_hook(grad):
                grad = grad.clone()
                grad[:, :, model.blank_idx] = 0
                return grad

            logits.raw_tensor.register_hook(_zero_grad_blank_hook)

        if model.out_blank_separated:
            assert config.bool("fixed_blank_sep_v1", False)
        log_probs = model.log_probs_wb_from_logits(logits)

        targets_spatial_dim = targets.get_time_dim_tag()
        targets_spatial_dim.get_size_tensor().mark_as_output("targets_size")
        scores = ctc_partial_scores(
            logits=log_probs,
            logits_normalized=True,
            input_spatial_dim=enc_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
            include_next_blank=config.typed_value("ctc_partial_scores_include_next_blank", False),
        )  # [B,T_out]
        scores.mark_as_output("partial_scores")
        scores_ta = TensorArray.unstack(scores, axis=targets_spatial_dim)

        source_grad_mult_with_source = config.bool("source_grad_mult_with_source", False)
        grad_norm_p = config.float("grad_norm_p", 2.0)

        grad_norms = []
        for t in range(targets_spatial_dim.get_dim_value()):
            source.raw_tensor.grad = None
            scores_t = scores_ta[t]  # [B], +log prob
            partial_loss = scores_t.raw_tensor.sum()
            partial_loss.backward(retain_graph=True)
            grad: torch.Tensor = source.raw_tensor.grad  # [B,T_in,D]  # noqa
            if source_grad_mult_with_source:
                grad = grad * source.raw_tensor
            grad_norm = torch.norm(grad, p=grad_norm_p, dim=2)  # [B,T_in]
            grad_norms.append(grad_norm)
        grad_norms = torch.stack(grad_norms, dim=0)  # [T_out,B,T_in]
        grad_norms_ = rf.convert_to_tensor(grad_norms, dims=[targets_spatial_dim, batch_dim, in_spatial_dim])
        grad_norms_.mark_as_default_output()


def get_aed_input_grads(
    model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_dict = dataset.get_extern_data()
    default_input_dict = extern_data_dict[dataset.get_default_input()]
    input_dims: Sequence[Dim] = (
        default_input_dict["dims"] if "dims" in default_input_dict else default_input_dict["dim_tags"]
    )
    assert isinstance(input_dims, (tuple, list)) and all(isinstance(dim, Dim) for dim in input_dims)
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    batch_dim, out_spatial_dim = default_target_dict["dim_tags"]
    feat_spatial_dim = Dim(None, name="feat_spatial_dim")  # it's not the input raw audio spatial dim...

    if config:
        config = config.copy()
    else:
        config = {}
    config.setdefault("__batch_size_dependent", True)
    config.setdefault("batch_size", 10_000 * _batch_size_factor)  # grads need more mem

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_aed_model_get_input_grads_step,
        config={
            "model_outputs": {
                "output": {"dims": [batch_dim, out_spatial_dim, feat_spatial_dim], "dtype": "float32"},
                "feat_size": {"dims": [batch_dim], "dtype": "int32"},
                "targets_size": {"dims": [batch_dim], "dtype": "int32"},
                "partial_scores": {"dims": [batch_dim, out_spatial_dim], "dtype": "float32"},
            },
            **config,
        },
        forward_rqmt={"time": 24},
    )


def _aed_model_get_input_grads_step(*, model: AedModel, extern_data: TensorDict, **_kwargs):
    import torch
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    feat_spatial_dim_ = expected_output.dims[-1]

    # Call: logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
    in_spatial_dim = source.get_time_dim_tag()

    # log mel filterbank features
    source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        source,
        in_spatial_dim=in_spatial_dim,
        out_dim=model.in_dim,
        sampling_rate=16_000,
    )
    in_spatial_dim.get_size_tensor().mark_as_output("feat_size")
    feat_spatial_dim_.declare_same_as(in_spatial_dim)
    if model.feature_batch_norm:
        source = model.feature_batch_norm(source)
    if model.feature_norm:
        source = rf.normalize(source, axis=in_spatial_dim)
    if model.feature_stats:
        source = (source - model.feature_stats.mean) / model.feature_stats.std_dev

    with torch.enable_grad():
        # Just that we have well-defined dim order, for the grad logic below.
        source = source.copy_transpose((batch_dim, in_spatial_dim, model.in_dim))  # [B,T_in,D]
        source.raw_tensor.requires_grad = True

        # (No Mixup, no SpecAugment)

        # Encoder including convolutional frontend
        enc, enc_spatial_dim = model.encoder(source, in_spatial_dim=in_spatial_dim)
        enc = model.decoder.transform_encoder(enc, axis=enc_spatial_dim)

        targets_spatial_dim = targets.get_time_dim_tag()
        targets_spatial_dim.get_size_tensor().mark_as_output("targets_size")

        batch_dims = targets.remaining_dims(targets_spatial_dim)
        # Just shift right, i.e. add BOS, cut off the last,
        # because we do not need the EOS log prob.
        input_labels = rf.shift_right(targets, axis=targets_spatial_dim, pad_value=model.bos_idx)

        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_spatial_dim,
            encoder=enc,
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )
        log_prob = rf.log_softmax(logits, axis=model.target_dim)  # [B,T_out,D]
        scores = rf.cross_entropy(
            target=targets, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )  # [B,T_out]

        scores.mark_as_output("partial_scores")
        scores_ta = TensorArray.unstack(scores, axis=targets_spatial_dim)

        source_grad_mult_with_source = config.bool("source_grad_mult_with_source", False)
        grad_norm_p = config.float("grad_norm_p", 2.0)

        grad_norms = []
        for t in range(targets_spatial_dim.get_dim_value()):
            source.raw_tensor.grad = None
            scores_t = scores_ta[t]  # [B], +log prob
            partial_loss = scores_t.raw_tensor.sum()
            partial_loss.backward(retain_graph=True)
            grad: torch.Tensor = source.raw_tensor.grad  # [B,T_in,D]  # noqa
            if source_grad_mult_with_source:
                grad = grad * source.raw_tensor
            grad_norm = torch.norm(grad, p=grad_norm_p, dim=2)  # [B,T_in]
            grad_norms.append(grad_norm)
        grad_norms = torch.stack(grad_norms, dim=0)  # [T_out,B,T_in]
        grad_norms_ = rf.convert_to_tensor(grad_norms, dims=[targets_spatial_dim, batch_dim, in_spatial_dim])
        grad_norms_.mark_as_default_output()


def visualize_grad_scores():
    # to run:
    # Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
    # Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"
    # Then: `python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align import visualize_grad_scores as vis; vis()"`  # noqa
    # play around here...

    import os
    import sys
    from i6_experiments.users.schmitt.hdf import load_hdf_data
    import i6_core.util as util
    import numpy as np

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib

    font_size = 22
    matplotlib.rcParams.update(
        {"font.size": font_size, "xtick.labelsize": font_size * 0.8, "ytick.labelsize": font_size * 0.8}
    )

    def _log_softmax(x: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
        max_score = np.max(x, axis=axis, keepdims=True)
        x = x - max_score
        return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))

    for name in [
        "ctc-grad-align/base",
        "ctc-grad-align/blankSep",
        "ctc-grad-align/blankSep-blankStopGrad-inclBlankState-p0.1",
        "ctc-grad-align/base-inclBlankState-p0.1",
        "ctc-grad-align/base-spm512-blankSep-blankStopGrad-inclBlankState-p0.1",
        "aed-grad-align/base-p0.1",
    ]:

        score_matrix_hdf = Path(f"output/exp2024_09_16_grad_align/{name}/input_grads.hdf")
        score_matrix_data_dict = load_hdf_data(score_matrix_hdf, num_dims=2)
        basename_tags = {os.path.basename(tag): tag for tag in score_matrix_data_dict.keys()}

        plot_dir = "output/exp2024_09_16_grad_align/visualize_grad_scores/" + name
        os.makedirs(plot_dir, exist_ok=True)

        seq_list = Path(
            "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
        )
        seq_list = open(seq_list.get_path()).read().splitlines()

        # or alternatively:
        # seq_list = list(score_matrix_data_dict.keys())

        for i, seq_tag in enumerate(seq_list):
            if i >= 2:
                break

            if seq_tag not in score_matrix_data_dict:
                if os.path.basename(seq_tag) in basename_tags:
                    seq_tag = basename_tags[os.path.basename(seq_tag)]

            score_matrix = score_matrix_data_dict[seq_tag]  # [S, T]
            S, T = score_matrix.shape  # noqa
            print(f"{name}, seq {seq_tag}, shape (SxT) {score_matrix.shape}")

            score_matrix = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

            alias = "log softmax"
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
            # score_matrix is [S,T]
            mat_ = ax.matshow(score_matrix, cmap="Blues", aspect="auto")
            ax.tick_params(direction="out", length=20, width=2)
            # ax.set_title(f"{alias} for seq {seq_tag}")
            print(f"{alias} for seq {seq_tag}")
            ax.set_xlabel("time")
            ax.set_ylabel("labels")
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.gca().xaxis.tick_bottom()

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mat_, cax=cax, orientation="vertical")

            plt.tight_layout()
            fn = f"{plot_dir}/alignment_{seq_tag.replace('/', '_')}.pdf"
            print("save to:", fn)
            plt.savefig(fn)


def visualize_train_scores():
    # to run:
    # Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
    # Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"
    # Then: `python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align import visualize_train_scores as vis; vis()"`  # noqa
    # play around here...

    import os
    import sys
    import i6_core.util as util
    import numpy as np

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    from matplotlib import pyplot as plt

    prefix = "output/exp2024_09_16_grad_align/visualize_train_scores/"
    fig, ax = plt.subplots()

    for shortname, fullname in [
        # (  # ctc forced align: 110.7/43.7ms
        #     "noBias",  # 5.65/5.94, better baseline
        #     "v6-relPosAttDef-noBias"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-spm10k-bpeSample001",
        #     "spm10k",
        # ),
        (  # ctc forced align: 111.5/52.9ms
            "baseline",  # 5.77/6.03
            # output/ctc/v6-relPosAttDef-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001/recog_results_best
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
        ),
        (  # ctc forced align: 116.8/74.4ms
            "normed grad (0.5, 1.1, batch est)",  # 5.71/5.87
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC05_11P1",
        ),
        (
            "normed grad (0.5, 1.1, seq est)",  # 5.83/5.91
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC05_11P1Seq",
        ),
        # (
        #     "lpNormedGradC01_11P1",  # 6.21/6.55
        #     "v6-relPosAttDef"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-spm10k-bpeSample001"
        #     "-lpNormedGradC01_11P1",
        #     "spm10k",
        # ),
        (  # ctc forced align: 98.5/77.6ms
            "separated blank",  # 5.73/6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-blankSep",
        ),
        # (  # ctc forced align: 75.4/42.7ms
        #     "base-spm512",  # 5.97/6.21
        #     "v6-relPosAttDef"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-spm512-bpeSample001",
        #     "spm512",
        # ),
        # (  # ctc forced align: 59.6/48.5ms
        #     "base-spm512-blankSep",  # 6.02/6.04
        #     "v6-relPosAttDef"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-spm512-bpeSample001"
        #     "-blankSep",
        #     "spm512",
        # ),
        # (  # ctc forced align: 113.9/68.1ms
        #     "base-bpe10k",  # 6.18/6.35
        #     "v6-relPosAttDef"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-bpe10k-bpeSample001",
        #     "bpe10k",
        # ),
        # (  # ctc forced align: 84.9/64.2ms
        #     "base-bpe10k-blankSep",  # 5.98/6.13
        #     "v6-relPosAttDef"
        #     "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        #     "-featBN-speedpertV2-bpe10k-bpeSample001"
        #     "-blankSep",
        #     "bpe10k",
        # ),
        # (
        #     "ebranchformer",  # 5.54/5.69
        #     # output/ctc/v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001/recog_results_best
        #     "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
        #     "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
        #     "spm10k",
        # ),
    ]:
        scores = eval(
            open(f"alias/ctc/{fullname}/train/output/learning_rates").read(),
            {"EpochData": dict, "nan": float("nan"), "inf": float("inf")},
        )

        partition_epoch = 20
        multi_gpu = 4
        x = np.array(list(scores.keys())) / partition_epoch * multi_gpu
        y = np.array([v["error"]["dev_loss_ctc"] for v in scores.values()])
        ax.plot(x, y, label=shortname)
        # ax.set_title(f"CTC training")
        ax.set_xlabel("epoch")
        ax.set_ylabel("CTC dev loss")
        ax.legend()
        # ax.set_ylim(ax.get_ylim()[::-1])

    plt.tight_layout()
    plot_dir = prefix
    os.makedirs(plot_dir, exist_ok=True)
    fn = f"{plot_dir}/train_scores_all.pdf"
    print("save to:", fn)
    plt.savefig(fn)
