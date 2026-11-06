from dataclasses import replace

from sisyphus import tk
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings

from diagnostics.qat_ffnn_scores.tree_nbest_blank_penalty import run_tree_nbest_blank_penalty
from recipe.experiments.librispeech.recognition import qat_ffnn_transducer_bpe as digital_recognition
from recipe.experiments.librispeech.recognition.memristor import qat_ffnn_transducer_bpe as recognition
from recipe.experiments.librispeech.training import qat_ffnn_transducer_bpe as training


def py():
    model = training.run(
        descriptor="qat_ffnn_transducer_full_quant_v2_lowbs",
        qat_args={
            "weight_bit_prec": 8,
            "activation_bit_prec": 8,
            "weight_dropout": 0.0,
            "weight_pruning_config": None,
        },
    )
    converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    joint_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    variants = [
        recognition.nbs_mbs1024_offline_tree_recog_variant(
            num_shards=10,
            cycles=1,
            resumable=False,
            nbest_size=20,
            blank_penalty=penalty,
        )
        for penalty in (0.9, 1.44)
    ]
    lm_variant = recognition.ilm04_elm05_tree_4gram_recog_variant(
        num_shards=10, cycles=1, resumable=False,
    )
    lm_variant = replace(
        lm_variant,
        descriptor=f"{lm_variant.descriptor}_nbs20_bp1.44",
        search_mode_params=replace(lm_variant.search_mode_params, nbest_size=20),
        blank_penalty=1.44,
    )
    # LM and no-LM candidates use different score functions: optimize separately.
    variant_groups = {"pooled": variants, "ilm0.4_elm0.5": [lm_variant]}
    for group_name, group_variants in variant_groups.items():
        # Digital uses one anchor at zero for each score function.
        anchor = group_variants[0]
        digital_search_params = anchor.search_algorithm_params
        digital_ilm_scale = 0.0
        digital_group_name = "pooled"
        if group_name != "pooled":
            # report_digital.txt: 4.9% dev-other WER at ILM/ELM 0.2/0.7
            # (tied with 0.3/0.7 at the report's precision).
            digital_ilm_scale = 0.2
            digital_group_name = "ilm0.2_elm0.7"
            digital_search_params = replace(
                digital_search_params,
                word_lm_params=replace(digital_search_params.word_lm_params, scale=0.7),
            )
        digital_variant = digital_recognition.TransducerRecogVariant(
            descriptor=(
                "tree_mbs1024_nbs20_bp0.0"
                if group_name == "pooled"
                else "tree_4gram_ilm0.2_elm0.7_nbs20_bp0.0"
            ),
            search_algorithm_params=digital_search_params,
            search_mode_params=replace(anchor.search_mode_params, batched_decoder=True, num_shards=1),
            epoch=anchor.epoch,
            ilm_scale=digital_ilm_scale,
            bpe_lstm_lm_scale=anchor.bpe_lstm_lm_scale,
            blank_penalty=0.0,
            use_cache=True,
        )
        digital_results = digital_recognition.run(
            model=model,
            variants=[digital_variant],
            corpora=["dev-other"],
        )
        # Match the current collection: both joint layers use 2prec/6range.
        _, results = recognition.run(
            model=model,
            variants=group_variants,
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=joint_converter_hardware_settings,
            joint_l2_converter_hardware_settings=joint_converter_hardware_settings,
            correction_settings=None,
            max_runs=1,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        )
        for backend_name, backend_results in (
            ("digital", digital_results),
            ("jl1_2prec_6range_jl2_2prec_6range", results),
        ):
            diagnostic_group_name = digital_group_name if backend_name == "digital" else group_name
            targets = tk.sis_graph.targets_dict
            jobs_by_corpus = {}
            for result in backend_results:
                # Follow the registered output's producer (the merge job for shards).
                output_name = f"recognition/{result.descriptor}/{result.corpus_name}/nbest.json"
                nbest_file, = targets[output_name].required_full_list
                jobs_by_corpus.setdefault(result.corpus_name, []).append(nbest_file.creator)

            for corpus_name, recognition_jobs in jobs_by_corpus.items():
                run_tree_nbest_blank_penalty(
                    recognition_jobs,
                    output_prefix=(
                        f"diagnostics/tree_nbest_blank_penalty/{diagnostic_group_name}/"
                        f"{backend_name}/{corpus_name}"
                    ),
                    preferred_penalty=0.0 if backend_name == "digital" else 1.44,
                )
