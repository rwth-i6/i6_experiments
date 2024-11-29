from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.common import DatasetSettings, build_test_dataset_from_zip
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.bpe import build_bpe_training_datasets, get_text_lexicon, build_custom_bpe_lexicon
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.lm import get_4gram_binary_lm
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.pipeline import training, prepare_asr_model, search, ASRModel
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.storage import get_ctc_model, get_synthetic_data
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.report import tune_and_evalue_report


def bpe_ls960_0924_relposencoder(lex, lm):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_test/ls960_ctc_bpe_relposencoder_0924"

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config,
                                 lm_scales, prior_scales):
        tune_parameters = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))


    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig,
            dev_dataset_tuples,
            test_dataset_tuples,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    asr_model = get_ctc_model(network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8")
    med_wmt22_n2_bliss, med_wmt22_n2_oggzip = get_synthetic_data("medical_medline_test_number_two")
    # (dataset, bliss)
    dev_dataset_tuples = {"medline_wmt22_n2": (build_test_dataset_from_zip(med_wmt22_n2_oggzip, asr_model.settings), med_wmt22_n2_bliss)}


    bpe_lexicon = build_custom_bpe_lexicon(lex, asr_model.label_datastream.codes, asr_model.label_datastream.vocab)
    default_decoder_config_bpe = DecoderConfig(
        lexicon=asr_model.lexicon,
        returnn_vocab=asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    tune_and_evaluate_helper(
        prefix_name + "/medline_wmt22_ende_n2",
        dev_dataset_tuples, {}, asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )

    # decoding with changed LM only
    ufal_lm_config = DecoderConfig(
        lexicon=asr_model.lexicon,
        returnn_vocab=asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=lm["ufal_v1_lslex"],
        beam_threshold=14,
    )

    tune_and_evaluate_helper(
        prefix_name + "/medline_wmt22_ende_n2_ufal_lm",
        dev_dataset_tuples, {}, asr_model, ufal_lm_config,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )

    # decoding with changed LM and updated lexicon
    ufal_lm_config = DecoderConfig(
        lexicon=bpe_lexicon,
        returnn_vocab=asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=lm["ufal_v1_mixlex_v2"],
        beam_threshold=14,
    )

    tune_and_evaluate_helper(
        prefix_name + "/medline_wmt22_ende_n2_ufal_lm_mixlex",
        dev_dataset_tuples, {}, asr_model, ufal_lm_config,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )


    # dev other reference
    dev_other_bliss, dev_other_oggzip = get_synthetic_data("dev-other")
    _, dev_other_noise03_oggzip = get_synthetic_data("dev-other-noise03")
    # (dataset, bliss)
    dev_dataset_tuples = {"dev_other": (build_test_dataset_from_zip(dev_other_oggzip, asr_model.settings), dev_other_bliss)}
    dev_dataset_tuples_noise03 = {"dev_other": (build_test_dataset_from_zip(dev_other_noise03_oggzip, asr_model.settings), dev_other_bliss)}

    tune_and_evaluate_helper(
        prefix_name + "/dev_other",
        dev_dataset_tuples, {}, asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )

    tune_and_evaluate_helper(
        prefix_name + "/dev_other_noise03",
        dev_dataset_tuples_noise03, {}, asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )