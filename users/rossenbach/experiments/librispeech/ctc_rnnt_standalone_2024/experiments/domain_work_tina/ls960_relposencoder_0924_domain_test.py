from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.report.report import GenerateReportStringJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.common import DatasetSettings, build_test_dataset_from_zip
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.bpe import build_bpe_training_datasets, get_text_lexicon, build_custom_bpe_lexicon
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.lm import get_4gram_binary_lm
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.pipeline import training, prepare_asr_model, search, ASRModel
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.storage import get_ctc_model, get_synthetic_data, get_rnnt_model, get_aed_model

def report_template(report_values):
    from i6_core.util import instanciate_delayed
    report_values = instanciate_delayed(report_values)

    string = f"Results for {report_values['corpus_name']}:\n"
    string += f"Best LM: {report_values['best_lm']}\n"
    string += f"Best Prior: {report_values['best_prior']}\n\n"

    string += f"Final WER: {report_values['best_wer']}\n"

    return string


def bpe_ls960_0924_relposencoder(lex, lm):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_test/ls960_ctc_bpe_relposencoder_0924"

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    def ctc_tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config,
                                 lm_scales, prior_scales):
        tune_parameters = []
        report_values = {}
        tune_values = []
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
                tune_values.append(list(wers.values())[0])
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
        dev_name = list(dev_dataset_tuples.keys())[0]
        report_values = {
            "corpus_name": dev_name,
            "best_lm": pick_optimal_params_job.out_optimal_parameters[0],
            "best_prior": pick_optimal_params_job.out_optimal_parameters[1],
            "best_wer": pick_optimal_params_job.out_optimal_value
        }
        report = GenerateReportStringJob(report_values=report_values, report_template=report_template,
                                         compress=False).out_report
        tk.register_output(training_name + "/%s_report.txt" % dev_name, report)


    def rnnt_evaluate_helper(
        training_name: str,
        dev_dataset_tuples,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        beam_size: int = 1,
        use_gpu=False,
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass

        """
        decoder_config = copy.deepcopy(base_decoder_config)
        decoder_config.beam_size = beam_size
        search_name = training_name + "/search_bs%i" % beam_size
        search_jobs, wers = search(
            search_name,
            forward_config= {"seed": 2} if use_gpu else {},
            asr_model=asr_model,
            decoder_module="rnnt.decoder.experimental_rnnt_decoder",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
        )

    def aed_evaluate_helper(
            training_name: str,
            dev_dataset_tuples,
            asr_model: ASRModel,
            decoder_config,
            seed: Optional[int] = None,
            use_gpu: bool = False,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_bs"
        search_jobs, wers = search(
            search_name,
            forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.beam_search_single_v1",
            decoder_args={"config": asdict(decoder_config)},
            use_gpu=use_gpu,
            debug=True,
            test_dataset_tuples={**dev_dataset_tuples},
            **default_returnn,
        )


    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    bpe_ctc_asr_model = get_ctc_model(network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8")
    med_wmt22_n2_bliss, med_wmt22_n2_oggzip = get_synthetic_data("wmt22_medline_v1_sequiturg2p_glowtts460_noise07")
    _, med_wmt22_n2_noise03_oggzip = get_synthetic_data("wmt22_medline_v1_sequiturg2p_glowtts460_noise03")
    # (dataset, bliss)
    ddt_medline_wmt22_noise07 = {"medline_wmt22_n2": (build_test_dataset_from_zip(med_wmt22_n2_oggzip, bpe_ctc_asr_model.settings), med_wmt22_n2_bliss)}
    ddt_medline_wmt22_noise03 = {"medline_wmt22_n2_noise03": (build_test_dataset_from_zip(med_wmt22_n2_noise03_oggzip, bpe_ctc_asr_model.settings), med_wmt22_n2_bliss)}


    bpe_lexicon = {
        name: build_custom_bpe_lexicon(lex, bpe_ctc_asr_model.label_datastream.codes, bpe_ctc_asr_model.label_datastream.vocab)
        for name, lex in lex.items()
    }
    default_decoder_config_bpe = DecoderConfig(
        lexicon=bpe_ctc_asr_model.lexicon,
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + "/medline_wmt22_ende_n2",
        ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )

    # decoding with changed LM only
    ufal_lm_config = DecoderConfig(
        lexicon=bpe_ctc_asr_model.lexicon,
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=lm["ufal_v1_lslex"],
        beam_threshold=14,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + "/medline_wmt22_ende_n2_ufal_lm",
        ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, ufal_lm_config,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )


    for lex_lm_key in ["ufal_v1_mixlex_v2", "ufal_v1_3more_only"]:
        # decoding with changed LM and updated lexicon
        ufal_lm_config = DecoderConfig(
            lexicon=bpe_lexicon[lex_lm_key],
            returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=lm[lex_lm_key],
            beam_threshold=14,
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/medline_wmt22_ende_n2_noise07_{lex_lm_key}",
            ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, ufal_lm_config,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/medline_wmt22_ende_n2_noise03_{lex_lm_key}",
            ddt_medline_wmt22_noise03, {}, bpe_ctc_asr_model, ufal_lm_config,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        ufal_lm_config_nols = DecoderConfig(
            lexicon=bpe_lexicon[lex_lm_key + "_nols"],
            returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=lm[lex_lm_key],
            beam_threshold=14,
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/medline_wmt22_ende_n2_noise07_{lex_lm_key}_nols",
            ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, ufal_lm_config_nols,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )


    # dev other reference
    dev_other_bliss, dev_other_oggzip = get_synthetic_data("dev-other_sequiturg2p_glowtts460_noise07")
    _, dev_other_noise06_oggzip = get_synthetic_data("dev-other_sequiturg2p_glowtts460_noise06")
    _, dev_other_noise055_oggzip = get_synthetic_data("dev-other_sequiturg2p_glowtts460_noise055")
    _, dev_other_noise05_oggzip = get_synthetic_data("dev-other_sequiturg2p_glowtts460_noise05")
    _, dev_other_noise03_oggzip = get_synthetic_data("dev-other_sequiturg2p_glowtts460_noise03")
    # (dataset, bliss)
    ddt_dev_other_noise07 = {"dev_other": (build_test_dataset_from_zip(dev_other_oggzip, bpe_ctc_asr_model.settings), dev_other_bliss)}
    ddt_dev_other_noise06 = {"dev_other": (build_test_dataset_from_zip(dev_other_noise06_oggzip, bpe_ctc_asr_model.settings), dev_other_bliss)}
    ddt_dev_other_noise055 = {"dev_other": (build_test_dataset_from_zip(dev_other_noise055_oggzip, bpe_ctc_asr_model.settings), dev_other_bliss)}
    ddt_dev_other_noise05 = {"dev_other": (build_test_dataset_from_zip(dev_other_noise05_oggzip, bpe_ctc_asr_model.settings), dev_other_bliss)}
    ddt_dev_other_noise03 = {"dev_other": (build_test_dataset_from_zip(dev_other_noise03_oggzip, bpe_ctc_asr_model.settings), dev_other_bliss)}

    ctc_tune_and_evaluate_helper(
        prefix_name + "/dev_other",
        ddt_dev_other_noise07, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/dev_other_noise06",
        ddt_dev_other_noise06, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/dev_other_noise055",
        ddt_dev_other_noise055, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/dev_other_noise05",
        ddt_dev_other_noise05, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/dev_other_noise03",
        ddt_dev_other_noise03, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )

    ##### RNNT #####
    network_module = "rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    bpe_rnnt_asr_model = get_rnnt_model(network_module + ".bpe128.512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2")
    from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder import DecoderConfig as RnntDecoderConfig

    decoder_config_bpeany_greedy = RnntDecoderConfig(
        beam_size=1,  # greedy as default
        returnn_vocab=bpe_rnnt_asr_model.returnn_vocab
    )

    rnnt_evaluate_helper(
        prefix_name + "/rnnt_bpe/dev_other_noise07",
        dev_dataset_tuples=ddt_dev_other_noise07,
        asr_model=bpe_rnnt_asr_model,
        base_decoder_config=decoder_config_bpeany_greedy,
        beam_size=12,
    )

    rnnt_evaluate_helper(
        prefix_name + "/rnnt_bpe/medline_wmt22_noise07",
        dev_dataset_tuples=ddt_medline_wmt22_noise07,
        asr_model=bpe_rnnt_asr_model,
        base_decoder_config=decoder_config_bpeany_greedy,
        beam_size=12,
    )


    ##### AED #####
    network_module = "aed.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_v1"
    bpe_aed_asr_model = get_aed_model(network_module + ".bpe5000.512dim_sub6_48gbgpu4w_100eps_highlr_bs300")
    from ...pytorch_networks.aed.decoder.beam_search_single_v1 import DecoderConfig as BeamSearchDecoderConfig, \
        BeamSearchOpts

    bs_decoder_config = BeamSearchDecoderConfig(
        returnn_vocab=bpe_aed_asr_model.label_datastream.vocab,
        beam_search_opts=BeamSearchOpts(
            beam_size=12,
            length_normalization_exponent=1.0,
            length_reward=0,
            bos_label=0,
            eos_label=0,
            num_labels=bpe_aed_asr_model.label_datastream.vocab_size
        )
    )

    aed_evaluate_helper(
        prefix_name + "/aed_bpe/dev_other_noise07",
        dev_dataset_tuples=ddt_dev_other_noise07,
        asr_model=bpe_aed_asr_model,
        decoder_config=bs_decoder_config,
        use_gpu=False,
    )
    
    aed_evaluate_helper(
        prefix_name + "/aed_bpe/medline_wmt22_noise07",
        dev_dataset_tuples=ddt_medline_wmt22_noise07,
        asr_model=bpe_aed_asr_model,
        decoder_config=bs_decoder_config,
        use_gpu=False,
    )
