from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_core.report.report import GenerateReportStringJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.common import DatasetSettings, build_test_dataset_from_zip, build_test_dataset
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.bpe import build_bpe_training_datasets, get_text_lexicon, build_custom_bpe_lexicon
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.lm import get_4gram_binary_lm
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.pipeline import training, prepare_asr_model, search, ASRModel
from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.storage import get_ctc_model, get_synthetic_data, get_rnnt_model, get_aed_model, get_lm_model

from ... import PACKAGE

def report_template(report_values):
    from i6_core.util import instanciate_delayed
    report_values = instanciate_delayed(report_values)

    string = f"Results for {report_values['corpus_name']}:\n"
    string += f"Best LM: {report_values['best_lm']}\n"
    string += f"Best Prior: {report_values['best_prior']}\n\n"

    string += f"Final WER: {report_values['best_wer']}\n"

    return string


def create_eow_phonem_lex(rasr_bliss_lex):
    from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
    from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
    eow_lex = AddEowPhonemesToLexiconJob(rasr_bliss_lex).out_lexicon
    flashlight_lex = BlissLexiconToG2PLexiconJob(
        eow_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return flashlight_lex


def bpe_ls960_0924_relposencoder(lex, lm):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_test"

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

        for key in test_dataset_tuples.keys():
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn
            )
            report_values = {
                "corpus_name": dev_name,
                "best_lm": pick_optimal_params_job.out_optimal_parameters[0],
                "best_prior": pick_optimal_params_job.out_optimal_parameters[1],
                "best_wer": wers[training_name + "/" + key]
            }
            report = GenerateReportStringJob(report_values=report_values, report_template=report_template,
                                             compress=False).out_report
            tk.register_output(training_name + "/%s_report.txt" % key, report)


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
    eow_phon_ctc_asr_model = get_ctc_model(network_module + ".eow_phon.512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8")
    med_wmt22_n2_bliss, med_wmt22_n2_oggzip = get_synthetic_data("wmt22_medline_v1_sequiturg2p_glowtts460_noise07")
    _, med_wmt22_n2_noise03_oggzip = get_synthetic_data("wmt22_medline_v1_sequiturg2p_glowtts460_noise03")
    _, med_wmt22_n2_noise055_oggzip = get_synthetic_data("wmt22_medline_v1_sequiturg2p_glowtts460_noise055")

    med_wmt21_v2_bliss, med_wmt21_v2_noise055_oggzip = get_synthetic_data("wmt21_medline_v2_sequiturg2p_glowtts460_noise055")
    med_wmt23_v2_bliss, med_wmt23_v2_noise055_oggzip = get_synthetic_data("wmt23_medline_v2_sequiturg2p_glowtts460_noise055")

    MTG_trial4_dev_bliss, MTG_trial4_dev_oggzip = get_synthetic_data("MTG_trial4_dev_sequiturg2p_glowtts460_noise055")
    MTG_trial4_test_bliss, MTG_trial4_test_oggzip = get_synthetic_data("MTG_trial4_test_sequiturg2p_glowtts460_noise055")

    # (dataset, bliss)
    ddt_medline_wmt22_noise07 = {"medline_wmt22_n2": (build_test_dataset_from_zip(med_wmt22_n2_oggzip, bpe_ctc_asr_model.settings), med_wmt22_n2_bliss)}
    ddt_medline_wmt22_noise055 = {"medline_wmt22_n2_noise055": (build_test_dataset_from_zip(med_wmt22_n2_noise055_oggzip, bpe_ctc_asr_model.settings), med_wmt22_n2_bliss)}
    ddt_medline_wmt22_noise03 = {"medline_wmt22_n2_noise03": (build_test_dataset_from_zip(med_wmt22_n2_noise03_oggzip, bpe_ctc_asr_model.settings), med_wmt22_n2_bliss)}

    # test sets
    ddt_medline_wmt21_v2_noise055 = {"medline_wmt21_v2_noise055": (build_test_dataset_from_zip(med_wmt21_v2_noise055_oggzip, bpe_ctc_asr_model.settings), med_wmt21_v2_bliss)}
    ddt_medline_wmt23_v2_noise055 = {"medline_wmt23_v2_noise055": (build_test_dataset_from_zip(med_wmt23_v2_noise055_oggzip, bpe_ctc_asr_model.settings), med_wmt23_v2_bliss)}


    MTG_trial4_dev_noise055 = {"MTG_trial4_dev_noise055": (build_test_dataset_from_zip(MTG_trial4_dev_oggzip, bpe_ctc_asr_model.settings), MTG_trial4_dev_bliss)}
    MTG_trial4_test_noise055 = {"MTG_trial4_test_noise055": (build_test_dataset_from_zip(MTG_trial4_test_oggzip, bpe_ctc_asr_model.settings), MTG_trial4_test_bliss)}

    # Original LibriSpeech, but only optimize on dev-other
    dev_dataset_tuples = {}
    for testset in ["dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=bpe_ctc_asr_model.settings,
        )
    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=bpe_ctc_asr_model.settings,
        )


    bpe_lexicon = {
        name: build_custom_bpe_lexicon(lex, bpe_ctc_asr_model.label_datastream.codes, bpe_ctc_asr_model.label_datastream.vocab)
        for name, lex in lex.items() if "rasr" not in name
    }

    eow_phon_lexicon = {
        name: create_eow_phonem_lex(lex)
        for name, lex in lex.items() if "rasr" in name
    }

    default_decoder_config_bpe = DecoderConfig(
        lexicon=bpe_ctc_asr_model.lexicon,
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    default_decoder_config_eow_phon = DecoderConfig(
        lexicon=eow_phon_ctc_asr_model.lexicon,
        returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
        beam_size=1024,
        beam_size_token=16,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + "/bpe_ctc/medline_wmt22_ende_n2",
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
        prefix_name + "/bpe_ctc/medline_wmt22_ende_n2_ufal_lm",
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
            prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise07/{lex_lm_key}",
            ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, ufal_lm_config,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}",
            ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise03/{lex_lm_key}",
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
            prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise07/{lex_lm_key}_nols",
            ddt_medline_wmt22_noise07, {}, bpe_ctc_asr_model, ufal_lm_config_nols,
            lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        ctc_tune_and_evaluate_helper(
            prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols",
            ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config_nols,
            lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4, 2.6], prior_scales=[0.1, 0.2, 0.3, 0.4]
        )

        if lex_lm_key == "ufal_v1_3more_only":
            ctc_tune_and_evaluate_helper(
                prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_run2",
                ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config_nols,
                lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.1, 0.2, 0.3]
            )

            # Phon stuff
            ufal_lm_config_nols_eow_phon = DecoderConfig(
                lexicon=eow_phon_lexicon[lex_lm_key + "_rasr_lsoverride"],
                returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
                beam_size=1024,
                beam_size_token=16,  # makes it much faster
                arpa_lm=lm[lex_lm_key],
                beam_threshold=14,
            )
            ctc_tune_and_evaluate_helper(
                prefix_name + f"/phon_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols",
                ddt_medline_wmt22_noise055, {}, eow_phon_ctc_asr_model, ufal_lm_config_nols_eow_phon,
                lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6], prior_scales=[0.2, 0.3, 0.4]
            )

            # larger BPE search

            ufal_lm_config_nols_large_search = DecoderConfig(
                lexicon=bpe_lexicon[lex_lm_key + "_nols"],
                returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
                beam_size=2048,
                beam_size_token=32,  # makes it much faster
                arpa_lm=lm[lex_lm_key],
                beam_threshold=18,
            )
            ctc_tune_and_evaluate_helper(
                prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_large_search",
                ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config_nols_large_search,
                lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.1, 0.2, 0.3]
            )
            ctc_tune_and_evaluate_helper(
                prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_large_search2",
                ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config_nols_large_search,
                lm_scales=[3.0, 3.2, 3.4], prior_scales=[0.3, 0.4, 0.5]
            )

            ctc_tune_and_evaluate_helper(
                prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_final_v2_search",
                ddt_medline_wmt22_noise055, {**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055}, bpe_ctc_asr_model, ufal_lm_config_nols_large_search,
                lm_scales=[2.6, 2.8, 3.0, 3.2, 3.4], prior_scales=[0.2, 0.3, 0.4]
            )


            # final search Phon setup
            ufal_lm_config_nols_eow_phon_large_search = DecoderConfig(
                lexicon=eow_phon_lexicon[lex_lm_key + "_rasr_lsoverride"],
                returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
                beam_size=2048,
                beam_size_token=32,  # makes it much faster
                arpa_lm=lm[lex_lm_key],
                beam_threshold=18,
            )

            ctc_tune_and_evaluate_helper(
                prefix_name + f"/phon_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_final_v2_search",
                ddt_medline_wmt22_noise055, {**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055}, eow_phon_ctc_asr_model, ufal_lm_config_nols_eow_phon_large_search,
                lm_scales=[3.0, 3.2, 3.4, 3.6, 3.8, 4.0], prior_scales=[0.2, 0.3, 0.4, 0.5]
            )


            # # THIS WENT OOM
            #
            # ufal_lm_config_nols_ultra_large_search = DecoderConfig(
            #     lexicon=bpe_lexicon[lex_lm_key + "_nols"],
            #     returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
            #     beam_size=4096,
            #     beam_size_token=64,  # makes it much faster
            #     arpa_lm=lm[lex_lm_key],
            #     beam_threshold=20,
            # )
            # ctc_tune_and_evaluate_helper(
            #     prefix_name + f"/bpe_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_ultra_large_search",
            #     ddt_medline_wmt22_noise055, {}, bpe_ctc_asr_model, ufal_lm_config_nols_ultra_large_search,
            #     lm_scales=[3.0, 3.2, 3.4], prior_scales=[0.3, 0.4, 0.5]
            # )

            # larger phon search
            ufal_lm_config_nols_eow_phon = DecoderConfig(
                lexicon=eow_phon_lexicon[lex_lm_key + "_rasr_lsoverride"],
                returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
                beam_size=2048,
                beam_size_token=24,
                arpa_lm=lm[lex_lm_key],
                beam_threshold=18,
            )
            ctc_tune_and_evaluate_helper(
                prefix_name + f"/phon_ctc/medline_wmt22_ende_n2_noise055/{lex_lm_key}_nols_large_search",
                ddt_medline_wmt22_noise055, {}, eow_phon_ctc_asr_model, ufal_lm_config_nols_eow_phon,
                lm_scales=[2.6, 2.8, 3.0, 3.2, 3.4], prior_scales=[0.2, 0.3, 0.4]
            )


    # MTG BPE
    ls_lm_config = DecoderConfig(
        lexicon=bpe_ctc_asr_model.lexicon,
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=2048,
        beam_size_token=32,
        arpa_lm=arpa_4gram_lm,
        beam_threshold=16,
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + f"/bpe_ctc/MTG_trial4_dev_noise055_lslm",
        MTG_trial4_dev_noise055, {}, bpe_ctc_asr_model, ls_lm_config,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3]
    )

    lex_key = "MTG_trial4"
    MTG_lm_config = DecoderConfig(
        lexicon=bpe_lexicon[lex_key],
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=2048,
        beam_size_token=32,
        arpa_lm=lm[lex_key],
        beam_threshold=16,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + f"/bpe_ctc/MTG_trial4_dev_noise055/{lex_key}",
        MTG_trial4_dev_noise055, {}, bpe_ctc_asr_model, MTG_lm_config,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4], prior_scales=[0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + f"/bpe_ctc/MTG_trial4_dev_noise055/{lex_key}",
        MTG_trial4_dev_noise055, MTG_trial4_test_noise055, bpe_ctc_asr_model, MTG_lm_config,
        lm_scales=[3.2, 3.4, 3.6, 3.8, 4.0, 4.2], prior_scales=[0.3, 0.4, 0.5, 0.6, 0.7]
    )

    MTG_lm_config_large_search = DecoderConfig(
        lexicon=bpe_lexicon[lex_key],
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=4096,
        beam_size_token=64,
        arpa_lm=lm[lex_key],
        beam_threshold=18,
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + f"/bpe_ctc/MTG_trial4_dev_noise055_large_search/{lex_key}",
        MTG_trial4_dev_noise055, MTG_trial4_test_noise055, bpe_ctc_asr_model, MTG_lm_config_large_search,
        lm_scales=[3.2, 3.4, 3.6, 3.8, 4.0, 4.2], prior_scales=[0.3, 0.4, 0.5, 0.6, 0.7]
    )

    # MTG PHON

    MTG_lm_phon_config = DecoderConfig(
        lexicon=eow_phon_lexicon[lex_key + "_rasr_lsoverride"],
        returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
        beam_size=2048,
        beam_size_token=32,
        arpa_lm=lm[lex_key],
        beam_threshold=16,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + f"/phon_ctc/MTG_trial4_dev_noise055/{lex_key}",
        MTG_trial4_dev_noise055, MTG_trial4_test_noise055, eow_phon_ctc_asr_model, MTG_lm_phon_config,
        lm_scales=[3.0, 3.2, 3.4, 3.6, 3.8, 4.0], prior_scales=[0.3, 0.4, 0.5, 0.6]
    )

    MTG_lm_phon_config_large_search = DecoderConfig(
        lexicon=eow_phon_lexicon[lex_key + "_rasr_lsoverride"],
        returnn_vocab=eow_phon_ctc_asr_model.returnn_vocab,
        beam_size=4096,
        beam_size_token=32,
        arpa_lm=lm[lex_key],
        beam_threshold=18,
    )

    ctc_tune_and_evaluate_helper(
        prefix_name + f"/phon_ctc/MTG_trial4_dev_noise055/{lex_key}_bs4k_thresh18",
        MTG_trial4_dev_noise055, MTG_trial4_test_noise055, eow_phon_ctc_asr_model, MTG_lm_phon_config_large_search,
        lm_scales=[3.0, 3.2, 3.4, 3.6, 3.8, 4.0], prior_scales=[0.3, 0.4, 0.5, 0.6]
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
        prefix_name + "/bpe_ctc/dev_other_noise07",
        ddt_dev_other_noise07, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/bpe_ctc/dev_other_noise06",
        ddt_dev_other_noise06, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/bpe_ctc/dev_other_noise055",
        ddt_dev_other_noise055, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/bpe_ctc/dev_other_noise05",
        ddt_dev_other_noise05, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/bpe_ctc/dev_other_noise03",
        ddt_dev_other_noise03, {}, bpe_ctc_asr_model, default_decoder_config_bpe,
        lm_scales=[1.6, 1.8, 2.0, 2.2], prior_scales=[0.1, 0.2, 0.3, 0.4]
    )


    # Phoneme plot
    ctc_tune_and_evaluate_helper(
        prefix_name + "/phon_ctc/dev_other_noise03",
        ddt_dev_other_noise03, {}, eow_phon_ctc_asr_model, default_decoder_config_eow_phon,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/phon_ctc/dev_other_noise05",
        ddt_dev_other_noise05, {}, eow_phon_ctc_asr_model, default_decoder_config_eow_phon,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/phon_ctc/dev_other_noise055",
        ddt_dev_other_noise055, {}, eow_phon_ctc_asr_model, default_decoder_config_eow_phon,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/phon_ctc/dev_other_noise06",
        ddt_dev_other_noise06, {}, eow_phon_ctc_asr_model, default_decoder_config_eow_phon,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.2, 0.3, 0.4]
    )
    ctc_tune_and_evaluate_helper(
        prefix_name + "/phon_ctc/dev_other_noise07",
        ddt_dev_other_noise07, {}, eow_phon_ctc_asr_model, default_decoder_config_eow_phon,
        lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0], prior_scales=[0.2, 0.3, 0.4]
    )


    #### CTC with Beam Search Decoding and LSTM LM ####
    from ...tune_eval import tune_and_evaluate_helper
    from ...storage import NeuralLM
    lstm_2x1024_domainshift: NeuralLM = get_lm_model("bpe%i_2x1024_kazuki_lstmlm_3ep" % 128)
    lstm_2x1024_UFAL: NeuralLM = get_lm_model("ufal_bpe%i_2x1024_kazuki_lstmlm_12ep" % 128)
    lstm_2x1024_MTG: NeuralLM = get_lm_model("mtg_bpe%i_2x1024_kazuki_lstmlm_100ep" % 128)

    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import DecoderConfig as CTCDecoderConfig, DecoderExtraConfig
    extra_config = DecoderExtraConfig(
        lm_package=PACKAGE
    )

    decoder_config_ctc_lstmlm = CTCDecoderConfig(
        returnn_vocab=bpe_ctc_asr_model.returnn_vocab,
        beam_size=32,
        lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v3.Model",
        lm_model_args=lstm_2x1024_domainshift.net_args,
        lm_checkpoint=lstm_2x1024_domainshift.checkpoint,
        lm_states_need_label_axis=False,
        prior_scale=0.1,
        prior_file=bpe_ctc_asr_model.prior_file,
        lm_scale=1.0,
    )

    decoder_config_ctc_ufal_lstmlm = copy.deepcopy(decoder_config_ctc_lstmlm)
    decoder_config_ctc_ufal_lstmlm.lm_model_args = lstm_2x1024_UFAL.net_args
    decoder_config_ctc_ufal_lstmlm.lm_checkpoint = lstm_2x1024_UFAL.checkpoint

    decoder_config_ctc_mtg_lstmlm = copy.deepcopy(decoder_config_ctc_lstmlm)
    decoder_config_ctc_mtg_lstmlm.lm_model_args = lstm_2x1024_MTG.net_args
    decoder_config_ctc_mtg_lstmlm.lm_checkpoint = lstm_2x1024_MTG.checkpoint

    decoder_config_ctc_lstmlm_large_search = copy.deepcopy(decoder_config_ctc_lstmlm)
    decoder_config_ctc_lstmlm_large_search.beam_size = 128

    decoder_config_ctc_ufal_lstmlm_large_search = copy.deepcopy(decoder_config_ctc_ufal_lstmlm)
    decoder_config_ctc_ufal_lstmlm_large_search.beam_size = 128

    decoder_config_ctc_mtg_lstmlm_large_search = copy.deepcopy(decoder_config_ctc_mtg_lstmlm)
    decoder_config_ctc_mtg_lstmlm_large_search.beam_size = 128

    fixed_arguments = {
        "asr_model": bpe_ctc_asr_model,
        "default_returnn": default_returnn,
        "unhashed_decoder_config": extra_config,
        "use_gpu": True,
        "extra_rqmt": None,
        "extra_forward_config": {"batch_size": 200 * 16000},
    }
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/dev-other_lstmlm",
        base_decoder_config=decoder_config_ctc_lstmlm,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        lm_scales=[0.60, 0.70, 0.80, 0.90, 1.0, 1.1],
        prior_scales=[0.16, 0.20, 0.24, 0.28, 0.32],
        **fixed_arguments
    )
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/medline_ufal_lstmlm",
        base_decoder_config=decoder_config_ctc_ufal_lstmlm,
        dev_dataset_tuples=ddt_medline_wmt22_noise055,
        test_dataset_tuples={**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055},
        lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        prior_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **fixed_arguments
    )
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/mtg_mtg_lstmlm",
        base_decoder_config=decoder_config_ctc_mtg_lstmlm,
        dev_dataset_tuples=MTG_trial4_dev_noise055,
        test_dataset_tuples=MTG_trial4_test_noise055,
        lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        prior_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **fixed_arguments
    )

    # Beam 128
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/dev-other_lstmlm_large_search",
        base_decoder_config=decoder_config_ctc_lstmlm_large_search,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        lm_scales=[0.80, 0.90, 1.0, 1.1, 1.2],
        prior_scales=[0.32, 0.36, 0.40, 0.44, 0.48],
        **fixed_arguments
    )
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/dev-other-noise055_lstmlm_large_search",
        base_decoder_config=decoder_config_ctc_lstmlm_large_search,
        dev_dataset_tuples=ddt_dev_other_noise055,
        test_dataset_tuples={},
        lm_scales=[0.90, 1.0, 1.1, 1.2, 1.3],
        prior_scales=[0.36, 0.40, 0.44, 0.48, 0.52],
        **fixed_arguments
    )
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/medline_ufal_lstmlm_large_search",
        base_decoder_config=decoder_config_ctc_ufal_lstmlm_large_search,
        dev_dataset_tuples=ddt_medline_wmt22_noise055,
        test_dataset_tuples={**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055},
        lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        prior_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **fixed_arguments
    )
    tune_and_evaluate_helper(
        training_name=prefix_name + "/ctc_bpe/mtg_mtg_lstmlm_large_search",
        base_decoder_config=decoder_config_ctc_mtg_lstmlm_large_search,
        dev_dataset_tuples=MTG_trial4_dev_noise055,
        test_dataset_tuples=MTG_trial4_test_noise055,
        lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        prior_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **fixed_arguments
    )





    ##### RNNT #####
    network_module = "rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    bpe_rnnt_asr_model = get_rnnt_model(network_module + ".bpe128.512dim_sub6_24gbgpu_100eps_accum1_gradclip_fullspec11_sp_morel2_centerLR")
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
        prefix_name + "/rnnt_bpe/dev_other_noise055",
        dev_dataset_tuples=ddt_dev_other_noise055,
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

    # Experiments with LM
    # from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v8 import DecoderConfig

    # lm = get_lm_model("ufal_bpe%i_2x1024_kazuki_lstmlm_12ep" % BPE_SIZE)

    # decoder_config_medline_lm = DecoderConfig(
    #     returnn_vocab=bpe_rnnt_asr_model.returnn_vocab,
    #     beam_size=32,
    #     lm_module=lm.network_module,
    #     lm_model_args=lm.net_args,
    #     lm_checkpoint=lm.checkpoint,
    #     lm_states_need_label_axis=False,
    # )
    from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v9 import DecoderConfig as DecoderConfigV9, \
        ExtraConfig as ExtraConfigV9

    extra_config = ExtraConfigV9(
        lm_package=PACKAGE,
    )

    decoder_config_domainshift_ls_lstmlm = DecoderConfigV9(
        beam_size=32,
        returnn_vocab=bpe_rnnt_asr_model.returnn_vocab,
        lm_model_args=lstm_2x1024_domainshift.net_args,
        lm_checkpoint=lstm_2x1024_domainshift.checkpoint,
        lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v3.Model",
        lm_scale=0.2,
        zero_ilm_scale=0.1,
        lm_states_need_label_axis=False,
        lm_max_state_length=None,
        max_token_per_frame = 100,
    )

    decoder_config_domainshift_ufal_lstmlm = copy.deepcopy(decoder_config_domainshift_ls_lstmlm)
    decoder_config_domainshift_ufal_lstmlm.lm_model_args = lstm_2x1024_UFAL.net_args
    decoder_config_domainshift_ufal_lstmlm.lm_checkpoint = lstm_2x1024_UFAL.checkpoint

    decoder_config_domainshift_mtg_lstmlm = copy.deepcopy(decoder_config_domainshift_ls_lstmlm)
    decoder_config_domainshift_mtg_lstmlm.lm_model_args = lstm_2x1024_MTG.net_args
    decoder_config_domainshift_mtg_lstmlm.lm_checkpoint = lstm_2x1024_MTG.checkpoint


    fixed_arguments = {
        "asr_model": bpe_rnnt_asr_model,
        "default_returnn": default_returnn,
        "unhashed_decoder_config": extra_config,
        "use_gpu": True,
        "extra_rqmt": None,
        "extra_forward_config": {"batch_size": 200 * 16000},
    }

    tune_and_evaluate_helper(
        training_name=prefix_name + "/rnnt_bpe/dev-other_lstmlm",
        base_decoder_config=decoder_config_domainshift_ls_lstmlm,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        lm_scales=[0.45, 0.50, 0.55, 0.60, 0.65],
        prior_scales=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        **fixed_arguments
    )

    tune_and_evaluate_helper(
        prefix_name + "/rnnt_bpe/dev-other_noise0.55_lstmlm",
        base_decoder_config=decoder_config_domainshift_ls_lstmlm,
        dev_dataset_tuples=ddt_dev_other_noise055,
        test_dataset_tuples={},
        lm_scales=[0.45, 0.50, 0.55, 0.60, 0.65],
        prior_scales=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        **fixed_arguments,
    )

    tune_and_evaluate_helper(
        prefix_name + "/rnnt_bpe/medline_ls_lstmlm",
        base_decoder_config=decoder_config_domainshift_ls_lstmlm,
        dev_dataset_tuples=ddt_medline_wmt22_noise055,
        test_dataset_tuples={**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055},
        lm_scales=[0.50, 0.60, 0.70, 0.80, 0.90],
        prior_scales=[0.04, 0.08, 0.12, 0.16],
        **fixed_arguments,
    )

    tune_and_evaluate_helper(
        prefix_name + "/rnnt_bpe/medline_ufal_lstmlm",
        base_decoder_config=decoder_config_domainshift_ufal_lstmlm,
        dev_dataset_tuples=ddt_medline_wmt22_noise055,
        test_dataset_tuples={**ddt_medline_wmt21_v2_noise055, **ddt_medline_wmt23_v2_noise055},
        lm_scales=[0.80, 0.90, 1.0, 1.10, 1.20, 1.30],
        prior_scales=[0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
        **fixed_arguments,
    )

    tune_and_evaluate_helper(
        prefix_name + "/rnnt_bpe/mtg_ls_lstmlm",
        base_decoder_config=decoder_config_domainshift_ls_lstmlm,
        dev_dataset_tuples=MTG_trial4_dev_noise055,
        test_dataset_tuples=MTG_trial4_test_noise055,
        lm_scales=[0.50, 0.60, 0.70, 0.80, 0.90],
        prior_scales=[0.04, 0.08, 0.12, 0.16],
        **fixed_arguments,
    )

    tune_and_evaluate_helper(
        prefix_name + "/rnnt_bpe/mtg_mtg_lstmlm",
        base_decoder_config=decoder_config_domainshift_mtg_lstmlm,
        dev_dataset_tuples=MTG_trial4_dev_noise055,
        test_dataset_tuples=MTG_trial4_test_noise055,
        lm_scales=[0.80, 0.90, 1.0, 1.10, 1.20, 1.30],
        prior_scales=[0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
        **fixed_arguments,
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
