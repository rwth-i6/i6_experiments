from sisyphus import Path, tk
from ...default_tools import KENLM_BINARY_PATH

from i6_core.lm.kenlm import KenLMplzJob, CreateBinaryLMJob

from i6_experiments.users.rossenbach.corpus.generate import CreateBlissFromTextLinesJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

from ...data.tts.generation import create_data_lexicon, create_data_lexicon_rasr_style


def ufal_kenlm():
    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_work_tina"
    ufal_medical_version_1_text = Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz",
        hash_overwrite="UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz"
    )

    ufal_medical_version_1_lexicon = Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/vocab_2more_onlyvalid_librispeech_mix_v2.txt",
        hash_overwrite="/UFAL_medical_shuffled/clean_version_1/vocab_2more_onlyvalid_librispeech_mix_v2.txt"
    )

    lexicon_bliss = CreateBlissFromTextLinesJob(ufal_medical_version_1_lexicon, corpus_name="ufal_lexicon", sequence_prefix="ufal_").out_corpus
    lexicon = create_data_lexicon(
        prefix=prefix + "/normal_lex",
        lm_text_bliss=lexicon_bliss,
    )
    tk.register_output(prefix + "/ufal_librispeech_lexicon.xml.gz", lexicon)

    rasr_lexicon_with_unk = create_data_lexicon_rasr_style(
        prefix=prefix + "/rasr_lex_with_unk",
        lm_text_bliss=lexicon_bliss,
        with_unknown=True,
    )
    tk.register_output(prefix + "/ufal_librispeech_lexicon_rasr_with_unk.xml.gz", rasr_lexicon_with_unk)

    rasr_lexicon_with_unk = create_data_lexicon_rasr_style(
        prefix=prefix + "/rasr_lex_without_unk",
        lm_text_bliss=lexicon_bliss,
        with_unknown=False,
    )
    tk.register_output(prefix + "/ufal_librispeech_lexicon_rasr_without_unk.xml.gz", rasr_lexicon_with_unk)

    ufal_kenlm_lslex_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=get_bliss_lexicon(use_stress_marker=False, add_silence=False),
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )

    ufal_kenlm_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=ufal_medical_version_1_lexicon,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )

    ufal_kenlm_intermediate_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=ufal_medical_version_1_lexicon,
        compute_intermediate=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )

    ls_kenlm_intermediate_job = KenLMplzJob(
        text=[get_librispeech_normalized_lm_data()],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=ufal_medical_version_1_lexicon,
        compute_intermediate=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )

    some_ls_kenlm_intermediate_job = KenLMplzJob(
        text=[get_librispeech_normalized_lm_data()],
        order=5,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1, 1],
        vocabulary=ufal_medical_version_1_lexicon,
        compute_intermediate=True,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )


    tk.register_output(prefix + "/ufal_version1_lm2_extendevocab.gz", ufal_kenlm_job.out_lm)
    tk.register_output(prefix + "/ufal_version1_lm1.intermediate.gz", ufal_kenlm_intermediate_job.out_lm)
    tk.register_output(prefix + "/librispeech.intermediate.gz", ls_kenlm_intermediate_job.out_lm)

    tk.register_output("test.gz", some_ls_kenlm_intermediate_job.out_lm)
    return lexicon, {
        "ufal_v1_lslex": CreateBinaryLMJob(arpa_lm=ufal_kenlm_lslex_job.out_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm,
        "ufal_v1_mixlex_v2": CreateBinaryLMJob(arpa_lm=ufal_kenlm_job.out_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm,
    }
