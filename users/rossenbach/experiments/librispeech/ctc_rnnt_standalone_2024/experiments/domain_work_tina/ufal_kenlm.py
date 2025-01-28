from sisyphus import Path, tk
from ...default_tools import KENLM_BINARY_PATH

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lm.kenlm import KenLMplzJob, CreateBinaryLMJob

from i6_experiments.users.rossenbach.corpus.generate import CreateBlissFromTextLinesJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict

from ...data.tts.generation import create_data_lexicon, create_data_lexicon_v2, create_data_lexicon_rasr_style, create_data_lexicon_rasr_style_v2


def ufal_kenlm():
    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_work_tina"


    # Prepare for LibriSpeech
    dev_other_text = CorpusToTxtJob(get_bliss_corpus_dict(audio_format="ogg")["dev-other"]).out_txt
    tk.register_output(prefix + "/dev_other.txt", dev_other_text)

    librispeech_lexicon = get_bliss_lexicon(use_stress_marker=False, add_silence=False)

    # Full medical text version 1
    ufal_medical_version_1_text = Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz",
        hash_overwrite="UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz"
    )
    MTG_version_2_text = Path(
        "/work/asr4/rossenbach/domain_data/MTG/MTG_trial3_train.txt",
        hash_overwrite="MTG/MTG_trial3_train.txt"
    )

    ufal_kenlm_lslex_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=librispeech_lexicon,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )
    
    lex =  {
        "ufal_v1_lslex": None,  # with librispeech default lexicon is a special case, we get that from elsewhere later
    }

    lm = {
        "ufal_v1_lslex": CreateBinaryLMJob(arpa_lm=ufal_kenlm_lslex_job.out_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm,
    }

    prefix_name_data_lexicon_tuples = [
        (
            "ufal_",
            "ufal_v1_mixlex_v2",
            ufal_medical_version_1_text,
            "/UFAL_medical_shuffled/clean_version_1/vocab_2more_onlyvalid_librispeech_mix_v2.txt"
        ),
        (
            "ufal_",
            "ufal_v1_3more_only",
            ufal_medical_version_1_text,
            "/UFAL_medical_shuffled/clean_version_1/vocab_3more_onlyvalid.txt"
        ),
        (
            "MTG_",
            "MTG_trial3",
            MTG_version_2_text,
            "/MTG/MTG_trial3_lex.txt"
        )
    ]

    for seq_prefix, name, data, lexicon_path in prefix_name_data_lexicon_tuples:
        txt_lex = Path(
            "/work/asr4/rossenbach/domain_data" + lexicon_path,
            hash_overwrite=lexicon_path,
        )
        raw_bliss_lex = CreateBlissFromTextLinesJob(
            txt_lex,
            corpus_name=seq_prefix + "lexicon",
            sequence_prefix=seq_prefix
        ).out_corpus
        g2p_bliss_lex = create_data_lexicon(
            prefix=prefix + "/" + name + "/normal_lex",
            lexicon_bliss=raw_bliss_lex,
        )
        g2p_bliss_lex_nols = create_data_lexicon_v2(
            prefix=prefix + "/" + name + "/normal_lex_nols",
            lexicon_bliss=raw_bliss_lex,
        )
        rasr_with_unk_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_unk_lex",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=True
        )
        rasr_with_unk_3var_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_unk_lex_3var",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=True,
            variants=3,
        )
        rasr_with_unk_lsoverride_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_unk_lex_lsoverride",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=True,
            ls_override=True,
        )
        rasr_with_unk_3var_lsoverride_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_unk_lex_lsoverride_3var",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=True,
            variants=3,
            ls_override=True,
        )
        rasr_without_unk_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_non_unk_lex",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=False,
        )
        rasr_without_unk_lsoverride_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_non_unk_lex_lsoverride",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=False,
            ls_override=True,
        )
        rasr_without_unk_3var_bliss_lex = create_data_lexicon_rasr_style_v2(
            prefix=prefix + "/" + name + "/rasr_non_unk_lex_3var",
            lm_text_bliss=raw_bliss_lex,
            with_unknown=False,
            variants=3,
        )
        tk.register_output(prefix + f"/{name}.xml.gz", g2p_bliss_lex)
        tk.register_output(prefix + f"/{name}.rasr_with_unk.xml.gz", rasr_with_unk_bliss_lex)
        tk.register_output(prefix + f"/{name}.lsoverride.rasr_with_unk.xml.gz", rasr_with_unk_lsoverride_bliss_lex)
        tk.register_output(prefix + f"/{name}.3var.lsoverride.rasr_with_unk.xml.gz", rasr_with_unk_3var_lsoverride_bliss_lex)
        tk.register_output(prefix + f"/{name}.3var.rasr_with_unk.xml.gz", rasr_with_unk_3var_bliss_lex)
        tk.register_output(prefix + f"/{name}.rasr_without_unk.xml.gz", rasr_without_unk_bliss_lex)
        tk.register_output(prefix + f"/{name}.lsoverride.rasr_without_unk.xml.gz", rasr_without_unk_lsoverride_bliss_lex)
        tk.register_output(prefix + f"/{name}.3var.rasr_without_unk.xml.gz", rasr_without_unk_3var_bliss_lex)

        kenlm_job = KenLMplzJob(
            text=[data],
            order=4,
            interpolate_unigrams=True,
            pruning=[0, 0, 1, 1],
            vocabulary=txt_lex,
            kenlm_binary_folder=KENLM_BINARY_PATH,
            mem=12,
            time=4,
        )

        lex[name] = g2p_bliss_lex
        lex[name + "_nols"] = g2p_bliss_lex_nols
        lm[name] = CreateBinaryLMJob(arpa_lm=kenlm_job.out_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm
        tk.register_output(prefix + f"/{name}.lm.gz", kenlm_job.out_lm)


    # Legacy stuff

    ufal_v1_lex_2more_mix = Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/vocab_2more_onlyvalid_librispeech_mix_v2.txt",
        hash_overwrite="/UFAL_medical_shuffled/clean_version_1/vocab_2more_onlyvalid_librispeech_mix_v2.txt"
    )

    ufal_v1_lex_3more_only = Path(
        "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/vocab_3more_onlyvalid.txt",
        hash_overwrite="/UFAL_medical_shuffled/clean_version_1/vocab_3more_onlyvalid.txt"
    )

    ufal_v1_lex_2more_mix_raw_bliss = CreateBlissFromTextLinesJob(ufal_v1_lex_2more_mix, corpus_name="ufal_lexicon", sequence_prefix="ufal_").out_corpus
    ufal_v1_lex_2more_mix_bliss = create_data_lexicon(
        prefix=prefix + "/normal_lex",
        lexicon_bliss=ufal_v1_lex_2more_mix_raw_bliss,
    )
    tk.register_output(prefix + "/ufal_v1_lexicon_2more_mix.xml.gz", ufal_v1_lex_2more_mix_bliss)

    ufal_v1_lex_3more_raw_bliss = CreateBlissFromTextLinesJob(ufal_v1_lex_3more_only, corpus_name="ufal_lexicon", sequence_prefix="ufal_").out_corpus
    ufal_v1_lex_3more_bliss = create_data_lexicon(
        prefix=prefix + "/normal_lex",
        lexicon_bliss=ufal_v1_lex_3more_raw_bliss,
    )
    tk.register_output(prefix + "/ufal_v1_lexicon_3more.xml.gz", ufal_v1_lex_3more_bliss)

    rasr_lexicon_with_unk = create_data_lexicon_rasr_style(
        prefix=prefix + "/rasr_lex_with_unk",
        lm_text_bliss=ufal_v1_lex_2more_mix_raw_bliss,
        with_unknown=True,
    )
    tk.register_output(prefix + "/ufal_librispeech_lexicon_rasr_with_unk.xml.gz", rasr_lexicon_with_unk)

    rasr_lexicon_with_unk = create_data_lexicon_rasr_style(
        prefix=prefix + "/rasr_lex_without_unk",
        lm_text_bliss=ufal_v1_lex_2more_mix_raw_bliss,
        with_unknown=False,
    )
    tk.register_output(prefix + "/ufal_librispeech_lexicon_rasr_without_unk.xml.gz", rasr_lexicon_with_unk)



    ufal_v1_2more_mix_kenlm_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=ufal_v1_lex_2more_mix,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )
    tk.register_output(prefix + "/ufal_version1_lm2_extendevocab.gz", ufal_v1_2more_mix_kenlm_job.out_lm)
    tk.register_output(prefix + "/ufal_v1_lm_2more_mix.gz", ufal_v1_2more_mix_kenlm_job.out_lm)

    ufal_v1_3more_only_kenlm_job = KenLMplzJob(
        text=[ufal_medical_version_1_text],
        order=4,
        interpolate_unigrams=True,
        pruning=[0, 0, 1, 1],
        vocabulary=ufal_v1_lex_3more_only,
        kenlm_binary_folder=KENLM_BINARY_PATH,
        mem=12,
        time=4,
    )
    tk.register_output(prefix + "/ufal_v1_lm_3more.gz", ufal_v1_3more_only_kenlm_job.out_lm)


    return lex, lm
