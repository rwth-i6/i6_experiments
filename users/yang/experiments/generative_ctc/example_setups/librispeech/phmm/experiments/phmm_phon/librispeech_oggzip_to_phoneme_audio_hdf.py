from sisyphus import tk

from ...text_to_phoneme_jobs import OggZipTextToPhonemeHDFJob


def eow_phon_phmm_ls960_oggzip_to_phoneme_audio_hdf():
    prefix_name = (
        "example_setups/librispeech/phmm_standalone_2024/"
        "ls960_oggzip_to_phoneme_audio_hdf"
    )

    train_oggzip = tk.Path(
        "/u/zyang/setups/mini/work/i6_core/returnn/oggzip/"
        "BlissToOggZipJob.mNyFEqof29Q5/output/out.ogg.zip"
    )
    no_eow_lexicon = tk.Path(
        "/work/asr4/zyang/corpora/librispeech/960/lexicon/"
        "phmm_no_eow_special_phonemes.lexicon.xml.gz"
    )

    conversion_job = OggZipTextToPhonemeHDFJob(
        oggzip=train_oggzip,
        lexicon=no_eow_lexicon,
        output_filename="train_960_phoneme_text_no_eow.hdf",
        dump_audio_hdf=False,
        audio_output_filename="train_960_audio.hdf",
        unknown_word_mode="error",
        strip_eow=True,
        progress_interval=10_000,
        mem_rqmt=48,
        time_rqmt=24,
    )
    conversion_job.add_alias(prefix_name + "/train_960_no_eow_with_audio")
    tk.register_output(prefix_name + "/train_960_phoneme_text_no_eow.hdf", conversion_job.out_hdf)
    tk.register_output(prefix_name + "/train_960_phoneme_vocab.txt", conversion_job.out_vocab)
    tk.register_output(prefix_name + "/train_960_conversion_stats.txt", conversion_job.out_stats)


py = eow_phon_phmm_ls960_oggzip_to_phoneme_audio_hdf
