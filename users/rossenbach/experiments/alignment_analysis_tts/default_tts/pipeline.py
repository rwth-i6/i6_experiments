from sisyphus import tk
from i6_core.returnn import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
#  from i6_core.tools.graph import MultiJobCleanupJob
from i6_experiments.users.rossenbach.tools.graph import MultiJobCleanupJob
from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus
from i6_core.corpus.segments import SegmentCorpusJob
from i6_core.corpus.transform import MergeCorporaJob

from i6_experiments.users.rossenbach.tts.verify_corpus import VerifyCorpus
from i6_experiments.users.rossenbach.datasets.librispeech import get_corpus_object_dict

from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.evaluation.asr_swer_evaluation import asr_evaluation

from .config import get_forward_config, get_speaker_extraction_config
from .data import TTSForwardData


def tts_training(config, returnn_exe, returnn_root, prefix, num_epochs=200, mem=16):
    """

    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :param num_epochs:
    :return:
    """
    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=120,
        mem_rqmt=mem,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)

    return train_job


def tts_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
    """

    :param checkpoint:
    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :return:
    """
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=[],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    forward_job.add_alias(prefix + "/forward")

    return forward_job


def extract_speaker_embedding_hdf(checkpoint, returnn_common_root, returnn_exe, returnn_root, datasets, prefix, network_args):
    """

    :param returnn_common_root:
    :param returnn_exe:
    :param returnn_root:
    :param datasets:
    :param prefix:
    :param train_job:
    :return:
    """

    extraction_config = get_speaker_extraction_config(
        speaker_embedding_size=256,
        returnn_common_root=returnn_common_root,
        forward_dataset=TTSForwardData(
            dataset=datasets.cv, datastreams=datasets.datastreams  # cv is fine here cause we assume all speakers in cv
        ),
        **network_args,
    )
    extraction_job = tts_forward(
        checkpoint=checkpoint,
        config=extraction_config,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        prefix=prefix + "/extract_speak_emb",
    )
    speaker_embedding_hdf = extraction_job.out_default_hdf
    return speaker_embedding_hdf


def synthesize_with_splits(
        name,
        reference_corpus: tk.Path,
        corpus_name: str,
        job_splits: int,
        datasets: TTSForwardData,
        returnn_root,
        returnn_exe,
        returnn_common_root,
        checkpoint,
        tts_model_kwargs,
        vocoder,
        batch_size: int = 4000,
        peak_normalization: bool = True,
):
    """

    :param name:
    :param reference_corpus: Needs to be the matching corpus to recover the transcription
    :param corpus_name: Name of the corpus for the ReplaceOrthJob
    :param job_splits: number of splits performed
    :param datasets: datasets including datastream supposed to hold the audio data in .train
    :param returnn_root:
    :param returnn_exe:
    :param returnn_common_root:
    :param checkpoint:
    :param vocoder:
    :param tts_model_kwargs: kwargs to be passed to the tts model for synthesis
    :return:
    """
    forward_segments = SegmentCorpusJob(reference_corpus, job_splits)

    verifications = []
    output_corpora = []
    for i in range(job_splits):
        split_name = name + "/synth_corpus/part_%i" % i
        forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=datasets,
            batch_size=batch_size,
            **tts_model_kwargs,
        )
        forward_config.config["eval"]["datasets"]["audio"][
            "segment_file"
        ] = forward_segments.out_single_segment_files[i + 1]

        last_forward_job = ReturnnForwardJob(
            model_checkpoint=checkpoint,
            returnn_config=forward_config,
            hdf_outputs=[],
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
        )
        last_forward_job.set_keep_value(20)
        last_forward_job.add_alias(split_name + "/forward")
        forward_hdf = last_forward_job.out_hdf_files["output.hdf"]
        tk.register_output(split_name + "/foward.hdf", forward_hdf)

        forward_vocoded, vocoder_forward_job = vocoder.vocode(
            forward_hdf, iterations=30, cleanup=True, name=split_name, peak_normalization=peak_normalization
        )
        tk.register_output(split_name + "/synthesized_corpus.xml.gz", forward_vocoded)
        output_corpora.append(forward_vocoded)
        verification = VerifyCorpus(forward_vocoded).out
        verifications.append(verification)

        cleanup = MultiJobCleanupJob(
            [last_forward_job.out_default_hdf, vocoder_forward_job.out_default_hdf], verification, mode=MultiJobCleanupJob.CleanupMode.output_folder_only
        )
        tk.register_output(
            split_name + "/cleanup/cleanup.log", cleanup.out
        )

    from i6_core.corpus.transform import MergeStrategy

    merge_job = MergeCorporaJob(
        output_corpora, corpus_name, merge_strategy=MergeStrategy.FLAT
    )
    for verfication in verifications:
        merge_job.add_input(verfication)

    cv_synth_corpus = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=merge_job.out_merged_corpus,
        reference_bliss_corpus=reference_corpus,
    ).out_corpus

    tk.register_output(name + "/synth_corpus/synthesized_corpus.xml.gz", cv_synth_corpus)
    return cv_synth_corpus


def gl_swer(name, vocoder, checkpoint, config, returnn_root, returnn_exe):
    """
    Griffin Lin synthetic WER, using the librispeech-100-speaker-dev corpus

    :param name:
    :param vocoder:
    :param checkpoint:
    :param config:
    :param returnn_root:
    :param returnn_exe:
    :return:
    """
    forward_job = tts_forward(
        checkpoint=checkpoint,
        config=config,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        prefix=name,
    )
    forward_hdf = forward_job.out_hdf_files["output.hdf"]
    forward_vocoded, vocoder_forward_job = vocoder.vocode(
        forward_hdf, iterations=30, cleanup=True, name=name
    )

    verification = VerifyCorpus(forward_vocoded).out
    cleanup = MultiJobCleanupJob(
        [forward_job.out_default_hdf, vocoder_forward_job.out_default_hdf], verification, mode=MultiJobCleanupJob.CleanupMode.output_folder_only
    )
    tk.register_output(name + "/ctc_model/cleanup.log", cleanup.out)

    corpus_object_dict = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )
    cv_synth_corpus_job = CorpusReplaceOrthFromReferenceCorpus(
        forward_vocoded, corpus_object_dict["train-clean-100"].corpus_file
    )
    cv_synth_corpus_job.add_alias(name + "/speaker-dev-synth")
    cv_synth_corpus_job.add_input(verification)
    cv_synth_corpus = cv_synth_corpus_job.out_corpus
    librispeech_trafo = tk.Path(
        "/u/rossenbach/experiments/librispeech_tts/config/evaluation/asr/pretrained_configs/trafo.specaug4.12l.ffdim4."
        "pretrain3.natctc_recognize_pretrained.config"
    )
    asr_evaluation(
        config_file=librispeech_trafo,
        corpus=cv_synth_corpus,
        output_path=name,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
    )
