from typing import Callable, Dict, Tuple
from i6_core import corpus
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.meta.system import CorpusObject
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.berger import helpers
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdf
from i6_experiments.users.berger.helpers.rasr import (
    SeparatedCorpusHDFFiles,
    SeparatedCorpusObject,
)
from i6_experiments.users.berger.recipe.converse.data import (
    EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob,
    EnhancedMeetingDataToSplitBlissCorporaJob,
)
from i6_experiments.users.berger.recipe.lexicon.modification import (
    DeleteEmptyOrthJob,
    EnsureSilenceFirstJob,
    MakeBlankLexiconJob,
)
from i6_experiments.users.berger.recipe.converse.data import (
    EnhancedEvalDataToBlissCorpusJob,
)
from i6_experiments.users.berger.corpus.librispeech.lm_data import get_lm
from sisyphus import tk


def _get_hdf_files(
    *,
    gmm_system: GmmSystem,
    name: str,
    json_database: tk.Path,
    map_enhanced_audio_paths: Callable,
    map_mix_audio_paths: Callable,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    corpus_duration: float,
    concurrent: int,
    audio_format: str = "wav",
) -> SeparatedCorpusHDFFiles:
    bliss_corpora_job = EnhancedMeetingDataToSplitBlissCorporaJob(
        json_database=json_database,
        enhanced_audio_path_mapping=map_enhanced_audio_paths,
        mix_audio_path_mapping=map_mix_audio_paths,
        hash_audio_path_mapping=True,
        dataset_name=name,
    )

    sep_corpus_object = SeparatedCorpusObject(
        primary_corpus_file=bliss_corpora_job.out_bliss_corpus_primary,
        secondary_corpus_file=bliss_corpora_job.out_bliss_corpus_secondary,
        mix_corpus_file=bliss_corpora_job.out_bliss_corpus_mix,
        duration=corpus_duration,
        audio_format=audio_format,
    )

    gt_args = get_feature_extraction_args_16kHz()["gt"]

    feature_hdfs = {}
    for suffix, corpus_object in [
        ("primary", sep_corpus_object.get_primary_corpus_object()),
        ("secondary", sep_corpus_object.get_secondary_corpus_object()),
        ("mix", sep_corpus_object.get_mix_corpus_object()),
    ]:
        feature_hdfs[suffix] = build_rasr_feature_hdf(
            corpus=corpus_object,
            split=concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
        )

    alignments = gmm_system.outputs["train-other-960"]["final"].alignments.alternatives["bundle"]
    allophone_file = gmm_system.outputs["train-other-960"][
        "final"
    ].crp.acoustic_model_post_config.allophones.add_from_file
    state_tying_file = DumpStateTyingJob(gmm_system.outputs["train-other-960"]["final"].crp).out_state_tying

    alignment_hdf_job = EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob(
        dataset_name=name,
        json_database=json_database,
        feature_hdf=feature_hdfs["primary"],
        alignment_cache=alignments,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file,
        returnn_root=returnn_root,
    )
    alignment_hdf_job.rqmt.update({"mem": 10, "time": 24})

    all_segments = corpus.SegmentCorpusJob(bliss_corpora_job.out_bliss_corpus_primary, 1).out_single_segment_files[1]

    return SeparatedCorpusHDFFiles(
        primary_features_file=feature_hdfs["primary"],
        secondary_features_file=feature_hdfs["secondary"],
        alignments_file=alignment_hdf_job.out_hdf_file,
        segments=all_segments,
    )


def map_audio_paths_tfgridnet(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/data/rwth_train/enhanced_tfgridnet/",
        "/work/asr4/vieting/setups/converse/data/thilo_20230721_enhanced_tfgridnet/enhanced_tfgridnet/",
    )


def map_audio_paths_libricss_tfgridnet(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/experiments/rwth/19/evaluation/ckpt_120000_1/",
        "/work/asr4/vieting/setups/converse/data/20230727_libri_css_tvn_rwth_19_120000_1/",
    )


def map_audio_paths_blstm(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/data/rwth_train/enhanced/",
        "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/",
    )


def get_hdf_files(
    *,
    gmm_system: GmmSystem,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
) -> Dict[str, SeparatedCorpusHDFFiles]:
    return {
        "enhanced_tfgridnet_v0": _get_hdf_files(
            gmm_system=gmm_system,
            name="train_960",
            json_database=tk.Path(
                "/work/asr4/vieting/setups/converse/data/thilo_20230721_enhanced_tfgridnet/enhanced_tfgridnet/database.json",
                hash_overwrite="tfgridnet_json_database_v0",
            ),
            map_enhanced_audio_paths=map_audio_paths_tfgridnet,
            map_mix_audio_paths=map_audio_paths_tfgridnet,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            corpus_duration=1025.5,
            concurrent=200,
            audio_format="wav",
        ),
        "enhanced_blstm_v0": _get_hdf_files(
            gmm_system=gmm_system,
            name="train_960",
            json_database=tk.Path(
                "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/database.json",
                hash_overwrite="blstm_json_database_v0",
            ),
            map_enhanced_audio_paths=map_audio_paths_blstm,
            map_mix_audio_paths=map_audio_paths_blstm,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            corpus_duration=1025.5,
            concurrent=200,
            audio_format="wav",
        ),
    }


def get_eval_corpus_object_dict() -> Dict[str, CorpusObject]:
    corpus_object_dict = {}

    sub_corpora_prim = []
    sub_corpora_sec = []
    sub_corpora_mix = []
    for dataset_name in [
        "0S_segments",
        "0L_segments",
        "OV10_segments",
        "OV20_segments",
        "OV30_segments",
        "OV40_segments",
    ]:
        job = EnhancedEvalDataToBlissCorpusJob(
            json_database=tk.Path(
                "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1.json",
                hash_overwrite="libri_css_tfgridnet_json_database_v0",
            ),
            audio_path_mapping=map_audio_paths_libricss_tfgridnet,
            hash_audio_path_mapping=True,
            dataset_name=dataset_name,
        )
        sub_corpora_prim.append(job.out_bliss_corpus_primary)
        sub_corpora_sec.append(job.out_bliss_corpus_secondary)

    libri_css_bliss_prim = corpus.MergeCorporaJob(sub_corpora_prim, "libricss").out_merged_corpus
    libri_css_bliss_sec = corpus.MergeCorporaJob(sub_corpora_sec, "libricss").out_merged_corpus
    tk.register_output("data/libri_css_enhanced_tfgridnet_v0_prim.xml.gz", libri_css_bliss_prim)
    tk.register_output("data/libri_css_enhanced_tfgridnet_v0_sec.xml.gz", libri_css_bliss_sec)

    corpus_object_dict["libri_css_enhanced_tfgridnet_v0"] = SeparatedCorpusObject(
        primary_corpus_file=libri_css_bliss_prim,
        secondary_corpus_file=libri_css_bliss_sec,
        duration=11.0,
        audio_format="wav",
    )

    return corpus_object_dict


def get_data_inputs(
    add_unknown_phoneme_and_mapping: bool = False,
    use_stress: bool = False,
    ctc_lexicon: bool = False,
    lm_name: str = "4gram",
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    eval_corpus_objects = get_eval_corpus_object_dict()

    lm = get_lm(lm_name)

    original_bliss_lexicon = lbs_dataset.get_bliss_lexicon(
        use_stress_marker=use_stress,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )

    if use_augmented_lexicon:
        bliss_lexicon = lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=use_stress,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        )["train-other-960"]
    else:
        bliss_lexicon = original_bliss_lexicon

    bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon

    if ctc_lexicon:
        bliss_lexicon = DeleteEmptyOrthJob(bliss_lexicon).out_lexicon
        bliss_lexicon = MakeBlankLexiconJob(bliss_lexicon).out_lexicon

    lexicon_config = helpers.LexiconConfig(
        filename=bliss_lexicon,
        normalize_pronunciation=False,
        add_all_allophones=add_all_allophones,
        add_allophones_from_lexicon=not add_all_allophones,
    )

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    for dev_key in ["libri_css_enhanced_tfgridnet_v0"]:
        dev_data_inputs[dev_key] = helpers.RasrDataInput(
            corpus_object=eval_corpus_objects[dev_key],
            concurrent=40,
            lexicon=lexicon_config,
            lm=lm,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs
