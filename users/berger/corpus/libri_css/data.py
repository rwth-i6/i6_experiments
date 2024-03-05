from typing import Callable, Dict, Tuple, List, Optional
from i6_core import corpus
from i6_core.meta.system import CorpusObject
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.users.berger import helpers
import i6_experiments.common.datasets.librispeech as lbs_dataset
from i6_experiments.users.berger.args.jobs.rasr_init_args import (
    get_feature_extraction_args_16kHz,
)
from i6_experiments.users.berger.helpers.hdf import build_rasr_feature_hdfs
from i6_experiments.users.berger.helpers.rasr import (
    SeparatedCorpusHDFFiles,
    SeparatedCorpusObject,
)
from i6_experiments.users.berger.recipe.converse.data import (
    EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob,
    EnhancedMeetingDataToSplitBlissCorporaJob,
    EnhancedSegmentedEvalDataToBlissCorpusJob,
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
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from sisyphus import tk

CLEAN_ALIGNMENT = AlignmentData(
    alignment_cache_bundle=tk.Path(
        "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/"
        "alignment.cache.bundle"
    ),
    allophone_file=tk.Path("/u/berger/asr-exps/librispeech/20230804_libri_css/allophones_2/allophones"),
    state_tying_file=tk.Path("/u/berger/asr-exps/librispeech/20230804_libri_css/state_tying_2/state-tying"),
)

JSON_DATABASES = {
    "enhanced_tfgridnet_v1": tk.Path(
        "/work/asr3/converse/data/libri_css/thilo_20230814_enhanced_tfgridnet_v2.json",
        hash_overwrite="tfgridnet_json_database_v1",
    ),
    "enhanced_blstm_v1": tk.Path(
        "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/database.json",
        hash_overwrite="blstm_json_database_v1",
    ),
    "libri_css_blstm_v1": tk.Path(
        "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1/libri_css_enhanced.json",
        hash_overwrite="libri_css_blstm_json_database_v1",
    ),
    "segmented_libri_css_blstm_v1": tk.Path(
        "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1/libri_css_enhanced_segmented.json",
        hash_overwrite="segmented_libri_css_blstm_json_database_v1",
    ),
    "libri_css_tfgridnet_v1": tk.Path(
        "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1.json",
        hash_overwrite="libri_css_tfgridnet_json_database_v1",
    ),
    "segmented_libri_css_tfgridnet_v1": tk.Path(
        "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1_segmented.json",
        hash_overwrite="segmented_libri_css_tfgridnet_json_database_v1",
    ),
}


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
        feature_hdfs[suffix] = build_rasr_feature_hdfs(
            corpus=corpus_object,
            split=concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
        )

    # alignments = gmm_system.outputs["train-other-960"]["final"].alignments.alternatives["bundle"]
    alignments = tk.Path(
        "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/mm/alignment/AlignmentJob.hK21a0UU4iiJ/output/alignment.cache.bundle"
    )
    # allophone_file = gmm_system.outputs["train-other-960"][
    #     "final"
    # ].crp.acoustic_model_post_config.allophones.add_from_file
    # crp = copy.deepcopy(gmm_system.outputs["train-other-960"]["final"].crp)
    # assert crp is not None
    # crp.set_executables(rasr_binary_path)
    # crp.lexicon_config.file = "/work/asr4/raissi/setups/librispeech/960-ls/work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.JOqKFQpjp04H/output/oov.lexicon.gz"
    # crp.acoustic_model_config.state_tying.file = tk.Path(
    #     "/work/asr3/raissi/shared_workspaces/gunz/dependencies/cart-trees/ls960/tri.tree.xml.gz"
    # )
    allophone_file = tk.Path("/u/berger/asr-exps/librispeech/20230804_libri_css/allophones_2/allophones")
    # crp.acoustic_model_config.allophones.add_from_lexicon = False
    # crp.acoustic_model_config.allophones.add_all = False
    # crp.acoustic_model_post_config.allophones.add_from_file = allophone_file
    # state_tying_file = DumpStateTyingJob(crp).out_state_tying
    state_tying_file = tk.Path("/u/berger/asr-exps/librispeech/20230804_libri_css/state_tying_2/state-tying")

    alignment_hdf_job = EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob(
        dataset_name=name,
        json_database=json_database,
        feature_hdfs=feature_hdfs["primary"],
        alignment_cache=alignments,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file,
        returnn_root=returnn_root,
    )
    alignment_hdf_job.rqmt.update({"mem": 10, "time": 24})

    all_segments = corpus.SegmentCorpusJob(bliss_corpora_job.out_bliss_corpus_primary, 1).out_single_segment_files[1]

    return SeparatedCorpusHDFFiles(
        primary_features_files=feature_hdfs["primary"],
        secondary_features_files=feature_hdfs["secondary"],
        mix_features_files=feature_hdfs["mix"],
        alignments_file=alignment_hdf_job.out_hdf_file,
        segments=all_segments,
    )


def get_clean_align_hdf(corpus_key: str, feature_hdfs: List[tk.Path], returnn_root: tk.Path) -> tk.Path:
    alignment_hdf_job = EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob(
        dataset_name="train_960",
        json_database=JSON_DATABASES[corpus_key],
        feature_hdfs=feature_hdfs,
        alignment_cache=CLEAN_ALIGNMENT.alignment_cache_bundle,
        allophone_file=CLEAN_ALIGNMENT.allophone_file,
        state_tying_file=CLEAN_ALIGNMENT.state_tying_file,
        returnn_root=returnn_root,
    )
    alignment_hdf_job.rqmt.update({"mem": 10, "time": 24})

    return alignment_hdf_job.out_hdf_file


def _get_corpus_object(
    *,
    name: str,
    json_database: tk.Path,
    map_enhanced_audio_paths: Callable,
    map_mix_audio_paths: Callable,
    corpus_duration: float,
    audio_format: str = "wav",
) -> SeparatedCorpusObject:
    bliss_corpora_job = EnhancedMeetingDataToSplitBlissCorporaJob(
        json_database=json_database,
        enhanced_audio_path_mapping=map_enhanced_audio_paths,
        mix_audio_path_mapping=map_mix_audio_paths,
        dataset_name=name,
    )

    return SeparatedCorpusObject(
        primary_corpus_file=bliss_corpora_job.out_bliss_corpus_primary,
        secondary_corpus_file=bliss_corpora_job.out_bliss_corpus_secondary,
        mix_corpus_file=bliss_corpora_job.out_bliss_corpus_mix,
        duration=corpus_duration,
        audio_format=audio_format,
    )


def map_audio_paths_libricss_train_mix(audio_path: str) -> str:
    return "/work/asr3/converse/data/libri_css/libricss_train_mix/" + audio_path


def map_audio_paths_libricss_train_tfgridnet(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/data/rwth_train/enhanced_tfgridnet_v2/",
        "/work/asr3/converse/data/libri_css/thilo_20230814_enhanced_tfgridnet_v2/",
    )


def map_audio_paths_libricss_train_blstm(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/data/rwth_train/enhanced/",
        "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/",
    )


def map_audio_paths_libricss_mix(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/cbj/deploy/libri_css/",
        "/work/asr3/converse/data/libri_css/libri_css_mix/",
    )


def map_audio_paths_libricss_tfgridnet(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/experiments/rwth/19/evaluation/ckpt_120000_1/",
        "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1/",
    )


def map_audio_paths_segmented_libricss_tfgridnet(audio_path: str) -> str:
    return audio_path.replace(
        "/scratch/hpc-prf-nt2/tvn/experiments/rwth/19/evaluation/ckpt_120000_1/",
        "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1/",
    )


def map_audio_paths_libricss_blstm(audio_path: str) -> str:
    return "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1" + audio_path


def map_audio_paths_segmented_libricss_blstm(audio_path: str) -> str:
    return "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1" + audio_path


def get_hdf_files(
    *,
    gmm_system: GmmSystem,
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
) -> Dict[str, SeparatedCorpusHDFFiles]:
    return {
        "enhanced_tfgridnet_v1": _get_hdf_files(
            gmm_system=gmm_system,
            name="train_960",
            json_database=tk.Path(
                "/work/asr3/converse/data/libri_css/thilo_20230814_enhanced_tfgridnet_v2.json",
                hash_overwrite="tfgridnet_json_database_v1",
            ),
            map_enhanced_audio_paths=map_audio_paths_libricss_train_tfgridnet,
            map_mix_audio_paths=map_audio_paths_libricss_train_mix,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            corpus_duration=1025.5,
            concurrent=200,
            audio_format="wav",
        ),
        "enhanced_blstm_v1": _get_hdf_files(
            gmm_system=gmm_system,
            name="train_960",
            json_database=tk.Path(
                "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/database.json",
                hash_overwrite="blstm_json_database_v1",
            ),
            map_enhanced_audio_paths=map_audio_paths_libricss_train_blstm,
            map_mix_audio_paths=map_audio_paths_libricss_train_mix,
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            corpus_duration=1025.5,
            concurrent=200,
            audio_format="wav",
        ),
    }


def _get_train_corpus_object_dict() -> Dict[str, SeparatedCorpusObject]:
    return {
        "enhanced_tfgridnet_v1": _get_corpus_object(
            name="train_960",
            json_database=tk.Path(
                "/work/asr3/converse/data/libri_css/thilo_20230814_enhanced_tfgridnet_v2.json",
                hash_overwrite="tfgridnet_json_database_v1",
            ),
            map_enhanced_audio_paths=map_audio_paths_libricss_train_tfgridnet,
            map_mix_audio_paths=map_audio_paths_libricss_train_mix,
            corpus_duration=1025.5,
            audio_format="wav",
        ),
        "enhanced_blstm_v1": _get_corpus_object(
            name="train_960",
            json_database=tk.Path(
                "/work/asr4/vieting/setups/converse/data/thilo_20230706_enhanced/enhanced_blstm/database.json",
                hash_overwrite="blstm_json_database_v1",
            ),
            map_enhanced_audio_paths=map_audio_paths_libricss_train_blstm,
            map_mix_audio_paths=map_audio_paths_libricss_train_mix,
            corpus_duration=1025.5,
            audio_format="wav",
        ),
    }


def split_session0_dev(corpus_path: tk.Path) -> Tuple[tk.Path, tk.Path]:
    all_segments = corpus.SegmentCorpusJob(corpus_path, 1).out_single_segment_files

    dev_segments = corpus.FilterSegmentsByRegexJob(
        all_segments,
        ".*session0.*",
        invert_match=True,
    ).out_single_segment_files
    dev_corpus_filtered = corpus.FilterCorpusBySegmentsJob(
        corpus_path,
        list(dev_segments.values()),
    ).out_corpus

    eval_segments = corpus.FilterSegmentsByRegexJob(
        all_segments,
        ".*session0.*",
        invert_match=False,
    ).out_single_segment_files
    eval_corpus_filtered = corpus.FilterCorpusBySegmentsJob(
        corpus_path,
        list(eval_segments.values()),
    ).out_corpus

    return dev_corpus_filtered, eval_corpus_filtered


def _get_dev_eval_corpus_object_dict() -> Dict[str, SeparatedCorpusObject]:
    corpus_object_dict = {}

    for name, job_type, job_kwargs in [
        (
            "libri_css_tfgridnet",
            EnhancedEvalDataToBlissCorpusJob,
            {
                "json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1.json",
                    hash_overwrite="libri_css_tfgridnet_json_database_v1",
                ),
                "enhanced_audio_path_mapping": map_audio_paths_libricss_tfgridnet,
                "mix_audio_path_mapping": map_audio_paths_libricss_mix,
            },
        ),
        (
            "segmented_libri_css_tfgridnet",
            EnhancedSegmentedEvalDataToBlissCorpusJob,
            {
                "json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1_segmented.json",
                    hash_overwrite="segmented_libri_css_tfgridnet_json_database_v1",
                ),
                "unsegmented_json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230727_libri_css_tvn_rwth_19_120000_1.json",
                    hash_overwrite="libri_css_tfgridnet_json_database_v1",
                ),
                "enhanced_audio_path_mapping": map_audio_paths_libricss_tfgridnet,
                "mix_audio_path_mapping": map_audio_paths_libricss_mix,
                "segment_audio_path_mapping": map_audio_paths_segmented_libricss_tfgridnet,
            },
        ),
        (
            "libri_css_blstm",
            EnhancedEvalDataToBlissCorpusJob,
            {
                "json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1/libri_css_enhanced.json",
                    hash_overwrite="libri_css_blstm_json_database_v1",
                ),
                "enhanced_audio_path_mapping": map_audio_paths_libricss_blstm,
                "mix_audio_path_mapping": map_audio_paths_libricss_mix,
            },
        ),
        (
            "segmented_libri_css_blstm",
            EnhancedSegmentedEvalDataToBlissCorpusJob,
            {
                "json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1/libri_css_enhanced_segmented.json",
                    hash_overwrite="segmented_libri_css_blstm_json_database_v1",
                ),
                "unsegmented_json_database": tk.Path(
                    "/work/asr3/converse/data/libri_css/20230810_libri_css_tvn_rwth_9_990000_1/libri_css_enhanced.json",
                    hash_overwrite="libri_css_blstm_json_database_v1",
                ),
                "enhanced_audio_path_mapping": map_audio_paths_libricss_blstm,
                "mix_audio_path_mapping": map_audio_paths_libricss_mix,
                "segment_audio_path_mapping": map_audio_paths_segmented_libricss_blstm,
            },
        ),
    ]:
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
            job = job_type(dataset_name=dataset_name, **job_kwargs)
            sub_corpora_prim.append(job.out_bliss_corpus_primary)
            sub_corpora_sec.append(job.out_bliss_corpus_secondary)
            sub_corpora_mix.append(job.out_bliss_corpus_mix)

        libri_css_bliss_prim = corpus.MergeCorporaJob(sub_corpora_prim, "libricss").out_merged_corpus
        libri_css_bliss_sec = corpus.MergeCorporaJob(sub_corpora_sec, "libricss").out_merged_corpus
        libri_css_bliss_mix = corpus.MergeCorporaJob(sub_corpora_mix, "libricss").out_merged_corpus

        libri_css_bliss_prim_dev, libri_css_bliss_prim_eval = split_session0_dev(libri_css_bliss_prim)
        libri_css_bliss_sec_dev, libri_css_bliss_sec_eval = split_session0_dev(libri_css_bliss_sec)
        libri_css_bliss_mix_dev, libri_css_bliss_mix_eval = split_session0_dev(libri_css_bliss_mix)

        dev_name = f"{name}_dev_v1"
        eval_name = f"{name}_eval_v1"

        tk.register_output(f"data/{dev_name}_prim.xml.gz", libri_css_bliss_prim_dev)
        tk.register_output(f"data/{dev_name}_sec.xml.gz", libri_css_bliss_sec_dev)
        tk.register_output(f"data/{dev_name}_mix.xml.gz", libri_css_bliss_mix_dev)

        tk.register_output(f"data/{eval_name}_prim.xml.gz", libri_css_bliss_prim_eval)
        tk.register_output(f"data/{eval_name}_sec.xml.gz", libri_css_bliss_sec_eval)
        tk.register_output(f"data/{eval_name}_mix.xml.gz", libri_css_bliss_mix_eval)

        corpus_object_dict[dev_name] = SeparatedCorpusObject(
            primary_corpus_file=libri_css_bliss_prim_dev,
            secondary_corpus_file=libri_css_bliss_sec_dev,
            mix_corpus_file=libri_css_bliss_mix_dev,
            duration=11.0,  # ToDo: incorrect duration
            audio_format="wav",
        )

        corpus_object_dict[eval_name] = SeparatedCorpusObject(
            primary_corpus_file=libri_css_bliss_prim_eval,
            secondary_corpus_file=libri_css_bliss_sec_eval,
            mix_corpus_file=libri_css_bliss_mix_eval,
            duration=11.0,  # ToDo: incorrect duration
            audio_format="wav",
        )

    return corpus_object_dict


def get_corpus_object_dict() -> Dict[str, SeparatedCorpusObject]:
    return {**_get_train_corpus_object_dict(), **_get_dev_eval_corpus_object_dict()}


def get_data_inputs(
    train_key: str = "enhanced_tfgridnet_v1",
    dev_keys: Optional[List[str]] = None,
    test_keys: Optional[List[str]] = None,
    add_unknown_phoneme_and_mapping: bool = False,
    use_stress: bool = False,
    ctc_lexicon: bool = False,
    lm_names: Optional[List[str]] = None,
    use_augmented_lexicon: bool = True,
    add_all_allophones: bool = False,
) -> Tuple[Dict[str, helpers.RasrDataInput], ...]:
    if dev_keys is None:
        dev_keys = ["segmented_libri_css_tfgridnet_dev_v1"]
    if test_keys is None:
        test_keys = ["segmented_libri_css_tfgridnet_eval_v1"]
    if lm_names is None:
        lm_names = ["4gram"]

    corpus_object_dict = get_corpus_object_dict()

    lms = {lm_name: get_lm(lm_name) for lm_name in lm_names}

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

    if ctc_lexicon:
        bliss_lexicon = EnsureSilenceFirstJob(bliss_lexicon).out_lexicon
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

    concurrent = {
        "enhanced_tfgridnet_v1": 200,
        "enhanced_blstm_v1": 200,
        "libri_css_tfgridnet_dev_v1": 40,
        "libri_css_tfgridnet_eval_v1": 40,
        "libri_css_blstm_dev_v1": 40,
        "libri_css_blstm_eval_v1": 40,
        "segmented_libri_css_tfgridnet_dev_v1": 40,
        "segmented_libri_css_tfgridnet_eval_v1": 40,
        "segmented_libri_css_blstm_dev_v1": 40,
        "segmented_libri_css_blstm_eval_v1": 40,
    }

    train_data_inputs[train_key] = helpers.RasrDataInput(
        corpus_object=corpus_object_dict[train_key],
        concurrent=concurrent[train_key],
        lexicon=lexicon_config,
    )

    for dev_key in dev_keys:
        for lm_name, lm in lms.items():
            dev_data_inputs[f"{dev_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[dev_key],
                concurrent=concurrent[dev_key],
                lexicon=lexicon_config,
                lm=lm,
            )

    for test_key in test_keys:
        for lm_name, lm in lms.items():
            test_data_inputs[f"{dev_key}_{lm_name}"] = helpers.RasrDataInput(
                corpus_object=corpus_object_dict[test_key],
                concurrent=concurrent[test_key],
                lexicon=lexicon_config,
                lm=lm,
            )

    return train_data_inputs, dev_data_inputs, test_data_inputs
