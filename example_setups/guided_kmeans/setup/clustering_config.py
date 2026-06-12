from dataclasses import dataclass, field
from pathlib import Path

from i6_core.returnn.forward import ReturnnForwardJobV2
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase, DelayedFormat

from i6_core.text.info import CountLinesJob
from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.serialization.base import Import, Collection, CallImport, NonhashedCode

import i6_experiments
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.clustering import (
    GuidedKMeansClusteringCallback,
    StreamingStandardInitializer,
        PreloadCentroidsInitializer,
    PickleCentroidRandomMapInitializer,
    PickleCentroidFrequencyOrderMapInitializer,
    PickleCheatingCentroidInitializer
)
from ..lib.serialization import HashedCode

_INITIALIZER_ASSIGN_NAME = "cluster_initializer"
_INITIALIZER_CLASS_DICT = {
    "PreloadCentroidsInitializerConfig": PreloadCentroidsInitializer,
    "StreamingStandardInitializerConfig": StreamingStandardInitializer,
    "PickleCentroidFrequencyOrderMapInitializerConfig": PickleCentroidFrequencyOrderMapInitializer,
    "PickleCentroidRandomMapInitializerConfig": PickleCentroidRandomMapInitializer,
    "PickleCheatingCentroidInitializerConfig": PickleCheatingCentroidInitializer,
}

#RETURNN_PYTHON_EXE = tk.Path("/work/asr3/michel/mann/virtualenv/2025-04-23_tensorflow-2.17_onnx-1.20_v1/bin/python3.11")
RETURNN_PYTHON_EXE = tk.Path("/usr/bin/python3")
RETURNN_ROOT = tk.Path("/u/mann/src/returnn")

class _Config:
    pass

class _NeedsLexicon:
    lexicon_path: str | tk.Path | None = None

class LateInitConfig(_Config):
    pass

@dataclass
class ClusteringCallbackConfig:
    num_clusters: int
    initializer_config: _Config
    recognition_config: str | tk.Path
    lexicon_path: str | tk.Path
    subsampling: int = 3
    callback_opts: dict = field(default_factory=dict)
    num_seqs: int | DelayedBase | None = field(init=False, default=None)
    rasr_path: tk.Path | None = None

    def is_fully_specified(self) -> bool:
        return (
            self.num_seqs is not None and
            not isinstance(self.initializer_config, LateInitConfig)
        )

@dataclass
class PreloadCentroidsInitializerConfig(_Config):
    centroids_path: str | tk.Path

@dataclass
class StreamingStandardInitializerConfig(_Config):
    seed: int

@dataclass
class PickleCentroidFrequencyOrderMapInitializerConfig(_Config):
    kmeans_path: str | tk.Path
    alignments_path: str | tk.Path
    lexicon_path: str | tk.Path | None = None

@dataclass
class PickleCentroidRandomMapInitializerConfig(_Config, _NeedsLexicon):
    kmeans_path: str | tk.Path

@dataclass
class PickleCheatingCentroidInitializerConfig(_Config, _NeedsLexicon):
    centroids_path: str | tk.Path = "/u/jxu/setups/unsupervised/2025-05-30--marten-unsupervised/output/sampled_alignments.pkl"
    lexicon_path: str | tk.Path | None = None

# def get_import_rasr_config(rasr_path: tk.Path):
#     import_recipes = NonhashedCode(f'import sys\nsys.path.insert(0, "{recipe_root}")\n')
#     import_obj = Import(
#         "i6_experiments.users.mann.experiments.guided_kmeans.setup.returnn.base_config.*"
#     )
#     return ReturnnConfig(
#         config={},
#         python_prolog=Collection([import_recipes, import_obj])
#     )

def get_base_config():
    recipe_root = str(Path(i6_experiments.__file__).parent.parent)
    import_recipes = NonhashedCode(f'import sys\nsys.path.insert(0, "{recipe_root}")\n')
    import_obj = Import(
        "i6_experiments.example_setups.guided_kmeans.setup.returnn.base_config.*"
    )
    return ReturnnConfig(
        config={},
        python_prolog=Collection([import_recipes, import_obj])
    )

def get_dataset_config(
    num_epochs: int,
    sampled_segments: tk.Path,
    hdf_path: str | tk.Path | None = None,
):
    core_dataset = {
        "class": "HDFDataset",
        "files": [
            hdf_path or "/work/asr4/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/work/i6_core/returnn/hdf/BlissToPcmHDFJob.vExsEVfudAcd/output/audio.hdf"
        ],
        "partition_epoch": 1,
        "seq_list_filter_file": sampled_segments,
        "use_cache_manager": True,
    }

    config = dict(
        forward_data = {
            "class": "MultiEpochDataset",
            "dataset": core_dataset,
            "multi_epoch": num_epochs,
        }
    )

    return ReturnnConfig(config)

def build_initializer_call_object(num_clusters: int, config: _Config):
    _class = _INITIALIZER_CLASS_DICT.get(config.__class__.__name__)
    return CallImport(
        code_object_path=_class,
        unhashed_package_root=None,
        hashed_arguments={
            "num_clusters": num_clusters,
            **config.__dict__
        },
        unhashed_arguments={},
        import_as=_INITIALIZER_ASSIGN_NAME
    )

def get_clustering_call_config(
    callback_config: ClusteringCallbackConfig
):
    assert callback_config.is_fully_specified(), "ClusteringCallbackConfig is not fully specified. All fields except num_seqs must be set, and num_seqs must be > 0."

    serializer_objs = []

    # insert rasr into path
    if callback_config.rasr_path:
        path_insertion_code = DelayedFormat(
            'sys.path.insert(0, "{rp}")\n',
            rp=callback_config.rasr_path,
        )
        serializer_objs.append(HashedCode(path_insertion_code))


    initializer = build_initializer_call_object(callback_config.num_clusters, callback_config.initializer_config)
    serializer_objs.append(initializer)

    arguments = {
        "num_clusters": callback_config.num_clusters,
        "initializer": CodeWrapper(_INITIALIZER_ASSIGN_NAME),
        "lexicon_path": callback_config.lexicon_path,
        "num_seqs": callback_config.num_seqs,
        "recognition_config": callback_config.recognition_config,
        "subsampling": callback_config.subsampling,
        **callback_config.callback_opts
    }
    clustering_callback = CallImport(
        code_object_path=GuidedKMeansClusteringCallback,
        unhashed_package_root=None,
        hashed_arguments=arguments,
        unhashed_arguments={},
        import_as="forward_callback"
    )
    serializer_objs.append(clustering_callback)
    return ReturnnConfig(
        config={},
        python_epilog=Collection(serializer_objs), # pyright: ignore[reportArgumentType]
    )

@dataclass
class ClusteringExpResult:
    fwd_job: ReturnnForwardJobV2
    out_centroids: dict[int, tk.Path]
    out_statistics: tk.Path

def clustering(
    num_epochs: int,
    sampled_segments: tk.Path,
    cluster_callback_config: ClusteringCallbackConfig,
    returnn_python_exe: tk.Path | None = None,
    returnn_root: tk.Path | None = None,
    hdf_path: str | tk.Path | None = None,
) -> ClusteringExpResult:
    internal_num_epochs = num_epochs * 2 + 1
    # set defaults
    if returnn_python_exe is None:
        returnn_python_exe = RETURNN_PYTHON_EXE
    if returnn_root is None:
        returnn_root = RETURNN_ROOT

    base_config = get_base_config()
    dataset_config = get_dataset_config(
        num_epochs=internal_num_epochs,
        sampled_segments=sampled_segments,
        hdf_path=hdf_path,
    )

    num_seqs = CountLinesJob(sampled_segments).out_num_lines
    cluster_callback_config.num_seqs = num_seqs
    clustering_call_config = get_clustering_call_config(callback_config=cluster_callback_config)
    
    config = ReturnnConfig({})
    config.update(base_config)
    config.update(dataset_config)
    config.update(clustering_call_config)
    config.black_formatting = False

    centroid_files = {
        epoch: f"centroids.{epoch}.npy" for epoch in range(num_epochs)
    }
    statistics_file = "epoch_statistics.json"

    fwd_job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=[
            *centroid_files.values(),
            statistics_file
        ]
    )

    out_centroids = {
        epoch: fwd_job.out_files[filename] for epoch, filename in centroid_files.items()
    }
    out_statistics = fwd_job.out_files[statistics_file] 

    return ClusteringExpResult(
        fwd_job=fwd_job,
        out_centroids=out_centroids,
        out_statistics=out_statistics
    )