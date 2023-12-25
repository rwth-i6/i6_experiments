import copy
from typing import List, Optional

from i6_core import features, rasr, returnn
from sisyphus import tk, Path
from sisyphus.delayed_ops import DelayedFormat

ALIGNMENT_PATH_10MS = Path(
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/10ms-dev-eval-subset/AlignmentJob.PzEwoG5YbNUb/output/alignment.cache.bundle",
    cached=True,
)
ALIGNMENT_PATH_40MS = Path(
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/40ms-dev-eval-subset/AlignmentJob.sZ5qa544Xsus/output/alignment.cache.bundle",
    cached=True,
)
CORPUS_PATH = Path("/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/10ms-dev-eval-subset/corpus.xml")
FEATURE_PATH = Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_core/features/extraction/FeatureExtractionJob.Gammatone.gXcFN7bQQqYf/output/gt.cache.bundle",
    cached=True,
)
LEX_PATH = Path(
    "/work/asr3/raissi/shared_workspaces/gunz/dependencies/alignments/ls-960/scratch/10ms-dev-eval-subset/lexicon.xml.gz"
)
SEGMENT_PATH = Path("/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/10ms-dev-eval-subset/segments.1")


class ReturnnEvalJob(returnn.ReturnnForwardJob):
    def __init__(
        self,
        model_checkpoint: Optional[returnn.Checkpoint],
        returnn_config: returnn.ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        hdf_outputs: Optional[List[str]] = None,
        eval_mode: bool = False,
        *,  # args below are keyword only
        log_verbosity: int = 5,
        device: str = "gpu",
        time_rqmt: float = 4,
        mem_rqmt: float = 4,
        cpu_rqmt: int = 2,
    ):
        super().__init__(
            model_checkpoint=model_checkpoint,
            returnn_config=returnn_config,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            hdf_outputs=hdf_outputs,
            eval_mode=eval_mode,
            log_verbosity=log_verbosity,
            device=device,
            time_rqmt=time_rqmt,
            mem_rqmt=mem_rqmt,
            cpu_rqmt=cpu_rqmt,
        )

        self.out_score_file = self.output_path("scores")

    def create_files(self, *args, **kwargs):
        self.returnn_config.config["eval_output_file"] = self.out_score_file

        super().create_files(*args, **kwargs)


def eval_dev_other_score_10ms(*args, **kwargs) -> ReturnnEvalJob:
    return eval_dev_other_score(
        *args,
        add_all_allos=True,
        alignment_path=ALIGNMENT_PATH_10MS,
        **kwargs,
    )


def eval_dev_other_score_40ms(*args, **kwargs) -> ReturnnEvalJob:
    return eval_dev_other_score(
        *args,
        add_all_allos=False,
        alignment_path=ALIGNMENT_PATH_40MS,
        reduce_target_factor=4,
        **kwargs,
    )


def eval_dev_other_score(
    *,
    name: str,
    crp: rasr.CommonRasrParameters,
    ckpt: returnn.Checkpoint,
    returnn_config: returnn.ReturnnConfig,
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    add_all_allos: bool,
    alignment_path: tk.Path,
    n_states_per_phone: int = 1,
    reduce_target_factor: Optional[int] = None,
    device: str = "cpu",
) -> ReturnnEvalJob:
    crp = copy.deepcopy(crp)
    crp.acoustic_model_config.hmm.states_per_phone = n_states_per_phone
    crp.corpus_config.file = CORPUS_PATH
    crp.lexicon_config.file = LEX_PATH
    crp.segment_path = SEGMENT_PATH

    if add_all_allos:
        crp.acoustic_model_config.allophones.add_all = add_all_allos
        crp.acoustic_model_config.allophones.add_from_lexicon = not add_all_allos

    feature_path = rasr.FlagDependentFlowAttribute("cache_mode", {"bundle": FEATURE_PATH})
    feature_flow = features.basic_cache_flow(feature_path)
    flow = returnn.ReturnnRasrTrainingJob.create_flow(feature_flow=feature_flow, alignment=alignment_path)
    flow_write_job = rasr.WriteFlowNetworkJob(flow)

    rasr_config, rasr_post_config = returnn.ReturnnRasrTrainingJob.create_config(
        crp=crp,
        alignment=alignment_path,
        num_classes=None,
        buffer_size=200 * 1024,
        disregarded_classes=None,
        class_label_file=None,
        extra_rasr_config=None,
        extra_rasr_post_config=None,
        use_python_control=True,
    )
    rasr_config.neural_network_trainer.feature_extraction.file = flow_write_job.out_flow_file
    rasr_config_write_job = rasr.WriteRasrConfigJob(rasr_config, rasr_post_config)

    returnn_config = copy.deepcopy(returnn_config)
    dset_config = {
        "class": "ExternSprintDataset",
        "sprintTrainerExecPath": rasr.RasrCommand.select_exe(
            crp.nn_trainer_exe,
            "nn-trainer",
        ),
        "sprintConfigStr": DelayedFormat(
            "--config={} --*.LOGFILE=nn-trainer.eval.log --*.TASK=1",
            rasr_config_write_job.out_config,
        ),
        "partitionEpoch": 1,
    }
    if reduce_target_factor:
        dset_config["reduce_target_factor"] = reduce_target_factor
    returnn_config.config = {**returnn_config.config, "dev": dset_config}

    job = ReturnnEvalJob(
        model_checkpoint=ckpt,
        returnn_config=returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        eval_mode=True,
        device=device,
        cpu_rqmt=4,
        time_rqmt=24,
    )
    job.add_alias(f"eval/{name}")
    tk.register_output(f"eval/{name}/scores", job.out_score_file)

    return job
