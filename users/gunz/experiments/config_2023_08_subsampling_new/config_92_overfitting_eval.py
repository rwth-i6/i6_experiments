import copy

from i6_core import features, rasr, returnn
from sisyphus import tk, Path
from sisyphus.delayed_ops import DelayedFormat

ALIGNMENT_PATH = Path(
    "/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/dev-eval-subset/AlignmentJob.PzEwoG5YbNUb/output/alignment.cache.bundle"
)
CORPUS_PATH = Path("/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/dev-eval-subset/corpus.xml")
FEATURE_PATH = Path(
    "/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_core/features/extraction/FeatureExtractionJob.Gammatone.gXcFN7bQQqYf/output/gt.cache.bundle"
)
SEGMENT_PATH = Path("/u/mgunz/gunz/dependencies/alignments/ls-960/scratch/dev-eval-subset/segments.1")


class ReturnnEvalJob(returnn.ReturnnForwardJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_score_file = self.output_path("scores")

    def create_files(self, *args, **kwargs):
        self.returnn_config.config["eval_output_file"] = self.out_score_file

        super().create_files(*args, **kwargs)

    @classmethod
    def hash(cls, *args, **kwargs):
        d = {
            "returnn_config": super().create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)


def eval_dev_other_score(
    *,
    name: str,
    crp: rasr.CommonRasrParameters,
    ckpt: returnn.Checkpoint,
    returnn_config: returnn.ReturnnConfig,
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    device: str = "cpu",
) -> ReturnnEvalJob:
    crp = copy.deepcopy(crp)
    crp.corpus_config.file = CORPUS_PATH
    crp.segment_path = SEGMENT_PATH

    feature_path = rasr.FlagDependentFlowAttribute("cache_mode", {"bundle": FEATURE_PATH})
    feature_flow = features.basic_cache_flow(feature_path)
    feature_flow.flags = {"cache_mode": "bundle"}
    flow = returnn.ReturnnRasrTrainingJob.create_flow(feature_flow=feature_flow, alignment=ALIGNMENT_PATH)
    flow_write_job = rasr.WriteFlowNetworkJob(flow)

    rasr_config, rasr_post_config = returnn.ReturnnRasrTrainingJob.create_config(
        crp=crp,
        alignment=ALIGNMENT_PATH,
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
    returnn_config.config = {
        **returnn_config.config,
        "eval": {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(
                crp.nn_trainer_exe,
                "nn-trainer",
            ),
            "sprintConfigStr": DelayedFormat(
                "--config={}, --*.LOGFILE=nn-trainer.eval.log, --*.TASK=1",
                rasr_config_write_job.out_config,
            ),
            "partitionEpoch": 1,
        },
    }

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