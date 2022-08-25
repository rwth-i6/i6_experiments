import copy
from collections import ChainMap

from sisyphus import tk, Job, Task, gs

from i6_core.returnn import ReturnnForwardJob, ReturnnRasrTrainingJob
from i6_core.rasr import WriteRasrConfigJob
from i6_core import meta
from .nn_system.base_system import NNSystem
from .nn_system.trainer import SemiSupervisedTrainer
from i6_experiments.users.mann.experimental.sequence_training import add_fastbw_configs
from i6_experiments.users.mann.experimental.plots import PlotSoftAlignmentJob
from i6_core.lexicon.allophones import StoreAllophonesJob, DumpStateTyingJob

def setup_datasets(system: NNSystem):
    """Make Rasr datasets into dicts for forward job."""
    pass

def dump_model(system: NNSystem, model: str, epoch: int):
    pass


class HdfDumpster:
    DEFAULT_PLOT_ARGS = {
        "alignment": ("train", "init_align", -1),
        "occurrence_thresholds": (5.0, 0.05),
    }
    DEFAULT_RQMTS = {
        "time_rqmt": 0.1,
        "cpu_rqmt": 1,
    }

    def __init__(self, system: NNSystem, segments, global_plot_args={}):
        self.system = system
        self.global_plot_args = ChainMap(global_plot_args, self.DEFAULT_PLOT_ARGS)
        self.segments = segments
        self.init_dump_corpus()
    
    def init_dump_corpus(self):
        from i6_core.corpus import SegmentCorpusJob, FilterSegmentsByListJob
        all_segments = SegmentCorpusJob(self.system.corpora["train"].corpus_file, 1)
        segments = FilterSegmentsByListJob({1: all_segments.out_single_segment_files[1]}, self.segments, invert_match=True)
        overlay_name = "returnn_dump"
        self.system.add_overlay("train", overlay_name)
        self.system.crp[overlay_name].concurrent = 1
        self.system.crp[overlay_name].segment_path = segments.out_single_segment_files[1]

    def _dump_hdf_helper(self,):
        pass
    
    def init_hdf_dataset(self, training_args):
        from i6_core.returnn import ReturnnRasrDumpHDFJob
        dumps = self._dump_hdf_helper()
        return 
    
    def init_rasr_configs(
        self,
        **kwargs
    ):
        pass

    def instantiate_fast_bw_layer(self, returnn_config, fast_bw_args):
        fastbw_config, fastbw_post_config \
            = add_fastbw_configs(self.system.csp['train'], **fast_bw_args) # TODO: corpus dependent and training args
        write_job = WriteRasrConfigJob(fastbw_config["fastbw"], fastbw_post_config["fastbw"])
        config_path = write_job.out_config
        try:
            returnn_config.config["network"]["fast_bw"]["sprint_opts"]["sprintConfigStr"] \
                = config_path.rformat("--config={}")
                # = "--config={}".format(config_path)
        except KeyError:
            pass
        return returnn_config

    def _prepare_config(self, returnn_config, dataset, hdf_outputs):
        new_returnn_config = copy.deepcopy(returnn_config)
        new_returnn_config.config["eval"] = dataset
        network = new_returnn_config.config["network"]
        for output in hdf_outputs:
            assert output in network, "Layer {} is not in the network".format(output)
            network["dump_{}".format(output)] = {
                "class": "hdf_dump", "filename": "{}.hdf".format(output),
                "is_output_layer": True, "from": output
            }
        return new_returnn_config

    def forward(self, name, returnn_config, epoch, hdf_outputs=["fast_bw"], training_args={}, fast_bw_args={}, **kwargs):
        args = ChainMap(kwargs, training_args, self.system.default_nn_training_args)        
        args = args.new_child({"partition_epochs": args["partition_epochs"]["dev"]})
        dataset = SemiSupervisedTrainer(self.system).make_sprint_dataset("forward", corpus="returnn_dump", **args)
        returnn_config = self.instantiate_fast_bw_layer(returnn_config, fast_bw_args)
        returnn_root = training_args.get("returnn_root", self.system.returnn_root) or tk.Path(gs.RETURNN_ROOT)
        returnn_python_exe = training_args.get("returnn_python_exe", self.system.returnn_python_exe) or tk.Path(gs.RETURNN_PYTHON_EXE)
        forward_job = ReturnnForwardJob(
            model_checkpoint=self.system.nn_checkpoints[args["feature_corpus"]][name][epoch],
            returnn_config=self._prepare_config(returnn_config, dataset, hdf_outputs),
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            hdf_outputs=["{}.hdf".format(output) for output in hdf_outputs],
            **self.DEFAULT_RQMTS,
        )
        out = {name.rstrip(".hdf"): file for name, file in forward_job.out_hdf_files.items()}
        return out
    
    def plot(self,
        name,
        bw_dumps,
        corpus,
        alignment,
        segments,
        occurrence_thresholds=(5.0, 0.05)
    ):
        allophones = StoreAllophonesJob(
            self.system.crp[corpus],
        ).out_allophone_file
        state_tying = DumpStateTyingJob(
            self.system.crp[corpus]
        ).out_state_tying
        plot_job = PlotSoftAlignmentJob(
            bw_dumps,
            alignment=meta.select_element(self.system.alignments, corpus, alignment),
            allophones=allophones,
            state_tying=state_tying,
            segments=segments,
            occurrence_thresholds=occurrence_thresholds,
        )

        if name is None:
            return
        for seg, path in plot_job.out_plots.items():
            tk.register_output("bw_plots/{}/{}.png".format(name, seg.replace("/", "_")), path)
    
    def run(self, name, returnn_config, epoch, training_args, fast_bw_args={}, plot_args={}, **kwargs):
        dumps = self.forward(name, returnn_config, epoch, training_args=training_args, fast_bw_args=fast_bw_args, **kwargs)
        self.plot(
            name="-".join([name, str(epoch)]),
            bw_dumps=dumps["fast_bw"],
            corpus="train",
            segments=self.segments,
            **self.global_plot_args.new_child(plot_args)
        )
