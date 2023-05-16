import copy
from collections import ChainMap

from sisyphus import tk, Job, Task, gs, delayed_ops

from i6_core.returnn import ReturnnForwardJob, ReturnnRasrTrainingJob
from i6_core.rasr import WriteRasrConfigJob
from i6_core import meta
from .nn_system.base_system import NNSystem
from .nn_system.trainer import BaseTrainer, RasrTrainer
from i6_experiments.users.mann.experimental.sequence_training import add_fastbw_configs
from i6_experiments.users.mann.experimental.plots import PlotSoftAlignmentJob
from i6_core.lexicon.allophones import StoreAllophonesJob, DumpStateTyingJob

from i6_experiments.users.mann.nn import preload

def setup_datasets(system: NNSystem):
    """Make Rasr datasets into dicts for forward job."""
    pass

def dump_model(system: NNSystem, model: str, epoch: int):
    pass

class DelayedLearningRates(delayed_ops.Delayed):

    def get(self):
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        with open(self.a.get(), "rt") as lr_file:
            text = lr_file.read()

        return eval(text)


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

    def _dump_hdf_helper(self,
        corpus,
        feature_flow,
        alignment,
        num_classes,
        buffer_size=200 * 1024,
        cpu=2, mem=8, file_size=100,
        time=4,
        returnn_python_exe=None,
        returnn_root=None,
        **_ignored
    ):
        kwargs = locals().copy()
        kwargs.pop("_ignored"), kwargs.pop("self"), kwargs.pop("corpus"), kwargs.pop("feature_flow"), kwargs.pop("alignment")
        from i6_core.returnn import ReturnnRasrDumpHDFJob
        j = ReturnnRasrDumpHDFJob(
            crp=self.system.crp[corpus],
            feature_flow=meta.select_element(self.system.feature_flows, corpus, feature_flow),
            alignment=meta.select_element(self.system.alignments, corpus, alignment),
            **kwargs
        )
        return j.out_hdf
    
    def init_hdf_dataset(self, name, dump_args):
        from i6_core.returnn import ReturnnRasrDumpHDFJob
        dump_args = ChainMap(dump_args, self.system.default_nn_training_args)
        dumps = self._dump_hdf_helper(**dump_args)
        tk.register_output("hdf_dumps/{}.hdf".format(name), dumps)
        return dumps
    
    def init_rasr_configs(
        self,
        **kwargs
    ):
        pass

    def init_score_segments(self, corpus="crnn_train"):
        self.scores = {}
        self.lr_files = {}
        from i6_core.corpus import ShuffleAndSplitSegmentsJob
        new_segments = ShuffleAndSplitSegmentsJob(
            segment_file=self.system.crp[corpus].segment_path,
            split={"train": 0.01, "dev": 0.2, "rest": 0.79},
        )
        for key in ["train", "dev"]:
            overlay_name = "returnn_score_{}".format(key)
            self.system.add_overlay(corpus, overlay_name)
            self.system.crp[overlay_name].segment_path = new_segments.out_segments[key]

    def instantiate_fast_bw_layer(self, returnn_config, fast_bw_args):
        fast_bw_args = fast_bw_args.copy()
        fastbw_config, fastbw_post_config \
            = add_fastbw_configs(self.system.csp[fast_bw_args.pop("corpus", "train")], **fast_bw_args) # TODO: corpus dependent and training args
        write_job = WriteRasrConfigJob(fastbw_config["fastbw"], fastbw_post_config["fastbw"])
        config_path = write_job.out_config
        try:
            returnn_config.config["network"]["fast_bw"]["sprint_opts"]["sprintConfigStr"] \
                = config_path.rformat("--config={}")
        except KeyError:
            pass
        return returnn_config

    def _prepare_config(self, returnn_config, dataset, hdf_outputs):
        new_returnn_config = copy.deepcopy(returnn_config)
        new_returnn_config.config["eval"] = dataset
        network = new_returnn_config.config["network"]
        new_outputs = []
        output_map = {}
        for output in hdf_outputs:
            if "/" in output:
                parent_layer, sub_layer = output.split("/")
                assert parent_layer in network and (
                    sub_layer in network[parent_layer].get("subnetwork", {})
                    or sub_layer in network[parent_layer]
                ), "Layer {} is not in the network".format(output)
                output_name = "_".join((parent_layer, sub_layer))
            else:
                assert output in network, "Layer {} is not in the network".format(output)
                output_name = output
            output_file_name = "{}.hdf".format(output_name)
            output_map[output] = output_file_name
            network["dump_{}".format(output_name)] = {
                "class": "hdf_dump", "filename": output_file_name,
                "is_output_layer": True, "from": output
            }
            # new_outputs.append(output_file_name)
        return new_returnn_config, output_map

    def forward(
        self,
		name,
		returnn_config,
		epoch,
		hdf_outputs=["fast_bw"],
		training_args={},
		fast_bw_args={},
		corpus="returnn_dump",
        init_fast_bw_layer=True,
        eval_mode=False,
        time_rqmt=4,
        mem_rqmt=4,
		**kwargs
    ):
        args = ChainMap(kwargs, training_args, self.system.default_nn_training_args)        
        if returnn_config is None:
            returnn_config = self.system.nn_config_dicts["train"][name]
        args = args.new_child({
            "partition_epochs": args["partition_epochs"].get("dev", 1),
            "corpus": corpus
        })
        # dataset = BaseTrainer(self.system).make_sprint_dataset("forward", **args)
        dataset = RasrTrainer(self.system).make_rasr_dataset(name, "forward", **args)
        if init_fast_bw_layer:
            returnn_config = self.instantiate_fast_bw_layer(returnn_config, fast_bw_args)
        returnn_root = training_args.get("returnn_root", self.system.returnn_root) or tk.Path(gs.RETURNN_ROOT)
        returnn_python_exe = training_args.get("returnn_python_exe", self.system.returnn_python_exe) or tk.Path(gs.RETURNN_PYTHON_EXE)
        model_checkpoint = None
        if name is not None:
            model_checkpoint = meta.select_element(self.system.nn_checkpoints, args["feature_corpus"], (args["feature_corpus"], name, epoch))
        
        returnn_config, output_file_names = self._prepare_config(returnn_config, dataset, hdf_outputs)
        forward_job = ReturnnForwardJob(
            model_checkpoint=model_checkpoint,
            returnn_config=returnn_config,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            hdf_outputs=list(output_file_names.values()),
            eval_mode=eval_mode,
            time_rqmt=time_rqmt,
            mem_rqmt=mem_rqmt,
        )
        out = {name: forward_job.out_hdf_files[hdf_file] for name, hdf_file in output_file_names.items()}
        out["output"] = forward_job.out_hdf_files["output.hdf"]
        return out
    
    def score(self, name, returnn_config, epoch, extra_name=None, training_args={}, fast_bw_args={}, **kwargs):
        score_config = copy.deepcopy(self.system.nn_config_dicts["train"][name])
        if returnn_config is not None:
            score_config = copy.deepcopy(returnn_config)
        score_config.config["learning_rate"] = 0
        score_config.config.pop("learning_rates", None)
        feature_corpus = ChainMap(
            training_args,
            self.system.default_nn_training_args,
        ).get("feature_corpus", "train")
        preload.set_preload(self.system, score_config, (feature_corpus, name, epoch))

        extra_training_args = {
            "train_corpus": "returnn_score_train",
            "dev_corpus": "returnn_score_dev",
            "num_epochs": 1,
            "time_rqmt": 1,
        }

        scoring_name = "-".join((extra_name or name, "score"))
        self.system.nn_and_recog(
            name=scoring_name,
            crnn_config=score_config,
            epochs=[],
            training_args=ChainMap(extra_training_args, training_args),
            fast_bw_args=fast_bw_args,
            train_prefix="score_returnn",
            **kwargs
        )

        score_job = self.system.jobs[feature_corpus]["train_nn_" + scoring_name]

        self.lr_files[name] = score_job.out_learning_rates
        self.scores[name] = DelayedLearningRates(score_job.out_learning_rates)[1]["error"]

    
    def plot(self,
        name,
        bw_dumps,
        corpus,
        alignment,
        segments,
        occurrence_thresholds=(5.0, 0.05)
    ):
        plot_job = PlotSoftAlignmentJob(
            bw_dumps,
            alignment=meta.select_element(self.system.alignments, corpus, alignment),
            allophones=self.system.get_allophone_file(),
            state_tying=self.system.get_state_tying_file(),
            segments=segments,
            occurrence_thresholds=occurrence_thresholds,
            hmm_partition=self.system.crp[corpus].acoustic_model_config.hmm.states_per_phone
        )

        if name is None:
            return
        for seg, path in plot_job.out_plots.items():
            tk.register_output("bw_plots/{}/{}.png".format(name, seg.replace("/", "_")), path)
    
    def run(self, name, returnn_config, epoch, training_args, hdf_outputs=["fast_bw"], fast_bw_args={}, plot_args={}, **kwargs):
        assert len(hdf_outputs) == 1, "Only one output supported for now"
        dumps = self.forward(name, returnn_config, epoch, training_args=training_args, fast_bw_args=fast_bw_args, hdf_outputs=hdf_outputs, **kwargs)
        self.plot(
            name="-".join([name, str(epoch)]),
            bw_dumps=dumps[hdf_outputs[0]],
            corpus="train",
            segments=self.segments,
            **self.global_plot_args.new_child(plot_args)
        )
