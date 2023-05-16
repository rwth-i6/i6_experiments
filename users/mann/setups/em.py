import copy

from sisyphus import *

from recipe.i6_experiments.users.mann.nn import preload, tdps
from i6_experiments.users.mann.experimental.em import FindTransitionModelMaximumJob, TransitionModelFromCountsJob
from i6_experiments.users.mann.experimental.sequence_training import add_fastbw_configs
from i6_experiments.users.mann.nn import util as nn_util
from .nn_system.trainer import RasrTrainer
from i6_core.returnn.task import ReturnnCustomTaskJob
from i6_core.rasr.config import WriteRasrConfigJob
from .tdps import CombinedModel

NO_TDP_MODEL = CombinedModel.zeros()

def accumulate_transition_probabilities(engine, dataset, config=None):
    import numpy
    from returnn.tf.engine import Runner
    # assert isinstance(dataset, Dataset)
    if config:
        assert config is engine.config
    else:
        config = engine.config

    output_layer = engine._get_output_layer()
    assert config.has("output_file"), "output_file for priors numbers should be provided"
    output_file = config.value("output_file", "")
    assert not os.path.exists(output_file), "Already existing output file %r." % output_file
    # print("Compute priors, using output layer %r, writing to %r." % (output_layer, output_file), file=log.v2)

    class Accumulator(object):
        """
        Also see PriorEstimationTaskThread for reference.
        """

        def __init__(self):
            self.sum_posteriors = 0

        def __call__(self, outputs):
            """
            Called via extra_fetches_callback from the Runner.

            :param numpy.ndarray outputs: shape=(batch,n_labels,2)
            """
            outputs = numpy.array(outputs).sum(axis=0)
            self.sum_posteriors += outputs

    accumulator = Accumulator()
    batch_size = config.int("batch_size", 1)
    max_seqs = config.int("max_seqs", -1)
    epoch = config.int("epoch", 1)
    max_seq_length = config.float("max_seq_length", 0)
    if max_seq_length <= 0:
        max_seq_length = sys.maxsize
    dataset.init_seq_order(epoch=epoch)
    batches = dataset.generate_batches(
        recurrent_net=engine.network.recurrent,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_seqs=max_seqs,
        used_data_keys=engine.network.get_used_data_keys(),
    )
    forwarder = Runner(
        engine=engine,
        dataset=dataset,
        batches=batches,
        train=False,
        eval=False,
        extra_fetches={"outputs": output_layer.output},
        extra_fetches_callback=accumulator,
    )
    forwarder.run(report_prefix=engine.get_epoch_str() + " forward")
    if not forwarder.finalized:
        print("Error happened. Exit now.")
        forwarder.exit_due_to_error()
    
    print("sum_posteriors", accumulator.sum_posteriors)
    print("shape", accumulator.sum_posteriors.shape)
    numpy.savetxt(output_file, accumulator.sum_posteriors)
    print("Done.")

def run_task():
    from returnn.__main__ import (
        engine,
		config,
		train_data,
    )
    print("Run task accumulate_transition_probabilities")
    engine.init_network_from_config(config)
    accumulate_transition_probabilities(engine, dataset=train_data)


def make_net_untrainable(net):
    for layer in net.values():
        if layer["class"] not in {
            "linear", "hidden", "rec", "conv", "softmax", "variable"
        }: continue
        if layer["class"] == "subnetwork":
            make_net_untrainable(layer["subnetwork"])
            continue
        layer["trainable"] = False
    return net


def make_config_untrainable(config):
    res = copy.deepcopy(config)

    net = res.config["network"]
    make_net_untrainable(net)

    return res

class TransitionModelMaximizer:

    def __init__(
        self,
        system,
        returnn_config,
        returnn_python_exe,
        returnn_root,
    ):
        self.system = system
        self.returnn_config = returnn_config
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root
    
    def prepare_config(self, name, returnn_config, _debug=False):
        returnn_config = copy.deepcopy(returnn_config)
        nn_util.maybe_add_dependencies(
            returnn_config,
            "import os, sys",
            accumulate_transition_probabilities,
            run_task
        )

        returnn_config.config["forward_output_layer"] = "fast_bw_tdps"
        returnn_config.config["network"]["fast_bw_tdps"] = {"class": "copy", "from": "fast_bw/tdps", "is_output_layer": True}

        returnn_config.config["train"] = RasrTrainer(self.system).make_rasr_dataset(
            name=name,
            dataset_name="train",
            corpus="train",
            feature_flow="gt",
            alignment=None,
            filter_segments="head:10" if _debug else None,
        )

        return returnn_config
    
    def run(self, name, _debug=False):
        out_config = self.prepare_config(
            name,
            returnn_config=self.returnn_config,
            _debug=_debug,
        )
        # print("allow_random_init", out_config.config["allow_random_model_init"])
        task_job = ReturnnCustomTaskJob(
            out_config,
            self.returnn_python_exe,
            self.returnn_root,
            mem_rqmt=12,
        )

        return task_job.out_default_file


class EmRunner:
    TDP_ARCH = "label_speech_silence"

    def __init__(
        self,
		system,
        name,
		config,
        viterbi_config,
        exp_config,
        returnn_root,
        returnn_python_exe,
		num_epochs_per_maximization=1,
        partition_epochs=1,
        dump_alignment: bool = False,
        legacy_preload_fmt: bool = True,
    ):
        self.name = name
        self.system = system
        self.config = config
        self.viterbi_config = viterbi_config
        self.num_subepochs = num_epochs_per_maximization * partition_epochs
        self.exp_config = exp_config.extend(
            training_args={
                "num_epochs": self.num_subepochs,
                "partition_epochs": {
                    "train": partition_epochs,
                    "dev": 1,
                },
            }
        ).replace(epochs=[self.num_subepochs])
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe
        self.num_epochs_per_maximization = num_epochs_per_maximization
        self.arch_args = {
            "n_subclasses": 3,
            "div": 2,
            "silence_idx": system.silence_idx()
        }
        self.init_args = {
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
            "silence_idx": system.silence_idx(),
        }

        self.dump_alignment = dump_alignment
        self.legacy_preload_fmt = legacy_preload_fmt
        self.scores = {}

    def instantiate_fast_bw_layer(self, returnn_config, fast_bw_args):
        fast_bw_args = fast_bw_args.copy()
        fast_bw_args[
            "acoustic_model_extra_config"
        ] = NO_TDP_MODEL.to_acoustic_model_config()
        fastbw_config, fastbw_post_config \
            = add_fastbw_configs(self.system.crp[fast_bw_args.pop("corpus", "train")], **fast_bw_args) # TODO: corpus dependent and training args
        write_job = WriteRasrConfigJob(fastbw_config["fastbw"], fastbw_post_config["fastbw"])
        config_path = write_job.out_config
        try:
            returnn_config.config["network"]["fast_bw"]["sprint_opts"]["sprintConfigStr"] \
                = config_path.rformat("--config={}")
        except KeyError:
            pass
        return returnn_config
    
    def get_config_with_trans_model(self, config, trans_model: dict):
        config = copy.deepcopy(config)
        tdp_model = tdps.get_model(
            num_classes=self.system.num_classes(),
            arch=self.TDP_ARCH,
            extra_args=self.arch_args,
            init_args={
                "type": "smart",
                "silence_idx": self.system.silence_idx(),
                **trans_model
            }
        )
        tdp_model.set_config(config)

        net = config.config["network"]
        net["fast_bw_tdps"] = {"class": "expand_dims", "from": "fast_bw/tdps", "axis": "T",}

        return config
    
    def compute_trans_expectation(
        self,
        label_pos_model: str,
        trans_model: dict,
        _debug=False,
    ):
        config = copy.deepcopy(self.config)
        model_name = None
        # uniform label posterior model
        if label_pos_model is None:
            out_layer = config.config["network"]["output"]
            out_layer["forward_weights_init"] = 0.0
            out_layer["bias_init"] = 0.0
            config.config["allow_random_model_init"] = True
        else:
            model_name = label_pos_model
            config.config["load"] = self.system.nn_checkpoints["train_magic"][label_pos_model][self.num_subepochs],
            config.config["load_ignore_missing_vars"] = True
        
        config = self.get_config_with_trans_model(config, trans_model)
        config = self.instantiate_fast_bw_layer(config, self.exp_config.fast_bw_args)

        return TransitionModelMaximizer(
            self.system,
            config,
            self.returnn_python_exe,
            returnn_root=self.returnn_root
        ).run(name="trans_expectation-0", _debug=_debug)

    def make_fix_label_weights_config(
        self,
		returnn_config,
		trans_model: dict,
        label_pos_model: str,
        fast_bw_args: dict,
    ) -> str:
        res = copy.deepcopy(returnn_config)

        trans_config = self.get_config_with_trans_model(self.config, trans_model)
        self.instantiate_fast_bw_layer(trans_config, fast_bw_args)

        nn_util.maybe_add_dependencies(res, *trans_config.python_prolog)

        net = make_net_untrainable(
            trans_config.config["network"]
        )

        del net["output_bw"], net["output_tdps"]
        for key in ["fwd_1", "bwd_1"]:
            net[key]["from"] = "data"

        if label_pos_model is None:
            net["output"]["forward_weights_init"] = 0.0
            net["output"]["bias_init"] = 0.0

        res.config["network"]["label_weights"] = l = {
            "class": "subnetwork",
            "from": "data",
            "subnetwork": net
        }

        if label_pos_model:
            l["load_on_init"] = {
                "filename": self.system.nn_checkpoints["train_magic"][label_pos_model][self.num_subepochs],
                # "ignore_missing": True,
                "ignore_params_prefixes": ("tdps/",),
            }
        
            preload.set_preload(
                self.system,
                config=res,
                base_training=("train_magic", label_pos_model, self.num_subepochs),
                legacy_preload_fmt=self.legacy_preload_fmt,
            )

        res.config["network"]["output"]["target"] = "layer:label_weights/fast_bw"
        del res.config["network"]["output"]["loss_opts"]

        del res.config["chunking"]

        return res
    
    def maximize_label_pos(
        self,
        name: str,
        base_config,
        trans_model: dict,
        label_pos_model: tk.Path,
    ) -> str:
        if base_config is None:
            base_config = self.viterbi_config

        config = self.make_fix_label_weights_config(
            base_config,
            trans_model,
            label_pos_model,
            self.exp_config.fast_bw_args,
        )

        # print(config.python_prolog)

        self.system.run_exp(
            name,
            crnn_config=config,
            exp_config=self.exp_config,
            epochs=[self.num_subepochs],
        )

        if self.dump_alignment:
            self.system.dump_system.run(
                name=name,
                returnn_config=config,
                epoch=self.num_subepochs,
                training_args=self.exp_config.training_args,
                fast_bw_args=self.exp_config.fast_bw_args,
                hdf_outputs=["label_weights/fast_bw"],
            )

        return name

    def maximize_trans(self, trans_counts: tk.Path) -> dict:
        maximize_trans_job = TransitionModelFromCountsJob(
            trans_counts,
            silence_idx=self.system.silence_idx(),
        )
        res = {
            "speech_fwd": maximize_trans_job.out_speech_fwd,
            "silence_fwd": maximize_trans_job.out_silence_fwd,
        }
        return res
    
    def get_base_config_for_iteration(
        self,
        config,
        iteration: int,
    ):
        """Mostly adjusts learning rates."""
        res = copy.deepcopy(config)

        lrs = res.config["learning_rates"]
        subepoch = iteration * self.num_subepochs
        end_subepoch = self.num_subepochs + subepoch

        lr_start_idx = min(subepoch, len(lrs))
        lr_end_idx = min(end_subepoch, len(lrs))

        pad = self.num_subepochs - (lr_end_idx - lr_start_idx)

        res_lrs = lrs[lr_start_idx:lr_end_idx] + [lrs[-1]] * pad

        assert len(res_lrs) == self.num_subepochs

        res.config["learning_rates"] = res_lrs

        return res
    
    def score(
        self,
        name,
        trans_model,
        label_pos_model,
    ):
        score_config = copy.deepcopy(self.config)
        score_config = self.get_config_with_trans_model(score_config, trans_model)

        # preload.set_preload(
        #     self.system,
        #     config=score_config,
        #     base_training=("train_magic", label_pos_model, self.num_subepochs),
        # )
        self.system.dump_system.score(
            name=label_pos_model,
            epoch=self.num_subepochs,
            returnn_config=score_config,
            training_args=self.exp_config.training_args,
            fast_bw_args=self.exp_config.fast_bw_args,
        )
        self.scores[name] = self.system.dump_system.scores[name]["train_score_output_bw"]

    def run(
        self,
        name,
        base_config=None,
        num_iterations=1,
        train_trans=True,
        score_iterations=None,
    ):
        label_pos_model = None
        trans_model = {
            "speech_fwd": 1/3,
            "silence_fwd": 1/40,
        }
        # self.scores[name] = {}
        for it in range(num_iterations):
            # weights = self.compute_expectation(label_pos_model, trans_model)
            # label_pos_model, trans_model = self.compute_maximum(f"{self.name}-{it}", weights)
            if train_trans:
                trans_counts = self.compute_trans_expectation(label_pos_model, trans_model)
                new_trans_model = self.maximize_trans(trans_counts)
                for key, val in new_trans_model.items():
                    tk.register_output(f"weights/{name}/{key}-{it}", val)
            else:
                new_trans_model = trans_model

            if base_config:
                config = self.get_base_config_for_iteration(base_config, it)
            else:
                config = None
            new_label_pos_model = self.maximize_label_pos(
                f"{name}-{it}", config, trans_model, label_pos_model
            )

            label_pos_model, trans_model = new_label_pos_model, new_trans_model

            if isinstance(score_iterations, (list, tuple, set)) and it in score_iterations:
                self.score(f"{name}-{it}", trans_model, label_pos_model)

