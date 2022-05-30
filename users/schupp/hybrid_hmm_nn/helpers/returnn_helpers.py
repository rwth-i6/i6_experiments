# From : i6_experiments/users/zeineldeen/returnn/training.py (162ee0be549cd432a1559f2f5050c6933a586672)
import os
import shutil
import subprocess as sp

from sisyphus import Job, Task, gs, tk

from i6_core.returnn.training import Checkpoint, ReturnnTrainingJob
from i6_core.returnn.rasr_training import ReturnnRasrTrainingJob


import copy

import i6_core.rasr as rasr
import i6_core.mm as mm
import i6_core.util as util


class GetBestEpochJob(Job):
    """
    Provided a RETURNN model directory and an optional score key, finds the best epoch.
    The sorting is lower=better, so to acces the model with the highest values use negative index values (e.g. -1 for
    the model with the highest score)

    If no key is provided, will search for a key prefixed with "dev_score_output", and default to the first key
    starting with "dev_score" otherwise.
    """

    def __init__(self, model_dir, learning_rates, index=0, key=None):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        :param str key: a key from the learning rate file that is used to sort the models
        """
        self.model_dir = model_dir
        self.learning_rates = learning_rates
        self.index = index
        self.out_epoch = self.output_var("epoch")
        self.key = key

        assert isinstance(index, int)

    def run(self):
        # this has to be defined in order for "eval" to work
        def EpochData(learningRate, error):
            return {'learning_rate': learningRate, 'error': error}

        with open(self.learning_rates.get_path(), 'rt') as f:
            text = f.read()

        print("TBS:")
        data = eval(text, {'inf': 1e99, 'EpochData': EpochData})
        print(data)

        epochs = list(sorted(data.keys()))
        print(epochs)

        error_key = None
        if self.key == None:
            dev_score_keys = [k for k in data[epochs[-1]]['error'] if k.startswith('dev_score')]
            for key in dev_score_keys:
                if key.startswith("dev_score_output"):
                    error_key = key
            if not error_key:
                error_key = dev_score_keys[0]
        else:
            error_key = self.key

        scores = [(epoch, data[epoch]['error'][error_key]) for epoch in epochs if error_key in data[epoch]['error']]
        print(scores)
        sorted_scores = list(sorted(scores, key=lambda x: x[1]))

        self.out_epoch.set(sorted_scores[self.index][0])

    def tasks(self):
        yield Task('run', mini_task=True)


class GetBestCheckpointJob(GetBestEpochJob):
    """
    Returns the best checkpoint given a training model dir and a learning-rates file
    The best checkpoint will be HARD-linked, so that no space is wasted but also the model not
    deleted in case that the training folder is removed.



    """

    def __init__(self, model_dir, learning_rates, index=0, key=None):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param int index: index of the sorted list to access, 0 for the lowest, -1 for the highest score
        :param str key: a key from the learning rate file that is used to sort the models
        """
        super().__init__(model_dir, learning_rates, index, key)
        self._out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = Checkpoint(self.output_path("model/checkpoint.index"))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        super().run()

        try:
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get())
            )
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get())
            )
            os.link(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get())
            )
        except OSError:
            # the hardlink will fail when there was an imported job on a different filesystem,
            # thus do a copy instead then
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get())
            )
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get())
            )
            shutil.copy(
                os.path.join(self.model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
                os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get())
            )

        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.index" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.index")
        )
        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.meta" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.meta")
        )
        os.symlink(
            os.path.join(self._out_model_dir.get_path(), "epoch.%.3d.data-00000-of-00001" % self.out_epoch.get()),
            os.path.join(self._out_model_dir.get_path(), "checkpoint.data-00000-of-00001")
        )


class AverageCheckpointsJob(Job):

    def __init__(self, model_dir, epochs, returnn_python_exe, returnn_root):
        """

        :param tk.Path model_dir:
        :param list[int|tk.Path] epochs:
        :param tk.Path returnn_python_exe:
        :param tk.Path returnn_root:
        """
        self.model_dir = model_dir
        self.epochs = epochs
        self.returnn_python_exe   = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root         = returnn_root       if returnn_root is not None else gs.RETURNN_ROOT

        self.avg_model_dir = self.output_path("avg_model", directory=True)
        self.avg_epoch = self.output_var("epoch")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        import numpy
        epochs = [epoch.get() if isinstance(epoch, tk.Variable) else epoch for epoch in self.epochs]
        avg_epoch = int(numpy.round(numpy.average(epochs), 0))
        args = [
            self.returnn_python_exe.get_path(),
            os.path.join(
                self.returnn_root.get_path(), "tools/tf_avg_checkpoints.py"),
            "--checkpoints", ','.join([str(epoch) for epoch in self.epochs]),
            "--prefix", self.model_dir.get_path() + "/epoch.",
            "--output_path", os.path.join(self.avg_model_dir.get_path(), "epoch.%.3d" % avg_epoch)
        ]
        sp.check_call(args)
        self.avg_epoch.set(avg_epoch)


class ReturnnRasrTrainingJobDevtrain(ReturnnTrainingJob):
    """
    Train a RETURNN model using rnn.py that uses ExternSpringDataset, and needs
    to write RASR config and flow files.
    """

    def __init__(
        self,
        train_crp,
        dev_crp,
        devtrain_crp,
        feature_flow,
        alignment,
        returnn_config,
        num_classes=None,
        *,  # args below are keyword-only args
        # these arges are passed on to ReturnnTrainingJob, have to be made explicit so sisyphus can detect them
        log_verbosity=3,
        device="gpu",
        num_epochs=1,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        horovod_num_processes=None,
        returnn_python_exe=None,
        returnn_root=None,
        # these are new parameters
        disregarded_classes=None,
        class_label_file=None,
        buffer_size=200 * 1024,
        partition_epochs=None,
        extra_rasr_config=None,
        extra_rasr_post_config=None,
        additional_rasr_config_files=None,
        additional_rasr_post_config_files=None,
        use_python_control=True,
    ):
        """

        :param rasr.CommonRasrParameters train_crp:
        :param rasr.CommonRasrParameters dev_crp:
        :param rasr.FlowNetwork feature_flow: RASR flow file for feature extraction or feature cache
        :param Path alignment: path to an alignment cache or cache bundle
        :param ReturnnConfig returnn_config:
        :param int num_classes:
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param str device: "cpu" or "gpu"
        :param int num_epochs: number of epochs to run, will also set `num_epochs` in the config file
        :param int save_interval: save a checkpoint each n-th epoch
        :param list[int]|set[int]|None keep_epochs: specify which checkpoints are kept, use None for the RETURNN default
        :param int|float time_rqmt:
        :param int|float mem_rqmt:
        :param int cpu_rqmt:
        :param int horovod_num_processes:
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        :param disregarded_classes:
        :param class_label_file:
        :param buffer_size:
        :param dict[str, int]|None partition_epochs: a dict containing the partition values for "train" and "dev"
        :param extra_rasr_config:
        :param extra_rasr_post_config:
        :param additional_rasr_config_files:
        :param additional_rasr_post_config_files:
        :param use_python_control:
        """
        datasets = self.create_dataset_config(
            train_crp, returnn_config, partition_epochs
        )
        returnn_config.config["train"] = datasets["train"]
        returnn_config.config["dev"] = datasets["dev"]
        returnn_config.config["devtrain"] = datasets["devtrain"]
        returnn_config.config["eval_datasets"] = {"devtrain": datasets["devtrain"]} # Allso add it as eval dataset
        super().__init__(
            returnn_config=returnn_config,
            log_verbosity=log_verbosity,
            device=device,
            num_epochs=num_epochs,
            save_interval=save_interval,
            keep_epochs=keep_epochs,
            time_rqmt=time_rqmt,
            mem_rqmt=mem_rqmt,
            cpu_rqmt=cpu_rqmt,
            horovod_num_processes=horovod_num_processes,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )
        kwargs = locals()
        del kwargs["self"]

        self.num_classes = num_classes
        self.alignment = alignment  # allowed to be None
        self.rasr_exe = rasr.RasrCommand.select_exe(
            train_crp.nn_trainer_exe, "nn-trainer"
        )
        self.additional_rasr_config_files = (
            {} if additional_rasr_config_files is None else additional_rasr_config_files
        )
        self.additional_rasr_post_config_files = (
            {}
            if additional_rasr_post_config_files is None
            else additional_rasr_post_config_files
        )

        del kwargs["train_crp"]
        del kwargs["dev_crp"]
        del kwargs["devtrain_crp"]
        kwargs["crp"] = train_crp
        self.feature_flow = ReturnnRasrTrainingJob.create_flow(**kwargs)
        (
            self.rasr_train_config,
            self.rasr_train_post_config,
        ) = ReturnnRasrTrainingJob.create_config(**kwargs)
        kwargs["crp"] = dev_crp
        (
            self.rasr_dev_config,
            self.rasr_dev_post_config,
        ) = ReturnnRasrTrainingJob.create_config(**kwargs)


        kwargs["crp"] = devtrain_crp
        (
            self.rasr_devtrain_config,
            self.rasr_devtrain_post_config,
        ) = ReturnnRasrTrainingJob.create_config(**kwargs)

        if self.alignment is not None:
            self.out_class_labels = self.output_path("class.labels")

    def create_files(self):
        if self.num_classes is not None:
            if "num_outputs" not in self.returnn_config.config:
                self.returnn_config.config["num_outputs"] = {}
            self.returnn_config.config["num_outputs"]["classes"] = [
                util.get_val(self.num_classes),
                1,
            ]

        super().create_files()

        rasr.RasrCommand.write_config(
            self.rasr_train_config,
            self.rasr_train_post_config,
            "rasr.train.config",
        )
        rasr.RasrCommand.write_config(
            self.rasr_dev_config, self.rasr_dev_post_config, "rasr.dev.config"
        )


        rasr.RasrCommand.write_config(
            self.rasr_devtrain_config, self.rasr_devtrain_post_config, "rasr.devtrain.config"
        )

        additional_files = set(self.additional_rasr_config_files.keys())
        additional_files.update(set(self.additional_rasr_post_config_files.keys()))
        for f in additional_files:
            rasr.RasrCommand.write_config(
                self.additional_rasr_config_files.get(f, {}),
                self.additional_rasr_post_config_files.get(f),
                f + ".config",
            )

        self.feature_flow.write_to_file("feature.flow")
        with open("dummy.flow", "wt") as f:
            f.write(
                '<?xml version="1.0" ?>\n<network><out name="features" /></network>'
            )

    def run(self):
        super().run()
        if self.alignment is not None:
            self._relink("class.labels", self.out_class_labels.get_path())

    @classmethod
    def create_config(
        cls,
        crp,
        alignment,
        num_classes,
        buffer_size,
        disregarded_classes,
        class_label_file,
        extra_rasr_config,
        extra_rasr_post_config,
        use_python_control,
        **kwargs,
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "acoustic_model": "neural-network-trainer.model-combination.acoustic-model",
                "corpus": "neural-network-trainer.corpus",
                "lexicon": "neural-network-trainer.model-combination.lexicon",
            },
            parallelize=(crp.concurrent == 1),
        )

        if use_python_control:
            config.neural_network_trainer.action = "python-control"
            config.neural_network_trainer.feature_extraction.file = "feature.flow"
            config.neural_network_trainer.python_control_enabled = True
            config.neural_network_trainer.python_control_loop_type = "iterate-corpus"
            config.neural_network_trainer.extract_alignments = alignment is not None
            config.neural_network_trainer.soft_alignments = False
        else:
            config.neural_network_trainer.action = "supervised-training"
            config.neural_network_trainer.feature_extraction.file = "dummy.flow"
            config.neural_network_trainer.aligning_feature_extractor.feature_extraction.file = (
                "feature.flow"
            )

        config.neural_network_trainer.single_precision = True
        config.neural_network_trainer.silence_weight = 1.0
        config.neural_network_trainer.weighted_alignment = False
        config.neural_network_trainer.class_labels.disregard_classes = (
            disregarded_classes
        )
        config.neural_network_trainer.class_labels.load_from_file = class_label_file
        config.neural_network_trainer.class_labels.save_to_file = "class.labels"

        config.neural_network_trainer.estimator = "steepest-descent"
        config.neural_network_trainer.training_criterion = "cross-entropy"
        config.neural_network_trainer.trainer_output_dimension = num_classes
        config.neural_network_trainer.buffer_type = "utterance"
        config.neural_network_trainer.buffer_size = buffer_size
        config.neural_network_trainer.shuffle = False
        config.neural_network_trainer.window_size = 1
        config.neural_network_trainer.window_size_derivatives = 0
        config.neural_network_trainer.regression_window_size = 5

        config._update(extra_rasr_config)
        post_config._update(extra_rasr_post_config)

        return config, post_config

    @classmethod
    def create_dataset_config(cls, train_crp, returnn_config, partition_epochs):
        """
        :param rasr.CommonRasrParameters train_crp:
        :param ReturnnConfig returnn_config:
        :param dict[str, int]|None partition_epochs: a dict containing the partition values for "train" and "dev"
        :return:
        """
        datasets = {}

        assert not (
            "partition_epoch" in returnn_config.config.get("train", {})
            and partition_epochs
        )
        assert not (
            "partition_epoch" in returnn_config.config.get("dev", {})
            and partition_epochs
        )

        assert not (
            "partition_epoch" in returnn_config.config.get("devtrain", {})
            and partition_epochs
        )

        if partition_epochs is None:
            partition_epochs = {"train": 1, "dev": 1, "devtrain" : 1}

        for ds in ["train", "dev", "devtrain"]:
            partition = int(partition_epochs.get(ds, 1))
            datasets[ds] = {
                "class": "ExternSprintDataset",
                "sprintTrainerExecPath": rasr.RasrCommand.select_exe(
                    train_crp.nn_trainer_exe, "nn-trainer"
                ),
                "sprintConfigStr": "--config=rasr.%s.config --*.LOGFILE=nn-trainer.%s.log --*.TASK=1"
                % (ds, ds),
                "partitionEpoch": partition,
            }

        # update rasr defaults with custom definitions
        if "train" in returnn_config.config:
            datasets["train"] = {
                **datasets["train"],
                **returnn_config.config["train"].copy(),
            }
        if "dev" in returnn_config.config:
            datasets["dev"] = {**datasets["dev"], **returnn_config.config["dev"].copy()}

        if "devtrain" in returnn_config.config:
            datasets["devtrain"] = {**datasets["devtrain"], **returnn_config.config["devtrain"].copy()}

        return datasets

    @classmethod
    def create_flow(cls, feature_flow, alignment, **kwargs):
        if alignment is not None:
            flow = mm.cached_alignment_flow(feature_flow, alignment)
        else:
            flow = copy.deepcopy(feature_flow)
        flow.flags["cache_mode"] = "bundle"
        return flow

    @classmethod
    def hash(cls, kwargs):
        flow = cls.create_flow(**kwargs)
        kwargs = copy.copy(kwargs)
        train_crp = kwargs["train_crp"]
        dev_crp = kwargs["dev_crp"]
        devtrain_crp = kwargs["dev_crp"]

        del kwargs["train_crp"]
        del kwargs["dev_crp"]
        del kwargs["devtrain_crp"]


        kwargs["crp"] = train_crp
        train_config, train_post_config = cls.create_config(**kwargs)
        kwargs["crp"] = dev_crp
        dev_config, dev_post_config = cls.create_config(**kwargs)
        kwargs["crp"] = devtrain_crp
        devtrain_config, devtrain_post_config = cls.create_config(**kwargs)

        datasets = cls.create_dataset_config(
            train_crp, kwargs["returnn_config"], kwargs["partition_epochs"]
        )
        kwargs["returnn_config"].config["train"] = datasets["train"]
        kwargs["returnn_config"].config["dev"] = datasets["dev"]
        kwargs["returnn_config"].config["devtrain"] = datasets["devtrain"]
        kwargs["returnn_config"].config["eval_datasets"] = {"devtrain" : datasets["devtrain"]}
        returnn_config = ReturnnTrainingJob.create_returnn_config(**kwargs)
        d = {
            "train_config": train_config,
            "dev_config": dev_config,
            "devtrain_config": devtrain_config,
            "alignment_flow": flow,
            "returnn_config": returnn_config,
            "rasr_exe": train_crp.nn_trainer_exe,
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        if kwargs["additional_rasr_config_files"] is not None:
            d["additional_rasr_config_files"] = kwargs["additional_rasr_config_files"]

        return Job.hash(d)
