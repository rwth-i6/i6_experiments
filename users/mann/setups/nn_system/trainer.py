from recipe.crnn.helpers.mann.write import WriteSprintConfigJob
from recipe.crnn import CRNNTrainingJob
from recipe.meta.system import select_element
from collections import ChainMap

class SemiSupervisedTrainer:

    def __init__(self):
        pass

    def set_system(self, system, **kwargs):
        self.system = system

    @staticmethod
    def write_helper(csp, feature_flow, alignment,
            num_classes=None,
            disregarded_classes=None, class_label_file=None,
            buffer_size=200 * 1024,
            extra_sprint_config=None,
            use_python_control=True,
            **kwargs
        ):
            kwargs = locals()
            del kwargs["kwargs"]
            return WriteSprintConfigJob(**kwargs)

    def write(self, corpus, feature_corpus, feature_flow, alignment, num_classes, **kwargs):
        j = SemiSupervisedTrainer.write_helper(
            csp          = self.system.csp[corpus],
            feature_flow = self.system.feature_flows[feature_corpus][feature_flow],
            alignment    = select_element(self.system.alignments, feature_corpus, alignment),
            num_classes  = self.system.functor_value(num_classes),
            **kwargs)
        return j

    # @staticmethod
    def train_helper(
            self,
            train_data, dev_data, crnn_config,
            crnn_post_config=None, num_classes=None,
            log_verbosity=3, device='gpu',
            num_epochs=1, save_interval=1, keep_epochs=None,
            time_rqmt=4, mem_rqmt=4, cpu_rqmt=2,
            extra_python='', extra_python_hash=None,
            crnn_python_exe=None, crnn_root=None,
            **_ignored
        ):
            num_classes = self.system.functor_value(num_classes)
            kwargs = locals()
            del kwargs["_ignored"], kwargs["self"]
            return CRNNTrainingJob(**kwargs)

    def make_sprint_dataset(
            self,
            name,
            corpus, feature_corpus, feature_flow, alignment, num_classes,
            estimated_num_seqs=None, partition_epochs=None,
            **kwargs
        ):
            kwargs = locals()
            kwargs.update(kwargs.pop("kwargs", {}))
            del kwargs["self"]
            return self.write(**kwargs).create_dataset_config(csp=self.system.csp[corpus], **kwargs)

    def make_combined_ds(self, arg_mapping):
        keys = ["alignment", "teacher"]
        acc_num_seqs = sum(args["estimated_num_seqs"] for args in arg_mapping.values())
        datasets = {
            key: self.make_sprint_dataset(**arg_mapping[key]) for key in keys
        }
        return {
            "class": "CombinedDataset",
            "datasets": datasets,
            "data_map": {
                ("alignment", "data"): "data",
                ("teacher", "data"): "data",
                ("alignment", "classes"): "classes"
            },
            "seq_ordering": "random_dataset",
            "estimated_num_seqs": acc_num_seqs,
        }
    
    def train(self, name, train_data, dev_data, crnn_config, feature_corpus, **kwargs):
        train_data = self.make_combined_ds(train_data)
        dev_data = self.make_sprint_dataset(**dev_data)
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]
        j = self.train_helper(**training_args)
        # feature_corpus = "train"
        self.system.jobs[feature_corpus]['train_nn_%s' % name] = j
        self.system.nn_models[feature_corpus][name] = j.models
        self.system.nn_checkpoints[feature_corpus][name] = j.checkpoints
        self.system.nn_configs[feature_corpus][name] = j.crnn_config_file


class SoftAlignTrainer(SemiSupervisedTrainer):

    def get_segments_pkl(self, overlay_name):
        from recipe.crnn.multi_sprint_training import PickleSegments
        return PickleSegments(self.system.csp[overlay_name].segment_path).segment_pickle

    def make_hdf_dataset(self, soft_alignment):
        return {
            "class": "HDFDataset",
            "files": [soft_alignment],
            "use_cache_manager": True
        }

    def make_combined_ds(self, name, corpus, soft_alignment, partition_epoch, arg_mapping):
        # acc_num_seqs = sum(args["estimated_num_seqs"] for args in arg_mapping.values())
        # TODO:
        # * class: hdf
        # * data_map: map soft_align to data
        # * add entry in num_outputs
        datasets = {}
        datasets["sprint"] = self.make_sprint_dataset(name, corpus, **arg_mapping)
        datasets["soft_align"] = self.make_hdf_dataset(soft_alignment)
        return {
            "class": "MetaDataset",
            "datasets": datasets,
            "data_map": {
                "classes_soft_align": ("soft_align", "data"),
                "classes": ("sprint", "classes"),
                "data": ("sprint", "data")
            },
            "seq_list_file": self.get_segments_pkl(corpus),
            "partition_epoch": partition_epoch,
            "seq_ordering": "default"
        }

    def train(self, name, soft_alignment, partition_epochs, crnn_config, feature_corpus, num_classes, **kwargs):
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]
        crnn_config["num_outputs"]["classes_soft_align"] = [
            self.system.functor_value(num_classes), 2
        ]
        data = {
            key + "_data": self.make_combined_ds(
                key,
                "crnn_" + key,
                soft_alignment,
                partition_epochs[key],
                dict(num_classes=num_classes, feature_corpus=feature_corpus, **kwargs)
            )
            for key in ["train", "dev"]
        }
        # training_args.maps.insert(0, data)
        j = self.train_helper(**data, **training_args)
        # feature_corpus = "train"
        self.system.jobs[feature_corpus]['train_nn_%s' % name] = j
        self.system.nn_models[feature_corpus][name] = j.models
        self.system.nn_checkpoints[feature_corpus][name] = j.checkpoints
        self.system.nn_configs[feature_corpus][name] = j.crnn_config_file
