from sisyphus import *

import os

import i6_core.returnn as returnn
import i6_core.rasr as rasr

class WriteFlowNetworkJob(Job):
    def __init__(self, flow):
        self.flow = flow
        self.out_network_file = self.output_path("network.flow")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self.flow.write_to_file(self.out_network_file.get_path())

class PickleSegmentsJob(Job):
  def __init__(self, segment_file):
    self.segment_file = segment_file
    self.out_segment_pickle = self.output_path("segments.pkl")
  
  def tasks(self):
    yield Task('run', mini_task=True)
  
  def run(self):
    segments = open(self.segment_file, "r").read().split('\n')[:-1]
    import pickle
    pickle.dump(segments, open(self.out_segment_pickle.get_path(), "wb"))

class WriteRasrConfigJob(Job):
    def __init__(self, crp, feature_flow, alignment,
        num_classes=None,
        disregarded_classes=None, class_label_file=None,
        buffer_size=200 * 1024,
        extra_rasr_config=None, extra_rasr_post_config=None,
        use_python_control=True,
    ):
        kwargs = locals()
        del kwargs["self"]

        self.use_python_control = use_python_control

        self.feature_flow = returnn.ReturnnRasrTrainingJob.create_flow(**kwargs)
        self.sprint_config, self.sprint_post_config = WriteRasrConfigJob.create_config(**kwargs)
        
        self.config = self.output_path("train.config")
    
    @classmethod
    def create_config(cls, **kwargs):
        config, post_config = returnn.ReturnnRasrTrainingJob.create_config(**kwargs)
        config.neural_network_trainer.class_labels.save_to_file = None
        return config, post_config
    
    def add_feature_flow(self, feature_flow_path, dummy_flow_path=None):
        if self.use_python_control:
            self.sprint_config.neural_network_trainer.feature_extraction.file = feature_flow_path
        else:
            assert dummy_flow_path is not None
            self.sprint_config.neural_network_trainer.aligning_feature_extractor.feature_extraction_file = feature_flow_path
            self.sprint_config.neural_network_trainer.feature_extraction.file = dummy_flow_path

    
    def run(self):
        # write feature flow
        self.feature_flow.write_to_file('feature.flow')
        feature_flow_path = os.path.abspath("feature.flow")

        # maybe write dummy flow
        dummy_flow_path = None
        if not self.use_python_control:
            with open('dummy.flow', 'wt') as f:
                f.write('<?xml version="1.0" ?>\n<network><out name="features" /></network>')
            dummy_flow_path = os.path.abspath("dummy.flow")

        # adjust sprint configs
        self.add_feature_flow(feature_flow_path, dummy_flow_path) 

        rasr.RasrCommand.write_config(self.sprint_config, self.sprint_post_config, self.config.get())
    
    @classmethod
    def hash(cls, kwargs):
        feature_flow = returnn.ReturnnRasrTrainingJob.create_flow(**kwargs)
        sprint_config, sprint_post_config = cls.create_config(**kwargs)
        d = {
            "sprint_config": sprint_config,
            "alignment_flow": feature_flow
        }
        return Job.hash(d)

    @staticmethod
    def config_str(config_path, name):
        return "--config={config_path} --*.LOGFILE=nn-trainer.{name}.log --*.TASK=1".format(**locals())

    def create_dataset_config(self, name, crp, partition_epochs=None, estimated_num_seqs=None, **kwargs):
        """ Returns a dataset config for use inside a returnn config.
        
        "function" attribute must be called such that the the config path is recognized by sisyphus
        as a "Path" object. """ 
        dataset = { 
            'class'                 : 'ExternSprintDataset',
            'sprintTrainerExecPath' : rasr.RasrCommand.select_exe(crp.nn_trainer_exe, 'nn-trainer'),
            'sprintConfigStr'       : self.config.function(WriteRasrConfigJob.config_str, name)
        }
        if partition_epochs is not None:
            dataset["partitionEpoch"] = partition_epochs
        if estimated_num_seqs is not None:
            dataset["estimated_num_seqs"] = estimated_num_seqs
        return dataset
