__all__ = ['CRNNDumpSprintTrainingJob']

from sisyphus import *
Path = setup_path(__package__)

import copy
import os
import shutil
import recipe.experimental.mann.plot_adam as plota
from recipe.crnn import CRNNSprintTrainingJob

class CRNNDumpSprintTrainingJob(CRNNSprintTrainingJob):
    def __init__(self, train_csp, dev_csp, feature_flow, alignment, reduced_segment_path,
               crnn_config, crnn_post_config=None, num_classes=None, checkpoint=None,
               *, # args below are keyword-only args
               # these arges are passed on to CRNNTrainingJob, have to be made explicit so sisyphus can detect them
               log_verbosity=3, device='gpu',
               num_epochs=1, save_interval=1, keep_epochs=None,
               time_rqmt=4, mem_rqmt=4, cpu_rqmt=1,
               extra_python='', extra_python_hash=None,
               crnn_python_exe=None, crnn_root=None,
               disregarded_classes=None, class_label_file=None,
               buffer_size=200 * 1024, partition_epochs=None,
               extra_sprint_config=None, extra_sprint_post_config=None,
               additional_sprint_config_files=None, additional_sprint_post_config_files=None,
               use_python_control=True,
               # these are new parameters
               debug_dump=False, debug_crnn_root=None, plot=None):

        crnn_root = debug_crnn_root if debug_crnn_root else gs.CRNN_DEBUG_ROOT
        train_csp.segment_path = reduced_segment_path
        if checkpoint:
            crnn_config['import_model_train_epoch1'] = checkpoint

        # call super constructor
        super().__init__(train_csp=train_csp, dev_csp=dev_csp, feature_flow=feature_flow, alignment=alignment,
                   crnn_config=crnn_config, crnn_post_config=crnn_post_config, num_classes=num_classes,
                   log_verbosity=log_verbosity, device=device,
                   num_epochs=num_epochs, save_interval=save_interval, keep_epochs=keep_epochs,
                   time_rqmt=time_rqmt, mem_rqmt=mem_rqmt, cpu_rqmt=cpu_rqmt,
                   extra_python=extra_python, extra_python_hash=extra_python_hash,
                   crnn_python_exe=crnn_python_exe, crnn_root=crnn_root,
                   disregarded_classes=disregarded_classes, class_label_file=class_label_file,
                   buffer_size=buffer_size, partition_epochs=partition_epochs,
                   extra_sprint_config=extra_sprint_config, extra_sprint_post_config=extra_sprint_post_config,
                   additional_sprint_config_files=additional_sprint_config_files, additional_sprint_post_config_files=additional_sprint_post_config_files,
                   use_python_control=use_python_control)

        self.dump_dir = self.output_path("dump", directory=True)
        self.dumps = {
            (i, b): self.output_path('dump/alignment.dump.%d.%d' % (i,b)) for i, b in [(4,0)]
            }
        self.plot = plot
        if plot:
            self.plot = self.output_path("plot_alignment.png")


    def move_debug_files(self):
        """ Moves debug dump to the folder. """
        # move files
        count = 0
        for file in os.listdir('.'): 
            if "alignment.dump" in file or ".npy" in file:
                shutil.move(file, self.dump_dir.get_path())
                count += 1

        assert count > 0


    def plot_dumps(self):
        """ Plots the debug dump. """
        if isinstance(self.plot, int):
            dump_path = os.path.join(self.dump_dir.get_path(), 'alignment.dump.{idx}.0'.format(idx=self.plot))
            a = plota.Adam("epoch32", {'idx': [self.plot]}, dump_path,
                    cache_dir="plots/cache",
                    recache=False)
            a.plot(title_string='Dump', save=self.plot.get_path())
            

    def tasks(self):
        for task in super().tasks():
            if task.name != 'plot':
                yield task
        yield Task('move_debug_files', resume='move_debug_files', mini_task=True)
        if self.plot:
            yield Task('plot_dumps', resume='plot_dumps', mini_task=True)


    @classmethod
    def hash(cls, kwargs):
        flow = cls.create_flow(**kwargs)
        kwargs = copy.copy(kwargs)
        train_csp = kwargs['train_csp']
        dev_csp   = kwargs['dev_csp']
        del kwargs['train_csp']
        del kwargs['dev_csp']
        kwargs['csp'] = train_csp
        train_config, train_post_config = cls.create_config(**kwargs)
        kwargs['csp'] = dev_csp
        dev_config,   dev_post_config   = cls.create_config(**kwargs)
        extra_python_hash = kwargs['extra_python'] if kwargs['extra_python_hash'] is None else kwargs['extra_python_hash']

        d = { 'train_config'    : train_config,
              'dev_config'      : dev_config,
              'alignment_flow'  : flow,
              'crnn_config'     : kwargs['crnn_config'],
              'extra_python'    : extra_python_hash,
              'sprint_exe'      : train_csp.nn_trainer_exe,
              'crnn_python_exe' : kwargs['crnn_python_exe'],
              'crnn_root'       : kwargs['crnn_root'],
              }

        if kwargs['additional_sprint_config_files'] is not None:
            d['additional_sprint_config_files'] = kwargs['additional_sprint_config_files']

        if kwargs['partition_epochs'] is not None:
            d['partition_epochs'] = kwargs['partition_epochs']

        if kwargs['checkpoint'] is not None:
            d['checkpoint'] = kwargs['checkpoint']

        return Job.hash(d)
    

