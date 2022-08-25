import copy

from i6_core import meta, returnn
# from recipe.experimental.mann.extractors import LearningRates
from .pretrain import remove_pretrain

from typing import Union, Tuple

def set_preload(system, config: returnn.ReturnnConfig, base_training: tuple, del_pretrain: bool=False):
    nn_name, nn_epoch = base_training
    checkpoint = system.nn_checkpoints['train'][nn_name][nn_epoch]
    config.config['preload_from_files'] = {
        nn_name : {
            'filename': checkpoint,
            'init_for_train': True,
            'ignore_missing': True
        }
    }
    if del_pretrain:
        for k in list(config.keys()):
            if k.startswith("pretrain"):
                del config[k]



class Preloader:
    def __init__(self, system):
        self.system = system
    
    def apply(self, crnn_config, base_training, del_pretrain=False, **_ignored):
        set_preload(self.system, crnn_config, base_training, del_pretrain)


def get_continued_training(
        system: meta.System, 
        training_args: dict, 
        base_config: dict, 
        lr_continuation: Union[float, int, bool],
        base_training: Tuple[str, int], 
        copy_mode: str = 'preload', 
        pretrain: bool = False,
        alignment=None, 
        dryrun: bool = False,
        inplace: bool = False,
        ) -> (dict, dict):
    """
    Outputs training args and crnn config dict for the continuation of a base
    training.
    Returns
    -------
    dict
        new training args
    dict
        new crnn config dict 
    """
    assert copy_mode in ['import', 'preload', None]
    training_args = copy.deepcopy(training_args)
    if not inplace:
        base_config = copy.deepcopy(base_config)
    if not pretrain: # delete all pretrain stuff
        remove_pretrain(base_config)
        # _ = base_config.pop('pretrain', None), \
        #     base_config.pop('pretrain_repetitions', None), \
        #     training_args.pop('extra_python', None)
    if alignment:
        training_args['alignment'] = alignment
    if not base_training:
        return training_args, base_config
    if len(base_training) == 3:
        nn_name, nn_epoch, _ = base_training
    else:
        nn_name, nn_epoch = base_training
    if isinstance(lr_continuation, (int, float)) and not isinstance(lr_continuation, bool):
        base_config.pop('learning_rates', None)
        base_config['learning_rate'] = lr_continuation
    elif lr_continuation: # continue from last learning rate
        lrs = LearningRates(
            system.jobs['train']['train_nn_{}'.format(nn_name)].learning_rates)
        tk.register_output('lrs_{}.txt'.format(nn_name), lrs.learning_rates)
        if not dryrun:
            lr = lrs.learning_rates[nn_epoch-1]
            base_config['learning_rate'] = lr
            base_config.pop('learning_rates', None)
    checkpoint = system.nn_checkpoints['train'][nn_name][nn_epoch]
    if len(base_training) == 3:
        checkpoint = base_training[2]
    if copy_mode == 'import': # copy params 1 to 1
        base_config['import_model_train_epoch1'] = checkpoint
    elif copy_mode == 'preload': # copy only same params
        base_config['preload_from_files'] = {
            nn_name : {
                'filename': checkpoint,
                'init_for_train': True,
                'ignore_missing': True
            }
        }
    return training_args, base_config