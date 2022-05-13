# This script allowes to run a eval on all stored models of a train
# i.e.: for all setups that trained without 'devtrain' initally

# This will use a ReturnnRasrTrainJob with some hacks:
# change task = 'train' -> task = 'eval'
# add devtrain ...

# Path to the experiments train dir
experiment_train_path = ""

def load_returnn_config_from_file(path):
    # TODO: load the config
    return # RetrunnConfig

# Adds 'devtrain'
def search_modify_config(config):
    return