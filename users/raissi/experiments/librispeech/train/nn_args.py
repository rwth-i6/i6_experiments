__all__ = ['get_initial_nn_args']


def get_initial_nn_args():
    return {'num_input': 50,
            'partition_epochs': {'train': 40, 'dev': 1},
    }