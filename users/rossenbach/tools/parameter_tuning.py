from sisyphus import Job, Task
from typing import List, Tuple, Iterator

import numpy as np

from i6_core.util import instanciate_delayed


class PickOptimalParametersJob(Job):


    def __init__(self, parameters: List[Tuple], values: List, mode="minimize"):
        """
        :param parameters:
        :param values:
        :param mode:
        """
        assert len(parameters) == len(values)
        for param in parameters[1:]:
            assert len(param) == len(parameters[0])
        assert mode in ["minimize", "maximize"]
        self.parameters = parameters
        self.values = values
        self.mode = mode
        self.num_values = len(values)
        self.num_parameters = len(parameters[0])

        self.out_optimal_parameters = [self.output_var("param_%i" % i) for i in range(self.num_parameters)]


    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        values = instanciate_delayed(self.values)

        if self.mode == "minimize":
            index = np.argmin(values)
        else:
            index = np.argmax(values)

        best_parameters = self.parameters[index]

        for i, param in enumerate(best_parameters):
            self.optimal_parameters[i].set(param)

