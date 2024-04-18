import h5py
from sisyphus import Job, Task, tk
import numpy as np


class MeanPhonemePredictionAccuracyJob(Job):
    def __init__(self, batch_accuracies_hdf: tk.Path):
        super().__init__()
        self.batch_accuracies_hdf = batch_accuracies_hdf

        self.out_accuracy = self.output_path("accuracy")

    def task(self):
        yield Task("run", mini_task=True, rqmt={"sbatch_args": ["-p", "cpu_slow"]})

    def run(self):
        data = h5py.File(self.batch_accuracies_hdf)

        inputs = np.array(data["inputs"])

        with open(self.out_accuracy.get_path(), "w+") as f:
            f.write(str(inputs.mean()) + "\n")
