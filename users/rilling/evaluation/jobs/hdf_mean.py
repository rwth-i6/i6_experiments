import h5py
from sisyphus import Job, Task, tk
import numpy as np


class MeanHDFContentJob(Job):
    def __init__(self, batch_hdf: tk.Path):
        super().__init__()
        self.batch_hdf = batch_hdf

        self.out_mean = self.output_path("mean")

    def task(self):
        yield Task("run", mini_task=True, rqmt={"sbatch_args": ["-p", "cpu_slow"]})

    def run(self):
        data = h5py.File(self.batch_hdf)

        inputs = np.array(data["inputs"])

        with open(self.out_mean.get_path(), "w+") as f:
            f.write(str(inputs.mean()) + "\n")
