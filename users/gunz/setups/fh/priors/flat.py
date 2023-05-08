__all__ = ["CreateFlatPriorsJob"]

import math
import typing

from sisyphus import Job, Task


class CreateFlatPriorsJob(Job):
    """
    Creates normalized priors with the same probability for every state.

    Most useful if a given prior scale is 0.
    """

    def __init__(self, shape: typing.Union[int, typing.Tuple[int], typing.Tuple[int, int]]):
        assert isinstance(shape, int) or len(shape) in [1, 2]

        self.shape = shape
        self.out_priors = self.output_path("priors.xml")

    def tasks(self) -> typing.Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        if isinstance(self.shape, int):
            num_priors = self.shape
            priors_to_gen: typing.Union[int, typing.Tuple[int, int]] = self.shape
        elif len(self.shape) == 1:
            num_priors = self.shape[0]
            priors_to_gen = self.shape[0]
        elif len(self.shape) == 2:
            num_priors = math.prod(self.shape)
            priors_to_gen = self.shape
        else:
            raise AttributeError(f"unable to process {self.shape}")

        p_value = math.log(1.0 / float(num_priors))

        with open(self.out_priors, "w") as xml:
            xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')

            if isinstance(priors_to_gen, int):
                xml.write(f'<vector-f32 size="{priors_to_gen}">\n')

                xml.write(" ".join(priors_to_gen * [f"{p_value:.20e}"]))

                xml.write("\n</vector-f32>")
            else:
                n_rows, n_columns = priors_to_gen
                xml.write(f'<matrix-f32 nRows="{n_rows}" nColumns="{n_columns}">\n')

                single_row = " ".join(n_columns * [f"{p_value:.20e}"])
                xml.write("\n".join(n_rows * [single_row]))

                xml.write("\n</matrix-f32>")
