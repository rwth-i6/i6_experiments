import typing

from sisyphus import Job, Path, Task

from ..factored import LabelInfo


class JoinRightContextPriorsJob(Job):
    def __init__(self, log_prior_txts: typing.List[Path], label_info: LabelInfo):
        self.prior_file_paths = log_prior_txts
        self.label_info = label_info

        self.out_prior_txt = self.output_path("priors.txt")
        self.out_prior_xml = self.output_path("priors.xml")

    def tasks(self) -> typing.Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        def chunks(lst: typing.List, n: int):
            """Yield successive n-sized chunks from lst."""

            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        priors = []

        for path in self.prior_file_paths:
            with open(path, "r") as file:
                priors.extend(file.readlines())

        with open(self.out_prior_txt, "w") as txt:
            txt.writelines(priors)

        with open(self.out_prior_xml, "w") as xml:
            per_c_l_context = chunks(lst=priors, n=self.label_info.n_contexts)

            n_rows = self.label_info.n_contexts * self.label_info.get_n_state_classes()

            xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            xml.write(f'<matrix-f32 nRows="{n_rows}" nColumns="{self.label_info.n_contexts}">\n')
            for prior_chunk in per_c_l_context:
                xml.write(" ".join(prior_chunk))
                xml.write("\n")
            xml.write("\n</matrix-f32>")
