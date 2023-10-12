from sisyphus import Job, Path, Task

from .processor import AlignmentProcessor


class ComputeSilencePercentageJob(Job):
    def __init__(self, alignment_bundle: Path, allophones_file: Path, sil_allophone: str = "[SILENCE]"):
        self.alignment_bundle = alignment_bundle
        self.allophone_file = allophones_file
        self.sil_allophone = sil_allophone

        self.out_percent_sil = self.output_var("sil")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 4}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        processor = AlignmentProcessor(
            alignment_bundle_path=self.alignment_bundle.get_path(),
            allophones_path=self.allophone_file.get_path(),
            sil_allophone=self.sil_allophone,
            monophone=False,
        )
        self.out_percent_sil.set(processor.percent_silence(0))
