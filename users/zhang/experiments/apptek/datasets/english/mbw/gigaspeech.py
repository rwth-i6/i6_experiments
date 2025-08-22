from io import BytesIO
import logging
import math
import os
import shutil
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

from sisyphus import Job, Task


logger = logging.getLogger(__name__)


split_durations = {
    "xl": 10_000,
    "dev": 12,
    "test": 40,
}
max_shards = 600


PARALLEL = 64


class GigaspeechCorpusToOggZipJob(Job):
    def __init__(self, split: str, hours_per_oggzip: int = 10):
        assert hours_per_oggzip > 0
        self.num_oggzips = min(math.ceil(split_durations[split] / hours_per_oggzip), max_shards)
        self.split = split

        self.out_oggzips = [self.output_path(f"gigaspeech.{split}.{i}.ogg.zip") for i in range(self.num_oggzips)]

        self.rqmt = {
            "cpu": 2,
            "mem": 8,
            "time": 0.2 * PARALLEL,
            "sbatch_args": ["-x", "c-[01-13]"],  # exclude weak CPU nodes
        }

    def tasks(self):
        yield Task(
            "run",
            args=range(self.num_oggzips),
            resume="run",
            rqmt=self.rqmt,
            parallel=PARALLEL,
        )
        yield Task("cleanup", mini_task=True)

    def run(self, oggzip_idx: int):
        for _ in range(3):
            try:
                self.run_(oggzip_idx)
                return
            except Exception as exc:
                last_exc = exc
                logging.warning(f"Error during run: {exc}, waiting 5min and retrying...")
                time.sleep(5 * 60)
        raise last_exc

    def run_(self, oggzip_idx: int):
        if os.path.exists(self.out_oggzips[oggzip_idx]):
            logging.info(f"{self.out_oggzips[oggzip_idx]} already exists, skipping.")
            return

        os.environ["HF_HOME"] = self._hf_path()

        from datasets import Dataset, DatasetDict, load_dataset
        import soundfile as sf

        datasets: DatasetDict = load_dataset(
            "speechcolab/gigaspeech",
            self.split,
            keep_in_memory=True,
            streaming=True,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )
        ds: Dataset = datasets["train"].shard(self.num_oggzips, oggzip_idx, contiguous=True)

        logging.info(f"Creating .ogg.zip {oggzip_idx}.")

        ogg_zip_file_name_without_zip = os.path.basename(self.out_oggzips[oggzip_idx]).removesuffix(".zip")

        with TemporaryDirectory() as tmpdir:
            zip_file_path = os.path.join(tmpdir, "ogg.zip")
            with ZipFile(zip_file_path, "w") as zip_file:
                meta_file = BytesIO()
                meta_file.write("[\n".encode("utf-8"))

                for entry in ds:
                    ogg_file_name = entry["segment_id"] + ".ogg"
                    meta = {
                        "duration": round(entry["end_time"] - entry["begin_time"], 5),
                        "file": ogg_file_name,
                        "seq_name": entry["segment_id"],
                        "speaker_name": entry["speaker"],
                        "text": self._remove_punctuation_tags(entry["text"]),
                    }
                    meta_file.write(f"  {repr(meta)},\n".encode("utf-8"))

                    data = entry["audio"]["array"]
                    assert entry["audio"]["sampling_rate"] == 16_000, "only 16kHz supported for now"

                    ogg_data = BytesIO()
                    sf.write(ogg_data, data, samplerate=16_000, format="ogg")

                    zip_file.writestr(f"{ogg_zip_file_name_without_zip}/{ogg_file_name}", ogg_data.getvalue())

                meta_file.write("]\n".encode("utf-8"))
                zip_file.writestr(f"{ogg_zip_file_name_without_zip}.txt", meta_file.getvalue())
            shutil.move(zip_file_path, self.out_oggzips[oggzip_idx])

    def cleanup(self):
        shutil.rmtree(self._hf_path())

    @staticmethod
    def _hf_path() -> str:
        return os.path.join(os.getcwd(), "hf")

    @staticmethod
    def _remove_punctuation_tags(text: str) -> str:
        for v in [
            "<COMMA>",
            "<PERIOD>",
            "<QUESTIONMARK>",
            "<EXCLAMATIONPOINT>",
            "<SIL>",
            "<MUSIC>",
            "<NOISE>",
            "<OTHER>",
        ]:
            text = text.replace(v, "")
        return text
