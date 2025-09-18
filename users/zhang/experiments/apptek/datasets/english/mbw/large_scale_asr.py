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
    "clean": 13_000,
    "large": 25_000,
    "small": 250,
    "dev": 15,
    "test": 21,
}


PARALLEL = 64


class LargeScaleASRCorpusToOggZipJob(Job):
    def __init__(self, split: str, hours_per_oggzip: int = 10):
        assert hours_per_oggzip > 0
        self.num_oggzips = math.ceil(split_durations[split] / hours_per_oggzip)
        self.split = split

        self.out_oggzips = [self.output_path(f"large-scale-asr.{split}.{i}.ogg.zip") for i in range(self.num_oggzips)]

        self.download_rqmt = {"cpu": 16, "mem": 16, "time": 168}
        self.rqmt = {"cpu": 2, "mem": 8, "time": 0.2 * PARALLEL}

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
            "speechbrain/LargeScaleASR", self.split, keep_in_memory=True, streaming=True, token=os.getenv("HF_TOKEN")
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
                    ogg_file_name = entry["ID"] + ".ogg"
                    meta = {
                        "duration": round(entry["duration"], 5),
                        "file": ogg_file_name,
                        "seq_name": entry["ID"],
                        "speaker_name": entry["spk_id"],
                        "text": entry["text"],
                    }
                    meta_file.write(f"  {repr(meta)},\n".encode("utf-8"))

                    data_ = BytesIO(entry["wav"]["bytes"])
                    data, srate = sf.read(data_)
                    assert srate == 16_000, "only 16kHz supported for now"

                    ogg_data = BytesIO()
                    sf.write(ogg_data, data, samplerate=srate, format="ogg")

                    zip_file.writestr(f"{ogg_zip_file_name_without_zip}/{ogg_file_name}", ogg_data.getvalue())

                meta_file.write("]\n".encode("utf-8"))
                zip_file.writestr(f"{ogg_zip_file_name_without_zip}.txt", meta_file.getvalue())
            shutil.move(zip_file_path, self.out_oggzips[oggzip_idx])

    def cleanup(self):
        shutil.rmtree(self._hf_path())

    @staticmethod
    def _hf_path() -> str:
        return os.path.join(os.getcwd(), "hf")
