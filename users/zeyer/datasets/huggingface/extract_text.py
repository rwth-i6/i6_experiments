from sisyphus import Job, Task, tk
from i6_core.util import uopen
from i6_experiments.users.zeyer.external_models.huggingface import set_hf_offline_mode, get_content_dir_from_hub_cache_dir


class ExtractTextFromHuggingFaceDatasetJob(Job):
    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_name: str,
        dataset_split: str,
    ):
        """
        :param dataset_dir: e.g. via DownloadHuggingFaceRepoJobV2 or DownloadAndPrepareHuggingFaceDatasetJob
            of the esb-datasets, which is also used for the OpenASRLeaderboard
            (or anything compatible to that).
            (Assumes that there is an `id` and `text` column.)
        :param dataset_name:
        :param dataset_split:
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.rqmt = {"time": 4, "cpu": 2, "mem": 10}

        self.out_text = self.output_path("ref.txt.py.gz")  # gzipped textdict

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        set_hf_offline_mode()

        from datasets import load_dataset

        # https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/data_utils.py
        # data_utils.load_data(args) is just load_dataset, nothing else
        dataset = load_dataset(
            get_content_dir_from_hub_cache_dir(self.dataset_dir),
            name=self.dataset_name,
            split=self.dataset_split,
            token=True,
        )
        print(f"Dataset: {dataset}")

        dataset = dataset.remove_columns(["audio"])
        dataset = dataset.filter(
            lambda ref: ref.strip() not in {"", "ignore time segment in scoring"}, input_columns=["text"]
        )

        # See SearchOutputRawReplaceJob and co.
        with uopen(self.out_text.get_path(), "wt") as out:
            out.write("{\n")
            for result in dataset:
                # https://huggingface.co/datasets/esb/datasets
                seq_tag = f"{self.dataset_name}/{self.dataset_split}/{result['id']}"
                ref = result["text"]
                assert isinstance(ref, str)
                out.write(f"{seq_tag!r}: {ref!r},\n")
            out.write("}\n")
