import os
import shutil
from typing import Dict, Generator, Optional

from sisyphus import Job, Task, tk

from i6_core.returnn.training import Checkpoint


class GetBestEpochJob(Job):
    """
    Provided a RETURNN model directory and an optional score key, finds the best epoch.

    If no key is provided, will search for a key prefixed with "dev_score_output", and default to the first key
    starting with "dev_score" otherwise.
    """

    def __init__(
        self, learning_rates: tk.Path, score_key: Optional[str] = None
    ) -> None:
        """
        :param Path learning_rates: learning_rates output from a RETURNNTrainingJob
        :param str score_key: a key from the learning rate file that is used to sort the models
        """
        self.learning_rates = learning_rates
        self.score_key = score_key
        self.out_epoch = self.output_var("epoch")
        self.out_score = self.output_var("score")

    def get_score_key(self, score_data: Dict[int, dict]) -> str:
        """
        :param score_data: dictionary that was extracted from learning_rate file
        :returns: some key that represents scores to sort the models by
        :raises ValueError: for invalid learning_rate files
        """

        assert score_data
        # Get data from last epoch because the first ones may be different due to pretraining
        score_keys = set(score_data.get(len(score_data), {}).get("error", {}).keys())
        if not score_keys:
            raise ValueError("Learning rate file contains no score keys")

        # If possible, use the given key
        if self.score_key is not None:
            if not self.score_key in score_keys:
                raise ValueError("Learning rate file does not contain given score key")
            return self.score_key

        # We iterate through these patterns until we find a matching key
        for pattern in [
            "dev_score_output",
            "dev_score",
            "dev_error",
            "train_score_output",
            "train_score",
            "train_error",
        ]:
            for key in score_keys:
                if pattern in key:
                    return key

        # None of the patterns match any of the existing keys
        raise ValueError("Learning rate file contains no suitable key")

    def run(self) -> None:
        inf_score = 1e99
        # this has to be defined in order for "eval" to work
        def EpochData(learningRate, error) -> dict:
            return {"learningRate": learningRate, "error": error}

        with open(self.learning_rates.get_path(), "rt") as f:
            text = f.read()

        data = eval(text, {"inf": inf_score, "EpochData": EpochData})

        if not data:
            raise ValueError("Learning rate file is empty")

        score_key = self.get_score_key(data)

        self.out_epoch.set(
            min(data, key=lambda ep: data[ep]["error"].get(score_key, inf_score))
        )

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)


class GetBestCheckpointJob(GetBestEpochJob):
    """
    Returns the best checkpoint given a training model dir and a learning-rates file
    The best checkpoint will be HARD-linked, so that no space is wasted but also the model not
    deleted in case that the training folder is removed.
    """

    def __init__(self, model_dir: tk.Path, *args, **kwargs):
        """

        :param Path model_dir: model_dir output from a RETURNNTrainingJob
        """
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.out_model_dir = self.output_path("model", directory=True)
        self.out_checkpoint = Checkpoint(self.output_path("model/checkpoint.index"))

    def tasks(self) -> Generator[Task, None, None]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        super().run()

        base_name = f"epoch.{self.out_epoch.get():03d}"

        for suffix in [
            "index",
            "meta",
            "data-00000-of-00001",
        ]:
            try:
                os.link(
                    os.path.join(self.model_dir.get_path(), f"{base_name}.{suffix}"),
                    os.path.join(
                        self.out_model_dir.get_path(), f"{base_name}.{suffix}"
                    ),
                )
            except OSError:
                # the hardlink will fail when there was an imported job on a different filesystem,
                # thus do a copy instead then
                shutil.copy(
                    os.path.join(self.model_dir.get_path(), f"{base_name}.{suffix}"),
                    os.path.join(
                        self.out_model_dir.get_path(), f"{base_name}.{suffix}"
                    ),
                )

            os.symlink(
                os.path.join(self.out_model_dir.get_path(), f"{base_name}.{suffix}"),
                os.path.join(self.out_model_dir.get_path(), f"checkpoint.{suffix}"),
            )
