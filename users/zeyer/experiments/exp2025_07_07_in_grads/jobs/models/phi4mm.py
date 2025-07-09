import torch
from .base import BaseModelInterface


class Phi4MM(BaseModelInterface):
    """
    Phi4MM model interface.
    """

    def __init__(
        self,
        *,
        model_dir: str,
        speech_prompt: str = "Transcribe the audio clip into text.",
        grad_wrt: str = "speech_embeddings",
    ):
        """
        :param model_dir: hub cache dir of model e.g. via DownloadHuggingFaceRepoJob.out_hub_cache_dir
        :param speech_prompt: text-only part of the prompt
        """
        super().__init__()
        self.speech_prompt = speech_prompt
        self.grad_wrt = grad_wrt

    def forward(self, inputs: torch.Tensor): ...
