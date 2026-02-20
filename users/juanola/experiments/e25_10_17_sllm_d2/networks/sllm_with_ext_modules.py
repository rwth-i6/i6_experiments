__all__ = ["SllmV4"]

from typing import Optional, Dict, Any, Tuple, Union

from returnn_common.nn import Tensor
from .conformer_qwen_v2 import SllmV2
from .conformer_qwen_v3 import SllmV3
from .qwen2_decoder_state import Qwen2DecoderState


class SllmV4(SllmV2):
    """
    SLLM with external modules:
    - self.external_ctc
    - self.external_lm
    """

    external_ctc: Optional[SllmV2]
    external_lm: Optional[SllmV2]

    def __init__(
        self,
        # Path to checkpoint or a dict containing the full SllmV3 kwargs
        external_ctc_args: Optional[Dict[str, Any]] = None,
        external_lm_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Initialize the main model (Encoder + Decoder + Adapter)
        super().__init__(**kwargs)

        # 1. Initialize External CTC (Encoder-only SllmV3)
        self.external_ctc = None
        if external_ctc_args:
            # We enforce using_decoder=False to keep it a 'Black Box' CTC
            external_ctc_args.update({
                "epoch": None,
                "step": None,

                "using_encoder": True,
                "using_decoder": False,
                "verbose": True}
            )
            self.external_ctc = SllmV2(**external_ctc_args)
            if self.verbose:
                print("--- External CTC Module Initialized ---")
        else:
            print("--- NO EXTERNAL ENCODER ---")

        # 2. Initialize External LM (Decoder-only SllmV3)
        self.external_lm = None
        if external_lm_args:
            # We enforce using_encoder=False to keep it a 'Black Box' LM
            external_lm_args.update({
                "epoch": None,
                "step": None,

                "using_encoder": False,
                "using_decoder": True,
                "verbose": True
            })
            self.external_lm = SllmV2(**external_lm_args)
            if self.verbose:
                print("--- External LM Module Initialized ---")
        else:
            print("--- NO EXTERNAL DECODER ---")

    def has_external_ctc(self):
        return self.external_ctc is not None

    def external_ctc_forward_encoder(self, raw_audio: Tensor, raw_audio_lens: Tensor, initial_beam_size: int) -> tuple[Union[Qwen2DecoderState | None], Tensor, Tensor]:
        """Standard protocol for CTC branch"""
        if self.external_ctc is None:
            raise Exception("External CTC Module Not Initialized")
        return self.external_ctc.forward_encoder(raw_audio, raw_audio_lens, initial_beam_size)

    def external_llm_step_decoder(
        self, labels: Tensor, state: Qwen2DecoderState
    ) -> Tuple[Tensor, Qwen2DecoderState]:
        """Standard protocol for LM branch"""
        if self.external_lm is None:
            raise Exception("External LM Module Not Initialized")
        return self.external_lm.step_decoder(labels, state)

