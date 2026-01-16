from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.conformer_qwen_v2 import SllmV2
from i6_experiments.users.juanola.experiments.e25_11_10_llm.networks.interfaces.lm_decoder_model_protocol import \
    LmDecoderModelProtocol


class SllmV2Lm(SllmV2, LmDecoderModelProtocol):

    def decode_seq_lm(self, x: Tensor, x_lens: Tensor) -> Tensor:
        if not self.using_decoder:
            raise Exception("Trying to use forward decoder for Model without decoder!")

        input_target_embeddings = self.decoder_embed_func(x)

        qwen_output: CausalLMOutputWithPast = self.decoder.forward(
            inputs_embeds=input_target_embeddings,
            logits_to_keep=x_lens.max().item(),
        )

        return qwen_output.logits