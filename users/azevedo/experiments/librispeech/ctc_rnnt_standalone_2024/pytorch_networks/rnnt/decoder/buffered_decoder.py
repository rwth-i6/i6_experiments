import torch
from typing import Optional, List, Tuple
from .experimental_rnnt_decoder import mask_tensor



def _encoder_infer(
    self,
    data_tensor: torch.Tensor, 
    mask_no_lookahead: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    x, mask_no_lookahead = self.encoder.frontend(data_tensor, mask_no_lookahead)  # [B, T, F']

    full_mask =  torch.ones_like(mask_no_lookahead).bool()
    for module in self.encoder.module_list:
        # take whole input (with right_context) through encoder
        x = module(x, full_mask)  # [B, T, F']

    return x, mask_no_lookahead

# TODO: carry over frames that are ignored during decoding (e.g. in the states)
def _infer(
    self,
    input: torch.Tensor,
    lengths: torch.Tensor,
    states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
    """
    Args:
        input (torch.Tensor): (C, 1)
        lengths (torch.Tensor): (C)
        states (Optional[List[List[torch.Tensor]]]): [(R, 1)]

            where C and R are the chunk size and lookahead size in [s] respectively.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]: 
    """

    lookahead_samples = states[0]
    lookahead_size = lookahead_samples.size(0)

    encoder_in = torch.cat((input, lookahead_samples), dim=0)
    squeezed_features = torch.squeeze(encoder_in)
    with torch.no_grad():
        audio_features, _ = self.feature_extraction(squeezed_features, lengths)

    # valid length of input without right_context for decoding
    no_lookahead_len = lengths - lookahead_size - self.feature_extraction.n_fft
    no_lookahead_len //= self.feature_extraction.hop_length
    no_lookahead_len += 1
    mask_no_lookahead = mask_tensor(audio_features, no_lookahead_len)
    
    encoder_out, out_mask = self._encoder_infer(audio_features, mask_no_lookahead)

    encoder_out = self.mapping(encoder_out)
    encoder_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

    return encoder_out, encoder_out_lengths