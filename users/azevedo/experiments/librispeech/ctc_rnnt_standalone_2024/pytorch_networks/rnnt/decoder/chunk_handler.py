from typing import Optional, Tuple, List
import collections
import math
import torch


"""
                raw_audio_len
--------------------------------------------------|

        raw_audio                                 
==================================================

stride
-------->
        -------->
                -------->
                        -------->
                                -------->
                                        --------> left    right
                                                ooooooooo++++++++
"""


class AudioStreamer:
    def __init__(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: int,
        left_size: int,
        right_size: Optional[int] = None,
        stride: Optional[int] = None,
        pad_value: Optional[float] = None,
    ) -> None:
        """Simulates audio stream for complete audio input.

        Args:
            raw_audio (torch.Tensor): incoming raw audio (might be padded)
            raw_audio_len (int): effective length of the raw audio (unpadded)
            left_size (int): size of to be transcribed chunk portion.
            right_size (Optional[int], optional): size of future chunk portion (not supposed to 
                be transcribed but emitted anyway for context). Defaults to None.
            stride (Optional[int], optional): Distance in num. raw_audio samples between start's of two 
                consecutive chunk's. Can be interpreted as the buffer_size for already emitted audio (i.e. 
                stride = left_size + right_size means buffer_size = 0). Defaults to None.
            pad_value (Optional[float], optional): value to pad chunk's by if raw_audio_len reached but 
                chunk not full. Defaults to None.
        """

        assert raw_audio.dim() == 2

        self.raw_audio = raw_audio[:raw_audio_len]
        self.raw_audio_len = raw_audio_len

        self.left_size = left_size
        self.right_size = right_size if right_size is not None else 0
        self.stride = stride if stride is not None else self.left_size

        assert self.stride <= self.left_size + self.right_size, "Stride shouldn't be bigger than chunk size"
        
        self.pad_value = pad_value if pad_value is not None else 0.0

    def __iter__(self) -> Tuple[torch.Tensor, int]:
        """Every chunk returned should have same size (self.left_size + self.right_size). 
        In real streaming setting we don't know if last chunk was received until we wait 
        longer than total chunk size.

        Yields:
            Tuple[torch.Tensor, int]: chunk with shape (left_size + right_size, 1) and the effective 
                chunk_size (num. of non padded samples)
        """
        total_chunk_size = self.left_size + self.right_size

        for i in range(0, self.raw_audio_len, self.stride):
            if i + 1 > self.raw_audio_len:
                # stop if left side can't even have a single frame
                break

            elif i + total_chunk_size > self.raw_audio_len:
                # raw audio needs to be padded to get chunk of size `total_chunk_size`
                chunk = self.raw_audio[i:]
                pad_tensor = torch.full((total_chunk_size - chunk.shape[0], 1), self.pad_value, device=chunk.device)
                chunk_padded = torch.cat((chunk, pad_tensor), dim=0)

                yield chunk_padded, chunk.shape[0]

            else:
                # normal chunk
                chunk = self.raw_audio[i:i+total_chunk_size].clone()
                yield chunk, total_chunk_size
                
                
class AudioStreamerV2(AudioStreamer):
    def __init__(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: int,
            left_size: int,
            right_size: Optional[int] = None,
            stride: Optional[int] = None,
            pad_value: Optional[float] = None,
    ) -> None:
        super().__init__(raw_audio, raw_audio_len, left_size, right_size, stride, pad_value)

    def __iter__(self) -> Tuple[torch.Tensor, int, int]:
        """Every chunk returned should have same size (self.left_size + self.right_size).
        In real streaming setting we don't know if last chunk was received until we wait
        longer than total chunk size.

        Yields:
            Tuple[torch.Tensor, int, int]: chunk with shape (left_size + right_size, 1), the effective
                chunk_size (num. of non padded samples) and the time step
        """
        total_chunk_size = self.left_size + self.right_size

        for i in range(0, self.raw_audio_len, self.stride):
            if i + 1 > self.raw_audio_len:
                # stop if left side can't even have a single frame
                break

            elif i + total_chunk_size > self.raw_audio_len:
                # raw audio needs to be padded to get chunk of size `total_chunk_size`
                chunk = self.raw_audio[i:]
                pad_tensor = torch.full((total_chunk_size - chunk.shape[0], 1), self.pad_value, device=chunk.device)
                chunk_padded = torch.cat((chunk, pad_tensor), dim=0)

                yield chunk_padded, chunk.shape[0], i

            else:
                # normal chunk
                chunk = self.raw_audio[i:i+total_chunk_size].clone()
                yield chunk, total_chunk_size, i



class StreamingASRContextManager:
    def __init__(self, decoder, carryover_sz, beam_sz):
        """..."""

        self.decoder = decoder
        self.beam_sz = beam_sz
        self.carryover_sz = carryover_sz
        
        self.states = None
        self.hypothesis = None


    def __enter__(self):
        self.states = collections.deque(maxlen=math.ceil(self.carryover_sz))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.states = None

    def process_chunk(self, ext_chunk, eff_chunk_sz) -> Tuple[torch.Tensor, List]:
        """
        :param chunk: shape [C, 1]
        :param future_ctx: shape [R, 1]
        :param eff_chunk_sz: L <= C+R

        :return: state and current hypothesis
        """
        self.hypothesis, state = self.decoder.infer(
            input=ext_chunk,
            length=torch.tensor(eff_chunk_sz),
            beam_width=self.beam_sz,
            state=tuple(self.states) if len(self.states) > 0 else None,
            hypothesis=self.hypothesis,
        )

        self.states.append(state)

        return state, self.hypothesis
