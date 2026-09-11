import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class VarianceCallback(ForwardCallbackIface):
    def __init__(
        self,
        *,
        out_dir: str = "variance",
        audio_output: str = "audio_states",
        text_output: str = "text_states",
        num_seqs_to_collect: int = 1000,
        vocab=None,
        **kwargs,
    ):
        self.out_dir = out_dir
        self.audio_output = audio_output
        self.text_output = text_output
        self.num_seqs_to_collect = num_seqs_to_collect

        self._audio_seq_means: List[np.ndarray] = []
        self._text_seq_means: List[np.ndarray] = []

    def init(self, *args, **kwargs):
        os.makedirs(self.out_dir, exist_ok=True)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        if self.audio_output in outputs.data:
            audio = np.asarray(outputs[self.audio_output].raw_tensor, dtype=np.float32)
            if audio.shape[0] > 0 and len(self._audio_seq_means) < self.num_seqs_to_collect:
                self._audio_seq_means.append(audio.mean(axis=0))
                
        if self.text_output in outputs.data:
            text = np.asarray(outputs[self.text_output].raw_tensor, dtype=np.float32)
            if text.shape[0] > 0 and len(self._text_seq_means) < self.num_seqs_to_collect:
                self._text_seq_means.append(text.mean(axis=0))

    def finish(self, **kwargs):
        audio_means_np = np.stack(self._audio_seq_means) if self._audio_seq_means else np.array([])
        text_means_np = np.stack(self._text_seq_means) if self._text_seq_means else np.array([])
        
        with open(os.path.join(self.out_dir, "variance_summary.txt"), "w") as var_f:
            var_f.write(f"num_audio_seqs={len(audio_means_np)}\n")
            var_f.write(f"num_text_seqs={len(text_means_np)}\n\n")

            if len(audio_means_np) > 1:
                audio_var = np.var(audio_means_np, axis=0).sum()
                var_f.write(f"Audio variance (trace of covariance over {len(audio_means_np)} seqs): {audio_var:.4f}\n")
            
            if len(text_means_np) > 1:
                text_var = np.var(text_means_np, axis=0).sum()
                var_f.write(f"Text variance (trace of covariance over {len(text_means_np)} seqs): {text_var:.4f}\n")
            
            if len(audio_means_np) > 1 and len(text_means_np) > 1:
                mixed_means_np = np.concatenate([audio_means_np, text_means_np], axis=0)
                mixed_var = np.var(mixed_means_np, axis=0).sum()
                var_f.write(f"Mixed variance (trace of covariance over {len(mixed_means_np)} seqs): {mixed_var:.4f}\n")
