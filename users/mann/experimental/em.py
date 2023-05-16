from sisyphus import tk, Job, Task

import sys

class FindTransitionModelMaximumJob(Job):

    def __init__(
        self,
        transition_model_weights: tk.Path,
        silence_idx: tk.Variable,
        returnn_root: tk.Path,
    ):
        self.transition_model_weights = transition_model_weights
        self.silence_idx = silence_idx
        self.returnn_root = returnn_root

        self.out_speech_fwd = self.output_var("speech_fwd")
        self.out_silence_fwd = self.output_var("silence_fwd")
    
    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        sys.path.append(self.returnn_root.get())
        import returnn.datasets.hdf as rnn
        import numpy as np
        dataset = rnn.HDFDataset(
            files=[self.transition_model_weights.get()],
            use_cache_manager=True
        )
        dataset.init_seq_order(epoch=1)
        nos = dataset.num_seqs
        dataset.load_seqs(0, nos)

        w = 0
        for seq_idx in range(nos):
            print(dataset.get_tag(seq_idx))
            o = dataset.get_data(seq_idx, key="data")
            w += o
    
        silence_idx = self.silence_idx.get() 
        sil_fwd = w[silence_idx,0] / w[silence_idx,:].sum()
        self.out_silence_fwd.set(sil_fwd)

        ws = w.sum(axis=0) - w[silence_idx,:]
        speech_fwd = ws[0] / ws.sum()
        self.out_speech_fwd.set(speech_fwd)


class TransitionModelFromCountsJob(Job):

    def __init__(
        self,
        transition_model_counts: tk.Path,
        silence_idx: tk.Variable,
    ):
        self.transition_model_counts = transition_model_counts
        self.silence_idx = silence_idx

        self.out_speech_fwd = self.output_var("speech_fwd")
        self.out_silence_fwd = self.output_var("silence_fwd")
    
    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import  numpy as np

        w = np.loadtxt(self.transition_model_counts.get())

        silence_idx = self.silence_idx.get() 
        sil_fwd = w[silence_idx,0] / w[silence_idx,:].sum()
        self.out_silence_fwd.set(sil_fwd)

        ws = w.sum(axis=0) - w[silence_idx,:]
        speech_fwd = ws[0] / ws.sum()
        self.out_speech_fwd.set(speech_fwd)
