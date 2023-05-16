from sisyphus import Job, Path, Task, tk, gs

import sys, os
import textwrap
import tempfile
import inspect

import numpy as np

from i6_core.returnn import ReturnnConfig

class _HelperConfig(ReturnnConfig):
    PYTHON_CODE = textwrap.dedent(
        """\
        ${SUPPORT_CODE}

        ${PROLOG}
    
        ${REGULAR_CONFIG}
    
        ${EPILOG}
        """
    )


class TransformerConfig:
    def call(self):
        raise NotImplementedError()
    
    def write_to_file(self, file):
        with open(file, "w") as f:
            f.write(inspect.getsource(self.__class__))
            f.write("\n" + repr(self))

from i6_core.util import get_val

class ShiftAlign(TransformerConfig):
    def __init__(self, shift, pad):
        self.shift = shift
        self.pad = pad

    def call(self, alignment):
        pad = get_val(self.pad)
        data = np.pad(alignment[self.shift:], (0, self.shift), 'constant', constant_values=(0, pad))
        return data

class MoveSilence(TransformerConfig):
    def __init__(self, silence_idx, start=True):
        self.silence_idx = silence_idx
        self.start = start

    def call(self, alignment):
        pad = get_val(self.silence_idx)
        # find first non-silence index
        if self.start:
            (idx, *_), *_ = np.where(alignment != self.silence_idx)
        else:
            *_, (idx, *_) = np.where(alignment != self.silence_idx)
        data = np.roll(alignment, -idx if self.start else idx)
        return data

class ReplaceSilenceByLastSpeech(TransformerConfig):
    def __init__(self, silence_idx):
        self.silence_idx = silence_idx
    
    @staticmethod
    def roll_first_segment(seq, silence, first_speech_end, first_silence_end):
        subseq = seq[:first_speech_end+1]
        subseq = np.roll(subseq, -first_silence_end)
        return np.concatenate((subseq, seq[first_speech_end+1:]))

    def replace_silence(self, seq, silence_idx):
        silence = seq == silence_idx
        pseq = np.pad(silence, (1, 1), "constant", constant_values=False)
        ri = np.where(np.logical_and(np.logical_not(pseq[:-2]), silence))[0]
        re = np.where(np.logical_and(silence, np.logical_not(pseq[2:])))[0]
        fillers = np.repeat(seq[ri-1], re - ri + 1)
        res = seq.copy()
        np.place(res, silence, fillers)
        return res

    def call(self, alignment):
        sidx = get_val(self.silence_idx)
        silence = alignment == sidx
        pseq = np.pad(silence, (1, 1), "constant", constant_values=True)
        first_speech_end = np.argmax(np.logical_and(np.logical_not(silence), pseq[2:]))
        first_silence_end = np.argmax(np.logical_and(pseq[:-2], np.logical_not(silence)))
        rseq = self.roll_first_segment(alignment, silence, first_speech_end, first_silence_end)
        return self.replace_silence(rseq, sidx)

class SqueezeSpeech(TransformerConfig):
    def __init__(self, silence_idx, repeat=1, offset=0):
        assert 0 <= offset <= 1 and isinstance(offset, (float, int))
        assert repeat >= 1 and isinstance(repeat, int)
        self.silence_idx = silence_idx
        self.repeat = repeat
        self.offset = offset

    def squeeze_speech(self, seq, repeat=1, offset=0):
        silence_idx = get_val(self.silence_idx)
        is_speech = seq != silence_idx
        first_speech_mask = np.logical_and(seq != np.roll(seq, -1), is_speech)
        speech_seq = seq[first_speech_mask]
        if repeat > 1:
            speech_seq = np.repeat(speech_seq, repeat)
        diff = len(seq) - len(speech_seq)
        assert diff >= 0
        begin = int(diff * offset)
        end = diff - begin
        out = np.pad(speech_seq, (begin, end), "constant", constant_values=silence_idx)
        return out
    
    def call(self, alignment):
        return self.squeeze_speech(alignment, self.repeat, self.offset)
    
    def __repr__(self):
        return f"SqueezeSpeech(silence_idx={self.silence_idx}, repeat={self.repeat}, offset={self.offset})"


class TransformAlignmentJob(Job):
    def __init__(
        self,
        transformer_config,
        alignment,
        start_seq=0,
        end_seq=float("inf"),
        returnn_root=None,
    ):
        assert isinstance(transformer_config, TransformerConfig)
        self.transformer_config = transformer_config
        self.alignment = alignment
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.returnn_root = returnn_root or gs.RETURNN_ROOT

        self.out_alignment = self.output_path("out.hdf")
        self.out_config_file = self.output_path("transformer.config")

        self.rqmts = {"cpu": 1, "mem": 8, "time": 2}
    
    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmts)
    
    def create_files(self):
        """Writes class implementation and object to file."""
        self.transformer_config.write_to_file(self.out_config_file.get_path())
    
    def run(self):
        sys.path.append(self.returnn_root)
        import returnn.datasets.hdf as rnn
        import numpy as np
        dataset = rnn.HDFDataset(
            files=[self.alignment],
            use_cache_manager=True
        )
        dataset.init_seq_order(epoch=1)

        # (fd, tmp_hdf_file) = tempfile.mkstemp(prefix=gs.TMP_PREFIX, suffix=".hdf")
        # os.close(fd)
        
        hdf_writer = rnn.SimpleHDFWriter(
            self.out_alignment.get_path(),
            dim=dataset.get_data_dim("classes"),
            ndim=1
        )
        
        seq_idx = self.start_seq
        while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= self.end_seq:
            # seq_idxs += [seq_idx]
            dataset.load_seqs(seq_idx, seq_idx + 1)
            alignment = dataset.get_data(seq_idx, "classes")
            seq_tag = dataset.get_tag(seq_idx)

            out_alignment = self.transformer_config.call(alignment)

            hdf_writer.insert_batch(
                np.array([out_alignment]),
                np.array([len(out_alignment)]),
                np.array([seq_tag])
            )

            seq_idx += 1
        
        hdf_writer.close()
        # shutil.move(tmp_hdf_file, self.out_alignment.get_path())

