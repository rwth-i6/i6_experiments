import numpy
from sisyphus import tk, Job, Task
from typing import Optional, Union

from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib import lexicon

from i6_experiments.users.rossenbach.lib.hdf import load_default_data


class ViterbiAlignmentToDurationsJob(Job):

    __sis_hash_exclude__ = {"blank_token": None}

    def __init__(
            self,
            viterbi_alignment_hdf: tk.Path,
            bliss_lexicon: tk.Path,
            returnn_root: Optional[tk.Path] = None,
            blank_token: Optional[Union[str, int]] = None,
            *,
            dataset_to_check=None,
            time_rqmt=2,
            mem_rqmt=4,
    ):
        """
        :param Path viterbi_alignment: Path to the alignment HDF produced by CTC/Viterbi
        :param Path bliss_lexicon: used to determine the epsilon and do some verification
        :param tk.Path|None returnn_root:
        :param blank_token: Value of the blank token in CTC, or the phoneme string of the lexicon.
            Will use the last phoneme-inventory index if not provided.
        :param tk.Path|None dataset_to_check:
        """
        self.viterbi_alignment_hdf = viterbi_alignment_hdf
        self.bliss_lexicon = bliss_lexicon
        self.returnn_root = returnn_root
        self.blank_token = blank_token
        self.check = dataset_to_check
        self.out_durations_hdf = self.output_path("durations.hdf")
        self.rqmt = {"time": time_rqmt, "mem": mem_rqmt}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        # Load HDF data
        alignment, tags = load_default_data(self.viterbi_alignment_hdf.get_path())
        durations_total = []

        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        if isinstance(self.blank_token, str):
            raise NotImplementedError
        elif isinstance(self.blank_token, int):
            skip_token = self.blank_token
        else:
            assert self.blank_token is None
            skip_token = len(lex.phonemes) - 1

        # sort based on tags
        if self.check is not None:
            check, check_tag = load_default_data(self.check.get_path())

        # Alignemnt to duration conversion
        for alignment_idx, s in enumerate(alignment):
            durations = []
            for idx, p in enumerate(s):
                # Skip_token only appears now if 2 labels following each other are the same.
                #   Example [1,skip_token,1] -> [1,1]
                if p == skip_token:
                    durations[-1] += 1
                    continue
                if idx != 0 and s[idx] != s[idx - 1]:
                    durations.append(1)
                elif idx != 0 and s[idx] == s[idx - 1]:
                    durations[-1] += 1
                else:
                    durations.append(1)
            # Check if lengths match if dataset is provided
            if self.check is not None:
                assert sum(durations) == len(check[alignment_idx]), (
                    f"durations {sum(durations)} and spectrogram length {len(check[alignment_idx])}"
                    f"do not match in length "
                )
            durations_total.append(durations)

        # Dump into HDF
        new_lengths = []
        for seq in durations_total:
            new_lengths.append([len(seq), 2, 2])
        duration_sequence = numpy.hstack(durations_total).astype(numpy.int32)
        dim = 1
        hdf_writer = get_returnn_simple_hdf_writer(self.returnn_root)
        writer = hdf_writer(self.out_durations_hdf.get_path(), dim=dim, ndim=2)
        offset = 0
        for tag, length in zip(tags, new_lengths):
            in_data = duration_sequence[offset : offset + length[0]]
            in_data = numpy.expand_dims(in_data, axis=1)
            offset += length[0]
            writer.insert_batch(numpy.asarray([in_data]), [in_data.shape[0]], [tag])
        print(f"Succesfully converted durations into {(self.out_durations_hdf.get_path())}")
        writer.close()

    @classmethod
    def hash(cls, parsed_args):
        d = {
            'viterbi_alignment_hdf': parsed_args['viterbi_alignment_hdf'],
            'returnn_root': parsed_args['returnn_root'],
            'blank_token': parsed_args['blank_token']
        }
        return super().hash(d)