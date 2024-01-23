import h5py
import numpy
from sisyphus import tk, Job, Task
from typing import Optional, Union

from i6_core.lib import lexicon
from i6_core.lib.hdf import get_returnn_simple_hdf_writer, get_input_dict_from_returnn_hdf
from i6_core.lib.rasr_cache import FileArchive
from i6_core.lib.corpus import Corpus

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


class ExtractDurationsFromRASRAlignmentJob(Job):
  """
  Takes given rasr alignment bundle and converts it to a hdf which contains the aligned durations
  """

  def __init__(
    self,
    rasr_alignment: tk.Path,
    rasr_allophones: tk.Path,
    bliss_corpus: tk.Path,
    returnn_root: tk.Path,
    target_duration_hdf: Optional[tk.Path] = None,
    time_rqmt: int = 2,
    mem_rqmt: int = 4,
    silence_token: str = "[SILENCE]",
    start_token: str = "[start]",
    end_token: str = "[end]",
    boundary_token: str = "[space]",
  ):
    """
    :param rasr_alignment: Path to the rasr alignment file.
    :param rasr_allophones: Path to the rasr allo file.
    :param bliss_corpus: Bliss corpus to possibly change label sequence based on the rasr alignment.
    :param target_duration_hdf: HDF File to compare alignment durations to (and modifies if difference)
    :param silence_token: boundary token of the given alignment
    :param start_token: start token to be inserted
    :param end_token: end token to be inserted
    :param boundary_token: boundary token to be inserted
    :param time_rqmt:
    :param mem_rqmt:
    """
    self.align = rasr_alignment
    self.bliss_corpus = bliss_corpus
    self.rasr_allophones = rasr_allophones
    self.target_duration_hdf = target_duration_hdf
    self.silence_token = silence_token
    self.boundary_token = boundary_token
    self.end_token = end_token
    self.start_token = start_token
    self.returnn_root = returnn_root

    self.out_durations_hdf = self.output_path("durations.hdf")
    self.out_bliss = self.output_path("labeled_corpus.xml.gz")

    self.rqmt = {"time": time_rqmt, "mem": mem_rqmt}

  def tasks(self):
    yield Task("run", rqmt=self.rqmt)

  def run(self):
    durations = []
    tags = []
    full_corpus_labels = {}
    empty_aligns = []
    if self.target_duration_hdf is not None:
      hdf_file = h5py.File(self.target_duration_hdf, "r")
      returnn_length_dict = get_input_dict_from_returnn_hdf(hdf_file=hdf_file)
    if self.align.get_path().endswith(".bundle"):
      files = open(self.align.get_path(), "rt")
      for cache in files:
        sprint_cache = FileArchive(cache.strip())
        sprint_cache.setAllophones(self.rasr_allophones.get_path())
        keys = [str(s) for s in sprint_cache.ft if not str(s).endswith(".attribs")]
        for key in keys:
          seq = []
          gmm_seq = []
          tags.append(key)
          start_time = 0
          space_insert = False
          alignment = [
            (a[0], sprint_cache.allophones[a[1]])
            for a in sprint_cache.read(key, "align")
          ]
          # Start computing duration sequence from rasr alignment
          last_allophone = None
          if len(alignment) == 0:
            empty_aligns.append(key)
            continue
          for time, allophone in alignment:
            phoneme = allophone.split("{", 1)[0].rstrip()
            # new sequence has boundary instead of silence token
            if phoneme == self.silence_token:
              phoneme = self.boundary_token
            # start of sequence
            if len(gmm_seq) == 0:
              gmm_seq.append(phoneme)
            # within phoneme or silence
            elif allophone == last_allophone:
              space_insert = False
            # Insert word boundary token between words where there is no silence
            elif (
              self.boundary_token not in gmm_seq[-1]
              and allophone != last_allophone
              and "@i" in allophone
              and phoneme != self.boundary_token
            ):
              seq.append(time - start_time)
              gmm_seq.append(self.boundary_token)
              seq.append(0)
              space_insert = True
              start_time = time
              gmm_seq.append(phoneme)
            # new phoneme but no word boundary
            elif allophone != last_allophone and space_insert is False:
              gmm_seq.append(phoneme)
              seq.append(time - start_time)
              start_time = time
            else:
              assert False, "Check your data, this should not be reached!"
            last_allophone = allophone
          # Add duration of last token
          if self.target_duration_hdf is not None:
            # take the difference between returnn feature extraction and rasr feature extraction in account
            difference = len(alignment) - returnn_length_dict[key]
            assert -1 <= difference <= 1, (
              "The difference must not be greater, remember to use center=False "
              "for the RETURNN feature extration",
              key
            )
          else:
            difference = 0
          seq.append(len(alignment) - start_time - difference)
          # Assert that number of durations fit the given alignment
          assert len(alignment) == (sum(seq) + difference), (
            key,
            len(alignment),
            sum(seq),
          )
          # manually add start/end token with duration 0
          seq = numpy.insert(seq, 0, 0)
          seq = numpy.append(seq, 0)
          gmm_seq.append(self.end_token)
          gmm_seq.insert(0, self.start_token)

          # Create new labels for corpus
          corpus_labels = gmm_seq
          # Assert that the number of labels fit the number of tokens in the duration sequence
          assert len(seq) == len(corpus_labels), (key, alignment, len(seq), len(corpus_labels))
          durations.append(seq)
          full_corpus_labels[key] = corpus_labels

    # Write durations to hdf file
    assert len(empty_aligns) == 0, (len(empty_aligns), empty_aligns)
    new_lengths = []
    for seq in durations:
      new_lengths.append([len(seq), 2, 2])
    duration_sequence = numpy.hstack(durations).astype(numpy.int32)
    dim = 1
    SimpleHDFWriter = get_returnn_simple_hdf_writer(self.returnn_root.get_path())
    writer = SimpleHDFWriter(self.out_durations_hdf.get_path(), dim=dim, ndim=2)
    offset = 0
    for tag, length in zip(tags, new_lengths):
      in_data = duration_sequence[offset : offset + length[0]]
      in_data = numpy.expand_dims(in_data, axis=1)
      offset += length[0]
      writer.insert_batch(numpy.asarray([in_data]), [in_data.shape[0]], [tag])
    writer.close()

    # Write new labels into the corpus such that label sequence fits the duration sequence
    bliss_corpus = Corpus()
    bliss_corpus.load(self.bliss_corpus.get_path())
    for s in bliss_corpus.segments():
      delimiter_str = " "
      new_orth = delimiter_str.join(full_corpus_labels[s.fullname()])
      s.orth = new_orth
    bliss_corpus.dump(self.out_bliss.get_path())