from typing import Union, Callable, Any
import h5py
import shutil
import os
import copy
import numpy as np

from sisyphus import Job, tk, Task
from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob

def map_sublabels_to_pseudo_label_indices(
    seq,
    is_eow_func: Callable[Any, bool],
    sil_idx=0
):
    """
    seq is a sequence of the form 
    [a, a, b, b, b, c#, c#, sil, sil,...]
    where a, b, c are sub-labels, and some of
    these sub-labels mark the EOW.

    This function turns all sub-labels belonging
    to the same word to the same index, representing
    one such word (though we might not know what that
    word is, for example in the phoneme case)

    Example:
    [a, a, b, b, b, c#, c#, sil, sil,...]
    becomes
    [1, 1, 1, 1, 1, 1, 1, sil, sil,...]
    It is not known what the exact word 1 is, but we
    know it is a word (true label)

    :param seq:
    :param is_eow_func: A function taking an input sublabel
    and return True if the sublabel marks EOW
    :param sil_idx: Index of silence, padding, etc.
    :returns: A numpy array which turns the sequence
    of sub-labels to the corresponding labels
    """
    res = np.empty_like(seq)
    word_idx = 0
    last_frame = sil_idx
    inside_word = False
    for i, frame in enumerate(seq):
        if frame == last_frame:
            if i > 0:
                res[i] = res[i-1]
            else:
                res[i] = sil_idx
        else:
            if last_frame == sil_idx:
                if inside_word:
                    res[i] = res[i-1]
                else:
                    word_idx += 1
                    inside_word = True
                    res[i] = word_idx
            else:
                if is_eow_func(last_frame):
                    if frame == sil_idx:
                        inside_word = False
                        res[i] = sil_idx
                    else:
                        word_idx += 1
                        inside_word = True
                        res[i] = word_idx
                else:
                    res[i] = res[i-1]
        last_frame = frame
    return res

class ConvertCTCPhonemeAlignmentsToPseudoWordAlignmentsJob(Job):
    """
    Convert the HDF of CTC phoneme alignments (39 phonemes
    + 39 EOW phonemes) to an "alignment" of pseudo words

    e.g. 0 0 0 0 1 1 1 1 1 0 2 2 2 2 2 2 2 2 0 0 3 3 3 0 0

    with 0 being silence and anything else being inside word boundaries.
    Same index is inside the same word. But we don't know the true words.
    This can only be used when the vocab has EOW symbols
    (maybe with modifications possible for BPE???)
    """

    def __init__(
        self,
        ctc_phoneme_alignments_hdf: Union[str, tk.Path],
        ctc_phoneme_vocab: Union[str, tk.Path],
        sil_idx: int = 0,
        cpu=4,
        mem=16,
        file_size=1000,
        time=4,
    ):
        """
        :param ctc_phoneme_alignments_hdf: Path to HDF of the CTC phoneme alignments.
            Format: "Standard" i6 HDF
        :param ctc_phoneme_vocab: Path to vocab of phonemes.
            Format: n lines, each line consists of <phoneme> <index>
            Needed to determine which phoneme if EOW.
        :param sil_idx: Silence index in the alignments. Will be kept
            the same in the output.
        """
        self.ctc_phoneme_alignments_hdf = ctc_phoneme_alignments_hdf
        self.ctc_phoneme_vocab = ctc_phoneme_vocab
        self.sil_idx = sil_idx
        self.rqmt = {
            "cpu": cpu,
            "mem": mem,
            "file_size": file_size,
            "time": time,
        }
        self.out_hdf = self.output_path("pseudo_word_alignments.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)
    
    def run(self):
        is_eow = {}
        with open(self.ctc_phoneme_vocab, "r") as vocab:
            for line in vocab.readlines():
                phoneme, idx = line.split()
                idx = int(idx)
                if phoneme[-1] == "#": # adapt this to change for BPE alignments
                    is_eow[idx] = True
                else:
                    is_eow[idx] = False
        
        # now loop over the alignments and determine word boundaries
        temp_path = "temp_out.hdf"
        shutil.copyfile(self.ctc_phoneme_alignments_hdf, temp_path)
        temp = h5py.File(temp_path, "a")
        seq_lens = temp["/seqLengths"][:, 0]

        seq_start = 0
        for seq_len in seq_lens:
            seq = temp["/inputs"][seq_start:seq_start+seq_len]
            res = map_sublabels_to_pseudo_label_indices(seq, is_eow_func=lambda x: is_eow[x], sil_idx=self.sil_idx)
            temp["/inputs"][seq_start:seq_start+seq_len] = res
            seq_start = seq_start + seq_len
        
        temp.close()
        shutil.copyfile(temp_path, self.out_hdf)
        os.remove(temp_path)
        


    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["cpu"]
        del parsed_args["mem"]
        del parsed_args["file_size"]
        del parsed_args["time"]
        return super().hash(parsed_args)

class ConvertGMMAlignmentIndicesToCTCPhonemeIndices(Job):
    """
    Convert the HDF of GMM alignments with 3-state indices to
    CTC phoneme indices.
    Requirements should be similar to dump ReturnnDumpHDFJob
    """
    def __init__(
        self,
        gmm_alignments_hdf: Union[str, tk.Path],
        gmm_state_tying: Union[str, tk.Path],
        ctc_state_tying: Union[str, tk.Path],
        gmm_silence_phone="[SILENCE]",
        ctc_blank="<blank>",
        cpu=4,
        mem=16,
        file_size=1000,
        time=4,
    ):
        self.gmm_alignments_hdf = gmm_alignments_hdf
        self.gmm_state_tying = gmm_state_tying
        self.ctc_state_tying = ctc_state_tying
        self.gmm_silence_phone = gmm_silence_phone
        self.ctc_blank = ctc_blank
        self.rqmt = {
            "cpu": cpu,
            "mem": mem,
            "file_size": file_size,
            "time": time,
        }
        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt = self.rqmt)
    
    def run(self):
        # dict mapping gmm state index -> phoneme (each phoneme 3 states in gmm)
        # line format: [SILENCE]{#+#}@i@f.0 117
        gmm_idx_to_phoneme = {}
        with open(self.gmm_state_tying, "r") as gmm_state_indices:
            for line in gmm_state_indices.readlines():
                center = line.split("{")[0]
                # Check if EOW
                if "@f" in line and center != self.gmm_silence_phone:
                    phoneme = center + "#"
                else:
                    phoneme = center
                if phoneme == self.gmm_silence_phone:
                    phoneme = self.ctc_blank
                idx = int(line.split()[-1])
                gmm_idx_to_phoneme[idx] = phoneme
        print("GMM indices to phonemes: ", gmm_idx_to_phoneme)

        # mapping phoneme -> ctc phoneme index
        # line format: AA{#+#}.0 1
        ctc_phoneme_to_idx = {}
        with open(self.ctc_state_tying, "r") as ctc_phoneme_idx:
            for line in ctc_phoneme_idx.readlines():
                phoneme = line.split("{")[0]
                idx = int(line.split()[-1])
                ctc_phoneme_to_idx[phoneme] = idx
        print("CTC phonemes to indices: ", ctc_phoneme_to_idx)

        index_mapping = {idx: ctc_phoneme_to_idx[gmm_idx_to_phoneme[idx]] for idx in gmm_idx_to_phoneme}
        print("GMM phoneme index to CTC phoneme index: ", index_mapping)

        # open the gmm alignment hdf, replace the indices and dump it
        # silence will become blank (0)
        # gmm = h5py.File(self.gmm_alignments_hdf, "r")
        temp_path = "temp_out.hdf"
        shutil.copyfile(self.gmm_alignments_hdf, temp_path)
        temp = h5py.File(temp_path, "a")
        arr = temp["/inputs"][:]
        map_func = lambda idx: index_mapping[idx]
        vec_map_func = np.vectorize(map_func) # this makes life good. for-loop does not
        arr_mapped = vec_map_func(arr)
        temp["/inputs"][:] = arr_mapped
        temp.close()
        shutil.copyfile(temp_path, self.out_hdf)
        os.remove(temp_path)

def get_gmm_pseudo_word_alignments():
    job = ConvertCTCPhonemeAlignmentsToPseudoWordAlignmentsJob(
        ctc_phoneme_alignments_hdf="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/gmm_alignments_with_ctc_phonemes.hdf",
        ctc_phoneme_vocab="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/ctc_phoneme_vocab.txt",
    )
    tk.register_output("gmm_pseudo_word_alignments/gmm_pseudo_word_alignments.hdf", job.out_hdf)
    return job

def get_ted2_gmm_pseudo_word_alignments():
    from i6_experiments.users.berger.systems.dataclasses import AlignmentData
    ted2_gmm_alignments_hdf=AlignmentData(
        alignment_cache_bundle=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.GHU5l8jskqKJ/output/alignment.cache.bundle"),
        allophone_file=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_work_dir/i6_core/lexicon/allophones/StoreAllophonesJob.GP1idt9FdeWe/output/allophones"),
        state_tying_file=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_export_setup/output/baselines/tedlium2/gmm/common_baseline/train_cart_mono_state_tying"),
        silence_phone="[SILENCE]",
    ).get_hdf(
        returnn_python_exe="/work/tools/users/zeyer/py-envs/py3.11-torch2.1/bin/python3.11",
        returnn_root="/u/minh-nghia.phan/tools/albert_returnn_2024_05_31",
    )
    ted2_ctc_phonemes_alignments_hdf = ConvertGMMAlignmentIndicesToCTCPhonemeIndices(
        gmm_alignments_hdf=ted2_gmm_alignments_hdf,
        gmm_state_tying=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_export_setup/output/baselines/tedlium2/gmm/common_baseline/train_cart_mono_state_tying"),
        ctc_state_tying=tk.Path("/u/minh-nghia.phan/setups/torch/librispeech/work/i6_core/lexicon/allophones/DumpStateTyingJob.vfz3NEyDiM05/output/state-tying"),
    ).out_hdf
    convert_job = ConvertCTCPhonemeAlignmentsToPseudoWordAlignmentsJob(
        ctc_phoneme_alignments_hdf=ted2_ctc_phonemes_alignments_hdf,
        ctc_phoneme_vocab="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/ctc_phoneme_vocab.txt",
    )
    return convert_job


# def get_ted2_gmm_pseudo_word_alignments_v2():
#     from i6_experiments.users.berger.systems.dataclasses import AlignmentData
#     ted2_gmm_alignments_hdf = RasrAlignmentDumpHDFJob(
#         alignment_caches=[tk.Path("/work/common/asr/tedliumv2/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.GHU5l8jskqKJ/output/alignment.cache.bundle")],
#         allophone_file=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_work_dir/i6_core/lexicon/allophones/StoreAllophonesJob.GP1idt9FdeWe/output/allophones"),
#         state_tying_file=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_export_setup/output/baselines/tedlium2/gmm/common_baseline/train_cart_mono_state_tying"),
#         returnn_root="/u/minh-nghia.phan/tools/albert_returnn_2024_05_31",
#     ).out_hdf_files[0]
#     ted2_ctc_phonemes_alignments_hdf = ConvertGMMAlignmentIndicesToCTCPhonemeIndices(
#         gmm_alignments_hdf=ted2_gmm_alignments_hdf,
#         gmm_state_tying=tk.Path("/work/common/asr/tedliumv2/data/sisyphus_export_setup/output/baselines/tedlium2/gmm/common_baseline/train_cart_mono_state_tying"),
#         ctc_state_tying=tk.Path("/u/minh-nghia.phan/setups/torch/librispeech/work/i6_core/lexicon/allophones/DumpStateTyingJob.vfz3NEyDiM05/output/state-tying"),
#     ).out_hdf
#     convert_job = ConvertCTCPhonemeAlignmentsToPseudoWordAlignmentsJob(
#         ctc_phoneme_alignments_hdf=ted2_ctc_phonemes_alignments_hdf,
#         ctc_phoneme_vocab="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/ctc_phoneme_vocab.txt",
#     )
#     return convert_job

# def py():
#     job = get_gmm_pseudo_word_alignments()
#     tk.register_output("gmm_pseudo_word_alignments/gmm_pseudo_word_alignments.hdf", job.out_hdf)

def py():
    job = get_ted2_gmm_pseudo_word_alignments()
    tk.register_output("ted2_gmm_pseudo_word_alignments/ted2_gmm_pseudo_word_alignments.hdf", job.out_hdf)
    
if __name__ == "__main__": # Simply for testing
    job = ConvertCTCPhonemeAlignmentsToPseudoWordAlignmentsJob(
        ctc_phoneme_alignments_hdf="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/gmm_alignments_with_ctc_phonemes.hdf",
        ctc_phoneme_vocab="/work/asr3/zyang/share/mnphan/alignment_data/lbs960/ctc_phoneme_vocab.txt",
    )
    job.run()
