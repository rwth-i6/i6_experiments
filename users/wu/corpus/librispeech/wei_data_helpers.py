from sisyphus import tk, Job, Task
from i6_core.returnn import ReturnnDumpHDFJob


class PeakyAlignmentJob(Job):
    def __init__(self, dataset_hdf: tk.Path) -> None:
        self.dataset_hdf = dataset_hdf
        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py

        hdf_file = h5py.File(self.dataset_hdf)

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        peaky_inputs = hdf_file["inputs"][:]
        for idx in range(len(peaky_inputs) - 1):
            if peaky_inputs[idx] == peaky_inputs[idx + 1]:
                peaky_inputs[idx] = 0

        out_hdf.create_dataset("inputs", data=peaky_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        hdf_file.copy(hdf_file["seqTags"], out_hdf, "seqTags")
        hdf_file.copy(hdf_file["seqLengths"], out_hdf, "seqLengths")
        hdf_file.copy(hdf_file["targets"], out_hdf, "targets")


class CorrectTrainSegmentNamesJob(Job):
    def __init__(self, dataset_hdf: tk.Path) -> None:
        self.dataset_hdf = dataset_hdf
        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py

        hdf_file = h5py.File(self.dataset_hdf)

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        seq_tags = [
            f"train-other-960/{parts[1]}-{parts[2]}/{parts[1]}-{parts[2]}".encode()
            for seq_tag in hdf_file["seqTags"][:]
            for parts in [seq_tag.decode().split("/")]
        ]

        hdf_file.copy(hdf_file["inputs"], out_hdf, "inputs")
        out_hdf.create_dataset("seqTags", data=seq_tags)
        for attr_key, attr_val in hdf_file["seqTags"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        hdf_file.copy(hdf_file["seqLengths"], out_hdf, "seqLengths")
        hdf_file.copy(hdf_file["targets"], out_hdf, "targets")


class MatchLengthsJob(Job):
    def __init__(self, data_hdf: tk.Path, match_hdf: tk.Path) -> None:
        self.data_hdf = data_hdf
        self.match_hdf = match_hdf

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py
        import numpy as np

        hdf_file = h5py.File(self.data_hdf)
        match_hdf_file = h5py.File(self.match_hdf)

        out_hdf = h5py.File(self.out_hdf, "w")
        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        match_length_map = dict(zip(match_hdf_file["seqTags"], [length[0] for length in match_hdf_file["seqLengths"]]))

        matched_inputs = []
        matched_lengths = []
        current_begin_pos = 0
        num_mismatches = 0
        for tag, length in zip(hdf_file["seqTags"], [length[0] for length in hdf_file["seqLengths"]]):
            target_length = match_length_map.get(tag, length)

            if target_length == length:
                matched_seq = hdf_file["inputs"][current_begin_pos : current_begin_pos + length]
            elif length < target_length:
                pad_value = hdf_file["inputs"][current_begin_pos + length - 1]
                pad_list = np.array([pad_value for _ in range(target_length - length)])
                matched_seq = np.concatenate(
                    [hdf_file["inputs"][current_begin_pos : current_begin_pos + length], pad_list], axis=0
                )
                print(
                    f"Length for segment {tag} is shorter ({length}) than the target ({target_length}). Append {pad_list}."
                )
                num_mismatches += 1
            else:  # length > target_length
                print(
                    f"Length for segment {tag} is longer ({length}) than the target ({target_length}). Cut off {hdf_file['inputs'][current_begin_pos + target_length : current_begin_pos + length]}."
                )
                matched_seq = hdf_file["inputs"][current_begin_pos : current_begin_pos + target_length]
                num_mismatches += 1

            assert len(matched_seq) == target_length
            matched_inputs.extend(matched_seq)
            matched_lengths.append([target_length])
            current_begin_pos += length

        print(f"Finished processing. Corrected {num_mismatches} mismatched lengths in total.")

        matched_inputs = np.array(matched_inputs, dtype=hdf_file["inputs"].dtype)
        matched_lengths = np.array(matched_lengths, dtype=hdf_file["seqLengths"].dtype)

        out_hdf.create_dataset("inputs", data=matched_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        hdf_file.copy(hdf_file["seqTags"], out_hdf, "seqTags")
        out_hdf.create_dataset("seqLengths", data=matched_lengths)
        for attr_key, attr_val in hdf_file["seqLengths"].attrs.items():
            out_hdf["seqLengths"].attrs[attr_key] = attr_val
        hdf_file.copy(hdf_file["targets"], out_hdf, "targets")


def get_wei_train_alignment(length_reference_hdf: tk.Path, returnn_python_exe: tk.Path, returnn_root: tk.Path):
    train_align_hdf = ReturnnDumpHDFJob(
        {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": tk.Path(
                        "/work/asr4/berger/dependencies/librispeech/alignments/train-960_conformer-ctc_wei/alignment.cache.bundle"
                    ),
                    "data_type": "align",
                    "allophone_labeling": {
                        "silence_phone": "[SILENCE]",
                        "allophone_file": tk.Path(
                            "/work/asr4/berger/dependencies/librispeech/alignments/train-960_conformer-ctc_wei/allophones"
                        ),
                        "state_tying_file": tk.Path(
                            "/work/asr4/berger/dependencies/librispeech/alignments/train-960_conformer-ctc_wei/state-tying"
                        ),
                    },
                },
            },
        },
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_hdf

    train_align_hdf = PeakyAlignmentJob(train_align_hdf).out_hdf
    train_align_hdf = CorrectTrainSegmentNamesJob(train_align_hdf).out_hdf
    train_align_hdf = MatchLengthsJob(train_align_hdf, length_reference_hdf).out_hdf

    return train_align_hdf
