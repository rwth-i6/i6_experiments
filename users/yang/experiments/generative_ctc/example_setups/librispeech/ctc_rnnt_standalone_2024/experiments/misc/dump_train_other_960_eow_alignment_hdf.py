import json
from collections import OrderedDict

from sisyphus import Job, Task, tk

from i6_core.returnn import ReturnnDumpHDFJob


class GenerateEowMonophoneStateTyingJob(Job):
    def __init__(
        self,
        *,
        allophone_file: tk.Path,
        phoneme_file: tk.Path,
        silence_phone: str = "[SILENCE]",
        num_states_per_phone: int = 3,
        eow_suffix: str = "#",
    ):
        self.allophone_file = allophone_file
        self.phoneme_file = phoneme_file
        self.silence_phone = silence_phone
        self.num_states_per_phone = num_states_per_phone
        self.eow_suffix = eow_suffix

        self.out_state_tying = self.output_path("state-tying")
        self.out_phoneme_to_idx_json = self.output_path("phoneme_to_idx.json")
        self.out_phoneme_to_idx_txt = self.output_path("phoneme_to_idx.txt")
        self.out_idx_to_phoneme_txt = self.output_path("idx_to_phoneme.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _is_special_phone(phone: str) -> bool:
        return phone.startswith("[") and phone.endswith("]")

    @staticmethod
    def _read_non_comment_lines(path: tk.Path):
        with open(path.get_path(), "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line

    def run(self):
        base_phones = []
        seen_base_phones = set()
        for phoneme in self._read_non_comment_lines(self.phoneme_file):
            base_phones.append(phoneme)
            seen_base_phones.add(phoneme)

        allophones = list(self._read_non_comment_lines(self.allophone_file))
        allophone_phones = []
        for allophone in allophones:
            phone = allophone.split("{", 1)[0]
            if phone not in seen_base_phones:
                base_phones.append(phone)
                seen_base_phones.add(phone)
            allophone_phones.append(phone)

        phoneme_to_idx = OrderedDict()
        for phone in base_phones:
            phoneme_to_idx[phone] = len(phoneme_to_idx)
        for phone in base_phones:
            if phone == self.silence_phone or self._is_special_phone(phone):
                continue
            phoneme_to_idx[f"{phone}{self.eow_suffix}"] = len(phoneme_to_idx)

        with open(self.out_state_tying.get_path(), "w") as f:
            for allophone, phone in zip(allophones, allophone_phones):
                if phone == self.silence_phone or self._is_special_phone(phone):
                    label = phone
                elif "@f" in allophone:
                    label = f"{phone}{self.eow_suffix}"
                else:
                    label = phone
                label_idx = phoneme_to_idx[label]
                for state_idx in range(self.num_states_per_phone):
                    f.write(f"{allophone}.{state_idx} {label_idx}\n")

        with open(self.out_phoneme_to_idx_json.get_path(), "w") as f:
            json.dump(phoneme_to_idx, f, indent=2, sort_keys=False)

        with open(self.out_phoneme_to_idx_txt.get_path(), "w") as f:
            for phoneme, idx in phoneme_to_idx.items():
                f.write(f"{phoneme} {idx}\n")

        with open(self.out_idx_to_phoneme_txt.get_path(), "w") as f:
            for phoneme, idx in phoneme_to_idx.items():
                f.write(f"{idx} {phoneme}\n")


def py():
    returnn_root = tk.Path(
        "/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/tools/20241021_returnn/returnn",
        hash_overwrite="/u/berger/repositories/returnn/",
    )
    returnn_python_exe = tk.Path("/usr/bin/python3")

    allophone_file = tk.Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    phoneme_file = tk.Path("/work/common/asr/librispeech/data/common/phonemes/phonemes.folded.txt")

    eow_state_tying_job = GenerateEowMonophoneStateTyingJob(
        allophone_file=allophone_file,
        phoneme_file=phoneme_file,
        silence_phone="[SILENCE]",
        num_states_per_phone=3,
    )
    tk.register_output("lbs_mono_phone_eow/state-tying", eow_state_tying_job.out_state_tying)
    tk.register_output("lbs_mono_phone_eow/phoneme_to_idx.json", eow_state_tying_job.out_phoneme_to_idx_json)
    tk.register_output("lbs_mono_phone_eow/phoneme_to_idx.txt", eow_state_tying_job.out_phoneme_to_idx_txt)
    tk.register_output("lbs_mono_phone_eow/idx_to_phoneme.txt", eow_state_tying_job.out_idx_to_phoneme_txt)

    for i in range(1, 201):
        filename = (
            f"/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/"
            f"AlignmentJob.ag68hD1B5N6T/output/alignment.cache.{i}"
        )
        dataset_config = {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": filename,
                    "data_type": "align",
                    "allophone_labeling": {
                        "allophone_file": allophone_file,
                        "silence_phone": "[SILENCE]",
                        "state_tying_file": eow_state_tying_job.out_state_tying,
                    },
                }
            },
        }

        hdf_file = ReturnnDumpHDFJob(
            dataset_config,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        ).out_hdf
        tk.register_output(f"lbs_mono_phone_eow/alignment_{i}.hdf", hdf_file)
