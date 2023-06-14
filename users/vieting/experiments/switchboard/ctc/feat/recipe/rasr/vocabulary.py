from sisyphus import Job, Task

from i6_core.util import uopen


class GenerateLabelFileFromStateTyingJob(Job):
    def __init__(self, state_tying_file, use_blank=False, add_eow=False, add_sow=False):
        """

        :param state_tying_file:
        """
        self.state_tying_file = state_tying_file
        self.use_blank = use_blank
        self.add_eow = add_eow
        self.add_sow = add_sow

        self.out_label_file = self.output_path("label_vocab_dict")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """Main function of the class, responsible for reading the state tying file, processing the data,
        and writing the output label file.

        The state tying file is read and processed to extract phonemes and their corresponding indices.
        The resulting data is then written to the output label file.
        """
        # A dictionary to store the state tying information
        state_tying_dict = {}

        # Open the state tying file for reading
        with uopen(self.state_tying_file, "r") as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                tokens = line.split()
                assert len(tokens) >= 2, line

                # Add the information to the dictionary
                state_tying_dict[tokens[0]] = " ".join(tokens[1:])

        # Dictionaries to store the start of word (sow) phonemes, end of word (eow) phonemes,
        # and the mapping between phonemes and their indices
        eow_phons = {}
        sow_phons = {}
        vocab_dict = {}

        # A variable to store the maximum index value
        max_id = 0

        # Process the state tying information
        for allo, index in state_tying_dict.items():
            phon = allo.split("{")[0]

            # Check if the phoneme is special
            if not phon.startswith("[") and not phon.endswith("]"):
                if self.add_eow and ("@f" in allo):
                    phon += "#"
                    if index in eow_phons:
                        # Ensure that there is no conflict in the eow phonemes
                        assert eow_phons[int(index)] == phon
                    else:
                        eow_phons[int(index)] = phon
                    continue
                elif self.add_sow and ("@i" in allo):
                    phon = "#" + phon
                    if index in sow_phons:
                        # Ensure that there is no conflict in the sow phonemes
                        assert sow_phons[int(index)] == phon
                    else:
                        sow_phons[int(index)] = phon
                    continue

            # Check if the phoneme is already in the dictionary
            if phon in vocab_dict:
                # Ensure that the indices are unique for each phoneme
                assert (
                    vocab_dict[phon] == index
                ), f"index conflict for {phon}: {index} vs. {vocab_dict[phon]} ({allo} {index})"
            else:
                vocab_dict[phon] = index

            # Update the maximum index value
            if int(index) > max_id:
                max_id = int(index)

        for index in sorted(sow_phons.keys()):
            max_id += 1
            vocab_dict[sow_phons[index]] = str(max_id)
        for index in sorted(eow_phons.keys()):
            max_id += 1
            vocab_dict[eow_phons[index]] = str(max_id)

        if self.use_blank:
            blank_key = None
            blank_index = None
            silence_key = None
            silence_index = None
            for key, index in vocab_dict.items():
                if "[blank]" in key.lower():
                    blank_key = key
                    blank_index = index
                if "silence" in key.lower():
                    silence_key = key
                    silence_index = index

            if blank_index is not None:
                vocab_dict["<blank>"] = blank_index
                del vocab_dict[blank_key]
            elif silence_index is not None:
                vocab_dict["<blank>"] = silence_index
                del vocab_dict[silence_key]
            else:
                vocab_dict["<blank>"] = max_id + 1

        with uopen(self.out_label_file, "w") as f:
            for v, idx in sorted(vocab_dict.items()):
                f.write(f"{v} {idx}\n")

        assert (
            len(set(vocab_dict.values())) == max_id + 1
        ), "expected number of classes %d" % (max_id + 1)


class GenerateLabelFileFromStateTyingJobV2(Job):
    def __init__(self, state_tying_file):
        """

        :param state_tying_file:
        """
        self.state_tying_file = state_tying_file

        self.out_label_file = self.output_path("label_vocab_dict")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """Main function of the class, responsible for reading the state tying file, processing the data,
        and writing the output label file.

        The state tying file is read and processed to extract phonemes and their corresponding indices.
        The resulting data is then written to the output label file.
        """
        # A dictionary to store the state tying information
        state_tying_dict = {}

        # Open the state tying file for reading
        with uopen(self.state_tying_file, "r") as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                tokens = line.split()
                assert len(tokens) == 2, line

                # Add the information to the dictionary
                state_tying_dict[tokens[0]] = tokens[1]

        vocab_dict = {}

        # Process the state tying information
        for allo, index in state_tying_dict.items():
            phon = allo.split("{")[0]

            # Check if the phoneme is already in the dictionary
            if phon in vocab_dict:
                # Ensure that the indices are unique for each phoneme
                assert (
                    vocab_dict[phon] == index
                ), f"index conflict for {phon}: {index} vs. {vocab_dict[phon]} ({allo} {index})"
            else:
                vocab_dict[phon] = index

        with uopen(self.out_label_file, "w") as f:
            for v, idx in sorted(vocab_dict.items()):
                f.write(f"{v} {idx}\n")
