import soundfile
import glob
import os
import random

from sisyphus import Job, Task, tk

class FairseqAudioManifestCreationJob(Job):
    """
    Creates required manifest files for wav2vec pretraining with fairseq. For the original
    facebook script consider https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py
    """
    def __init__(
        self, audio_dir_path, file_extension="wav", valid_percent=0.01, seed=42, path_must_contain=None
    ):
        """
        :param tk.Path audio_dir_path: path to raw audio files to be included
        :param str file_extension: file extension to look for in audio_dir_path
        :param float valid_percent: percentage of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        :param str|None path_must_contain: if set, path must contain this substring
            for a file to be included in the manifest
        """
        self.audio_dir_path = audio_dir_path
        self.file_extension = file_extension
        self.valid_percent = valid_percent
        assert 0 <= self.valid_percent <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain

        self.out_manifest_path = self.output_path("manifest/", directory=True)
        self.rqmt = {"time": 8, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        dir_path = os.path.realpath(self.audio_dir_path.get_path())
        search_path = os.path.join(dir_path, "**/*." + self.file_extension)
        rand = random.Random(self.seed)

        valid_f = (
            open(os.path.join(self.out_manifest_path, "valid.tsv"), "w")
            if self.valid_percent > 0
            else None
        )

        with open(os.path.join(self.out_manifest_path, "train.tsv"), "w") as train_f:
            print(dir_path, file=train_f)

            if valid_f is not None:
                print(dir_path, file=valid_f)

            for fname in glob.iglob(search_path, recursive=True):
                file_path = os.path.realpath(fname)

                if self.path_must_contain and self.path_must_contain not in file_path:
                    continue

                frames = soundfile.info(fname).frames
                dest = train_f if rand.random() > self.valid_percent else valid_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), 
                        file=dest
                )
        if valid_f is not None:
            valid_f.close()
