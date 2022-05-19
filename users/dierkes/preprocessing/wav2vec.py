import soundfile
import glob
import os
import random
import numpy as np

from sisyphus import Job, Task, tk

class FairseqAudioManifestCreationJob(Job):
    """
    Creates required manifest files for wav2vec pretraining with fairseq. For the
    script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py
    """

    __sis_hash_exclude__ = {
        "upsampling_alpha": None,
        "manifest_audio_paths": None,
    }

    def __init__(
        self,
        audio_dir_path,
        file_extension="wav",
        valid_percent=0.01,
        seed=42,
        path_must_contain=None,
        upsampling_alpha=None,
        manifest_audio_paths=None,
    ):
        """
        :param [tk.Path]|tk.Path audio_dir_path: list of paths or single path to raw audio files to be included
        :param str file_extension: file extension to look for in audio_dir_path
        :param float valid_percent: percentage of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        :param str|None path_must_contain: if set, path must contain this substring
            for a file to be included in the manifest
        :param float upsampling_alpha: specifies how much to upsample low resource audio directory paths. Upsampling
            calculation is done according to "Unsupervised Cross-Lingual Representation Learning for Speech
            Recognition", see  arXiv:2006.13979v2. According to this upsampling procedure it is not possible to upsample
            if only a single audio directory is given.
        :param [str]|[tk.Path]|str|tk.Path|None manifest_audio_paths: explicitly specifies output paths in manifest for each
            audio directory respectively. Allows to use different paths in the manifest than in the audio_dir_paths
        """
        if isinstance(audio_dir_path, tk.Path):
            self.audio_dir_paths = [audio_dir_path]
        else:
            assert isinstance(audio_dir_path, list)
            self.audio_dir_paths = audio_dir_path
        assert min([isinstance(path, tk.Path) for path in self.audio_dir_paths])

        if len(self.audio_dir_paths) == 1:
            assert (
                upsampling_alpha is None
            ), "Only one audio directory is given, upsampling not possible"

        if manifest_audio_paths:
            if not isinstance(manifest_audio_paths, list):
                manifest_audio_paths = [manifest_audio_paths]
            assert len(manifest_audio_paths) == len(self.audio_dir_paths)
            self.manifest_audio_paths = [
                os.path.realpath(path) for path in manifest_audio_paths
            ]
        else:
            self.manifest_audio_paths = None

        self.file_extension = file_extension
        self.valid_percent = valid_percent
        assert 0.0 <= self.valid_percent <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain
        self.upsampling_alpha = upsampling_alpha
        if self.upsampling_alpha is not None:
            assert 0.0 <= self.upsampling_alpha <= 1.0

        self.out_manifest_path = self.output_path("manifest/", directory=True)
        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        rand = random.Random(self.seed)

        dir_paths = [os.path.realpath(path.get_path()) for path in self.audio_dir_paths]

        if self.manifest_audio_paths:
            common_dir = os.path.commonpath(self.manifest_audio_paths)
        else:
            common_dir = os.path.commonpath(dir_paths)

        all_train_data = []

        valid_f = (
            open(os.path.join(self.out_manifest_path, "valid.tsv"), "w")
            if self.valid_percent > 0
            else None
        )
        if valid_f is not None:
            print(common_dir, file=valid_f)

        for i, dir_path in enumerate(dir_paths):
            train_data = []
            search_path = os.path.join(dir_path, "**/*." + self.file_extension)
            for path in glob.iglob(search_path, recursive=True):
                frames = soundfile.info(path).frames
                path = os.path.realpath(path)

                if self.path_must_contain and self.path_must_contain not in path:
                    continue

                if self.manifest_audio_paths:
                    rel_path = os.path.relpath(
                        self.manifest_audio_paths[i], common_dir
                    )
                    rel_path = os.path.join(rel_path, os.path.basename(path))
                else:
                    rel_path = os.path.relpath(path, common_dir)
                if rand.random() > self.valid_percent:
                    train_data.append((rel_path, frames))
                else:
                    print("{}\t{}".format(rel_path, frames), file=valid_f)
            all_train_data.append(train_data)
        if valid_f is not None:
            valid_f.close()

        if self.upsampling_alpha is not None and self.upsampling_alpha < 1:
            corpora_lengths = []
            for train_data in all_train_data:
                corpora_lengths.append(sum([frames for _, frames in train_data]))
            num_corpora = len(corpora_lengths)
            sum_corpora_length = sum(corpora_lengths)

            max_length_index = corpora_lengths.index(max(corpora_lengths))

            upsampled_proportions = [
                (l / sum_corpora_length) ** self.upsampling_alpha
                for l in corpora_lengths
            ]
            upsampled_probabilities = [
                p / sum(upsampled_proportions) for p in upsampled_proportions
            ]

            matrix_row_len = np.expand_dims(
                np.array(
                    corpora_lengths[:max_length_index]
                    + corpora_lengths[max_length_index + 1 :],
                    dtype=float,
                ),
                axis=0,
            )
            matrix_row_prob = np.expand_dims(
                np.array(
                    upsampled_probabilities[:max_length_index]
                    + upsampled_probabilities[max_length_index + 1 :],
                    dtype=float,
                ),
                axis=0,
            )
            if num_corpora > 2:
                w_mat = np.repeat(matrix_row_len, num_corpora - 1, axis=0)
                w_mat *= np.transpose(  # element-wise multiplication
                    np.repeat(matrix_row_prob, num_corpora - 1, axis=0)
                )
            else:
                w_mat = matrix_row_len * matrix_row_prob
            w_mat -= np.diag(matrix_row_len[0])
            b_vec = np.array(
                [
                    -upsampled_probabilities[i] * corpora_lengths[max_length_index]
                    for i in range(num_corpora)
                    if i != max_length_index
                ]
            )
            upsampling_factors = np.linalg.solve(w_mat, b_vec)
            # assert all entries larger than 1 because we only want to
            # upsample and never downsample
            assert min(
                upsampling_factors >= 1
            )

            upsampling_factors = np.insert(upsampling_factors, max_length_index, 1.0)
            upsampled_train_data = []

            for i, upsample in enumerate(upsampling_factors):
                upsampled_train_data.append([])
                while upsample >= 1:
                    upsampled_train_data[i].extend(all_train_data[i])
                    upsample -= 1
                if upsample > 0:
                    added_length = 0
                    random.shuffle(all_train_data[i])
                    j = 0
                    while corpora_lengths[i] * upsample > added_length:
                        upsampled_train_data[i].append(all_train_data[i][j])
                        added_length += all_train_data[i][j][1]
                        j += 1

            upsampled_train_data_lengths = []
            for train_data in upsampled_train_data:
                upsampled_train_data_lengths.append(
                    sum([frames for _, frames in train_data])
                )
            actual_upsampled_probabilities = np.array(upsampled_train_data_lengths) / sum(
                upsampled_train_data_lengths
            )
            # following test might fail if the input corpora are really small
            np.testing.assert_allclose(
                upsampled_probabilities, actual_upsampled_probabilities, rtol=0.1
            )
        else:
            upsampled_train_data = all_train_data

        with open(os.path.join(self.out_manifest_path, "train.tsv"), "w") as train_f:
            print(common_dir, file=train_f)
            for i, train_data in enumerate(upsampled_train_data):
                for path, frames in train_data:
                    print("{}\t{}".format(path, frames), file=train_f)
