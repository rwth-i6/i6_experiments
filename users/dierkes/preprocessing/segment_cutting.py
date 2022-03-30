import soundfile as sf
import numpy as np
import multiprocessing
import logging
import os
import matplotlib.pyplot as plt

from sisyphus import Job, Task, tk

from i6_core.lib import corpus


class CutAndStitchSpeechSegmentsFromCorpusJob(Job):
    """
    This Job uses a bliss corpus to cut all segments containing speech from the audio files
    and stitches them together to files of a target length. Segments that are
    longer than the target_length are stored as individual files.


    """

    def __init__(
        self,
        bliss_corpus_file,
        min_length=10,
        target_length=20,
        n_workers=4,
        file_extension="wav",
    ):
        """

        :param tk.Path bliss_corpus:
        :param int min_length: minimal length of the stitched extracted speech in seconds. Note that
            this constraint might be violated in some cases, e.g. when a recording contains less
            speech than min_length or because the greedy stitching strategy might produce leftover
            files for every recording shorter than min_length
        :param int target_length: target length of the stitched extracted speech in seconds
        :param int n_workers: number of threads for work on files in parallel
        :param str file_extension: currently supported are wav, flac and mp3
        """
        self.bliss_corpus_file = bliss_corpus_file
        self.min_length = min_length
        self.target_length = target_length
        self.n_workers = n_workers
        self.file_extension = file_extension
        assert file_extension in ["wav", "flac", "mp3"]

        self.out_audio_path = self.output_path("audio/", directory=True)
        self.out_length_hist = self.output_path("file_length_hist.png")
        self.cut_and_stitch_segments_rqmt = {
            "time": 8,
            "mem": 16,
            "cpu": self.n_workers,
        }
        self.length_dist_rqmt = {"time": 2, "mem": 1, "cpu": self.n_workers}

    def tasks(self):
        yield Task("cut_and_stitch_segments", rqmt=self.cut_and_stitch_segments_rqmt)
        yield Task("plot_length_distribution", rqmt=self.length_dist_rqmt)

    def cut_and_stitch_segments(self):
        self.corpus_object = corpus.Corpus()
        self.corpus_object.load(self.bliss_corpus_file.get_path())
        recordings = list(self.corpus_object.all_recordings())

        print(f"{len(recordings)} recordings detected")
        print(f"launching {self.n_workers} processes")

        tasks = [(r, self.out_audio_path.get_path(), self.file_extension) for r in recordings]
        with multiprocessing.Pool(processes=self.n_workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(self.cut_file, tasks)):
                if i % 100 == 0:
                    logging.info(f"{i} of {len(tasks)} files done")

    def plot_length_distribution(self):
        files = [
            f"{self.out_audio_path.get_path()}/{f}"
            for f in os.listdir(self.out_audio_path.get_path())
            if os.path.splitext(f)[1] == f".{self.file_extension}"
        ]
        num_files = len(files)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            file_lengths = list(pool.imap_unordered(self.get_length, files, 100))

        avg_length = round(sum(file_lengths) / num_files, 2)
        length = round(sum(file_lengths) / 3600, 2)
        fig, ax = plt.subplots()
        ax.hist(file_lengths, bins=80)
        fig.text(
            0.5,
            0.9,
            f"Average length: {avg_length}s, Summed length: {length}h",
            ha="center",
            va="center",
        )
        ax.set_ylabel("count")
        ax.set_xlabel("file length")
        plt.savefig(self.out_length_hist.get_path())

    def get_length(self, file):
        f = sf.SoundFile(file)
        return f.frames / f.samplerate

    def cut_file(self, task):
        recording, root_out, extension = task
        recording_path = recording.audio
        recording_name = recording.name

        to_cut = []
        for seg in recording.segments:
            to_cut += [(seg.start, seg.end)]

        data, samplerate = sf.read(recording_path)
        assert len(data.shape) == 1

        to_stitch = []
        length_accumulated = 0.0

        i = 0
        for start, end in to_cut:
            start_index = int(start * samplerate)
            end_index = int(end * samplerate)
            slice = data[start_index:end_index]

            if (end - start) > self.target_length:
                file_out = f"{root_out}/{recording_name}_{i}.{extension}"
                sf.write(file_out, slice, samplerate=samplerate)
                i += 1
            elif (
                length_accumulated + (end - start) > self.target_length
                and max(length_accumulated, end - start) >= self.min_length
            ):
                if (end - start) > length_accumulated:
                    file_out = f"{root_out}/{recording_name}_{i}.{extension}"
                    sf.write(file_out, slice, samplerate=samplerate)
                    i += 1
                else:
                    file_out = f"{root_out}/{recording_name}_{i}.{extension}"
                    sf.write(file_out, np.hstack(to_stitch), samplerate=samplerate)
                    to_stitch = [slice]
                    length_accumulated = end - start
                    i += 1
            elif (
                length_accumulated + (end - start) > self.target_length
                and max(length_accumulated, end - start) < self.min_length
            ):
                file_out = f"{root_out}/{recording_name}_{i}.{extension}"
                sf.write(
                    file_out, np.hstack(to_stitch + [slice]), samplerate=samplerate
                )
                to_stitch = []
                length_accumulated = 0
                i += 1
            else:
                to_stitch.append(slice)
                length_accumulated += end - start

        if to_stitch:
            file_out = f"{root_out}/{recording_name}_{i}.{extension}"
            sf.write(file_out, np.hstack(to_stitch), samplerate=samplerate)

    @classmethod
    def hash(cls, kwargs):
        d = {
            "bliss_corpus_file": kwargs["bliss_corpus_file"],
            "min_length": kwargs["min_length"],
            "target_length": kwargs["target_length"],
            "file_extension": kwargs["file_extension"],
        }
        return super().hash(d)
