import soundfile as sf
import numpy as np
from i6_core.lib import corpus
import multiprocessing
import tqdm

from sisyphus import Job, Task, tk

class CutAndStitchSpeechSegmentsFromCorpusJob(Job):
    """
    This Job uses a bliss corpus to cut all segments containing speech from the audio files
    and stitches them together to files of a target length. Segments that are
    longer than the target_length are stored as individual files.


    """
    def __init__(self, bliss_corpus_file, target_length=20, n_workers=16, file_extension='wav'):
        """

        :param tk.Path bliss_corpus:
        :param int target_length: target length of the stitched extracted speech in seconds
        :param int n_workers: number of threads for work on files in parallel
        :param str file_extension: currently supported are wav, flac and mp3
        """
        self.bliss_corpus_file = bliss_corpus_file
        self.target_length = target_length
        self.n_workers = n_workers
        self.file_extension = file_extension
        assert file_extension in ['wav', 'flac', 'mp3']

        self.out_audio_path = self.output_path("audio/", directory=True)
        self.rqmt = {'time': 8, 'mem': 16, 'cpu': self.n_workers}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def cut_file(self, task):
        recording, root_out, target_len_sec, extension = task
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

            # if a slice is longer than target_len_sec, we put it entirely in it's own piece
            if length_accumulated + (end - start) > target_len_sec and length_accumulated > 0:
                file_out = f"{root_out}/{recording_name}_{i}.{extension}"
                sf.write(file_out, np.hstack(to_stitch), samplerate=samplerate)
                to_stitch = []
                i += 1
                length_accumulated = 0

            to_stitch.append(slice)
            length_accumulated += end - start

        if to_stitch:
            file_out = f"{root_out}/{recording_name}_{i}.{extension}"
            sf.write(file_out, np.hstack(to_stitch), samplerate=samplerate)

    def run(self):
        self.corpus_object = corpus.Corpus()
        self.corpus_object.load(self.bliss_corpus_file.get_path())
        recordings = list(self.corpus_object.all_recordings())

        print(f"{len(recordings)} recordings detected")
        print(f"launching {self.n_workers} processes")

        tasks = [(r, self.out_audio_path, self.target_length, self.file_extension) for r in recordings]

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(self.cut_file, tasks), total=len(tasks)):
                pass
