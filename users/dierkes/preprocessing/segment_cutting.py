import pathlib
import soundfile as sf
import numpy as np
import json
import multiprocessing
import argparse
import tqdm
import os
import shutil
import logging

from sisyphus import Job, Task, tk

class CutSpeechSegmentsFromAudio(Job):
    """
    This Job uses a transcription file to cut all segments containing speech from the audio files
    and stitches them together to files of a target length. Segments that are
    longer than the target_length are stored as individual files.


    """
    def __init__(self, audio_dir, segments_dir, target_length=20, n_workers=32, file_extension='.wav'):
        """

        :param audio_dir: path to all the audio files. Subfolders are not yet supported in audio
            directory
        :param segments_dir: path to the transcriptions. Subfolders are not yet supported in
            transcription directory
        :param target_length: target length of the stitched extracted speech
        :param n_workers:
        :param file_extension: currently supported are wav, flac and mp3
        """
        self.audio_dir = audio_dir
        self.segments_dir = segments_dir
        self.target_length = target_length
        self.n_workers = n_workers
        self.file_extension = file_extension
        assert file_extension in ['.wav', '.flac', '.mp3']

        self.out_audio_folder = self.output_path("audio/", directory=True)
        self.cut_rqmt = {'time': 8, 'mem': 8, 'cpu': 4}

    def tasks(self):
        yield Task("cut_speech_segments", rqmt=self.cut_rqmt, mini_task=True)

    def cut_file(self, task):
        path_file, root_out, target_len_sec, extension, segments = task
        file_name = path_file.stem
        meta_file_path = f"{segments}/{file_name}.json"

        with open(meta_file_path, 'r') as f:
            meta = json.loads(f.read())["value"]["segments"]

        to_cut = []
        for seg in meta:
            to_cut += [(seg["start"], seg["end"])]

        data, samplerate = sf.read(path_file)
        assert len(data.shape) == 1
        assert samplerate == 8000

        to_stitch = []
        length_accumulated = 0.0

        i = 0
        for start, end in to_cut:
            start_index = int(start * samplerate)
            end_index = int(end * samplerate)
            slice = data[start_index:end_index]

            # if a slice is longer than target_len_sec, we put it entirely in it's own piece
            if length_accumulated + (end - start) > target_len_sec and length_accumulated > 0:
                file_out = f"{root_out}/{file_name}_{i}{extension}"
                sf.write(file_out, np.hstack(to_stitch), samplerate=8000)
                to_stitch = []
                i += 1
                length_accumulated = 0

            to_stitch.append(slice)
            length_accumulated += end - start

        if to_stitch:
            file_out = f"{root_out}/{file_name}_{i}{extension}"
            sf.write(file_out, np.hstack(to_stitch), samplerate=8000)

    def cut(self, input_dir_audio, input_dir_segments, output_dir_audio,
            target_len_sec=20, n_process=32, out_extension='.wav'):

        list_dir = pathlib.Path(input_dir_audio).glob(f"**/*{out_extension}")
        list_dir = [x for x in list_dir]

        print(f"{len(list_dir)} files detected")
        print(f"Launching {n_process} processes")

        tasks = [(path_book, output_dir_audio, target_len_sec, out_extension, input_dir_segments) for path_book in list_dir]

        with multiprocessing.Pool(processes=n_process) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(self.cut_file, tasks), total=len(tasks)):
                pass

    def cut_speech_segments(self):
        self.cut(self.audio_dir,
            self.segments_dir,
            self.out_audio_folder,
            self.target_length,
            self.n_workers,
            self.file_extension)
