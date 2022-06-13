import math
import multiprocessing
import os.path
import pickle
import subprocess
from typing import *

import numpy
from sisyphus import Job, Task, tk

from i6_core.audio.ffmpeg import BlissFfmpegJob
from i6_core.lib.rasr_cache import FileArchive
from i6_core.lib import corpus

from .util import run_duration_recover


def ffmpeg_silence_remove(
        bliss_corpus : tk.Path,
        stop_threshold : int,
        window : float = 0.02,
        stop_duration : float = 0,
        stop_silence : float = 0,
        mode : str = "rms",
        ffmpeg_binary : Optional[tk.Path] = None,
        force_output_format=None,
        alias_path=""):
    """
    Applies the ffmpeg "silenceremove" audio filter.

    See https://ffmpeg.org/ffmpeg-filters.html#silenceremove

    start_periods defaults to -1, so that all silence is removed

    :param bliss_corpus:
    :param stop_threshold: in NEGATIVE dezibel, everything below this will be treated as silence
    :param stop_duration: keeps remaining silence of given value in seconds
    :param
    :return: bliss corpus
    :rtype: tk.Path
    """
    ffmpeg_binary = ffmpeg_binary if ffmpeg_binary else "ffmpeg"
    filter_string = 'silenceremove=detection=%s:stop_periods=-1' % mode

    assert stop_threshold <= 0, "positive values do not make sense"

    filter_string += ':stop_threshold=%ddB' % stop_threshold
    filter_string += ':window=%f' % window
    filter_string += ':stop_duration=%f' % stop_duration
    filter_string += ':stop_silence=%f' % stop_silence
    bliss_ffmpeg_job = BlissFfmpegJob(bliss_corpus,
                                      ffmpeg_options=["-af", filter_string],
                                      recover_duration=True,
                                      output_format=force_output_format,
                                      ffmpeg_binary=ffmpeg_binary)
    bliss_ffmpeg_job.add_alias(os.path.join(alias_path, "ffmpeg_silence_remove"))
    return bliss_ffmpeg_job.out_corpus


class AlignmentCacheSilenceRemoval(Job):
    """
    This Job uses an alignment cache to do silence removal on a given bliss corpus
    """
    def __init__(self, bliss_corpus, alignment_cache, allophone_file, window_shift, pause_duration=0.0,
                 silence_symbol=None, silence_symbol_duration=0.1, output_format=None, ffmpeg_binary=None):
        """

        :param bliss_corpus:
        :param alignment_cache:
        :param allophone_file:
        :param window_shift:
        :param pause_duration:
        :param silence_symbol:
        :param silence_symbol_duration:
        :param output_format:
        :param ffmpeg_binary:
        """
        self.bliss_corpus = bliss_corpus
        self.alignment_cache = alignment_cache
        self.allophone_file = allophone_file
        self.window_shift = window_shift
        self.ffmpeg_binary = ffmpeg_binary or "ffmpeg"
        self.output_format = output_format
        self.pause_duration = pause_duration
        self.silence_symbol = silence_symbol
        self.silence_symbol_duration = silence_symbol_duration

        self.out_audio_folder = self.output_path("audio/", directory=True)
        self.out_corpus = self.output_path("corpus.xml.gz")

        self.cut_rqmt = {'time': 8, 'mem': 8, 'cpu': 4}
        self.recover_rqmt = {'time': 4, 'mem': 4, 'cpu': 1}

    def tasks(self):
        yield Task("extract_silence", mini_task=True)
        yield Task("cut_audio", rqmt=self.cut_rqmt)
        yield Task("recover_duration", rqmt=self.recover_rqmt)

    def extract_alignment(self, alignment_path):
        if alignment_path.endswith(".bundle"):
            files = open(alignment_path, "rt")
            for cache in files:
                sprint_cache = FileArchive(cache.strip())
                sprint_cache.setAllophones(tk.uncached_path(self.allophone_file))
                keys = [str(s) for s in sprint_cache.ft if not str(s).endswith(".attribs")]
                for key in keys:
                    # only exctract time and mix, the HMM state is not needed
                    alignment = [[a[0], a[1], sprint_cache.allophones[a[1]]] for a in sprint_cache.read(key, 'align')]
                    yield (key, alignment)
        else:
            sprint_cache = FileArchive(alignment_path)
            sprint_cache.setAllophones(tk.uncached_path(self.allophone_file))
            keys = [str(s) for s in sprint_cache.ft if not str(s).endswith(".attribs")]
            for key in keys:
                # only exctract time and mix, the HMM state is not needed
                alignment = [[a[0], a[1], sprint_cache.allophones[a[1]]] for a in sprint_cache.read(key, 'align')]
                yield (key, alignment)


    def extract_silence(self):
        """
        TODO: fix the high memory consumption
        :return:
        """
        alignment_path = tk.uncached_path(self.alignment_cache)

        groups_dict = {}
        for key, cache in self.extract_alignment(alignment_path):
            length = len(cache)
            indices = numpy.asarray([numpy.minimum(1, entry[1]) for entry in cache])
            word_tokens = []
            for i in range(length):
                word_tokens.append(cache[i][2].split("}")[-1])

            words = 0
            silence_duration = 0
            in_word = False
            in_silence = False
            silence_word_positions = []

            groups = []

            in_group = bool(indices[0])
            group_start = 0
            group_end = 0

            for i, (speech, word_token) in enumerate(zip(indices, word_tokens)):
                # dealing with word tokens
                assert word_token in ['', '@i', '@f', '@i@f']
                if word_token == "@i" and in_word == False and in_silence == True:
                    in_word = True
                    in_silence = False
                    words += 1
                if word_token == "@i" and in_word == False:
                    in_word = True
                if word_token == "@i" and silence_duration > 0:
                    # clip the silence duration to the maximum number of frames we allow, e.g. 500ms pause / 10ms shift = 50 frames
                    # after word 12 there are 80 frames silence -> clip to 50 frames
                    silence_word_positions.append(
                        (words, numpy.minimum(silence_duration, self.pause_duration / self.window_shift)))
                    silence_duration = 0

                if in_word and word_token == "@f":
                    words += 1
                    in_word = False
                if word_token == "@i@f" and speech == 0:
                    silence_duration += 1
                if word_token == "@i@f" and speech == 1:
                    in_silence = True

                # dealing with speech/silence
                if not in_group and speech == 1:
                    if (group_start == 0 and group_end == 0):
                        group_start = i * self.window_shift
                        in_group = True
                    elif (i * self.window_shift - group_end) > self.pause_duration:
                        group_end = group_end + (self.pause_duration) / 2
                        groups.append((group_start, group_end))
                        group_start = i * self.window_shift - (self.pause_duration) / 2
                        in_group = True
                    else:
                        in_group = True
                if in_group and speech == 0:
                    group_end = i * self.window_shift
                    in_group = False

            if (group_start < group_end):
                groups.append((group_start, group_end))
            if (group_start > group_end):
                group_end = group_start + self.window_shift
                groups.append((group_start, group_end))

            # store groups and silence, drop the first silence word position
            # as we never have silence in the beginning
            groups_dict[key] = [groups, silence_word_positions[1:]]

        pickle.dump(groups_dict, open("groups.pkl", "wb"))

    def run_subprocess(self, command):
        subprocess.check_call(command)

    def cut_audio(self):

        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))

        groups_dict = pickle.load(open("groups.pkl", "rb"))

        empty_recordings = []

        ffmpeg_commands = []

        for recording in c.all_recordings():

            assert len(recording.segments) == 1
            segment = recording.segments[0]
            in_file = recording.audio

            target_file = "_".join(segment.fullname().split("/"))
            if self.output_format:
                target_file += "." + self.output_format
            else:
                target_file += os.path.splitext(in_file)[1]

            target_file = os.path.join(tk.uncached_path(self.out_audio_folder), target_file)

            groups = groups_dict[segment.fullname()]

            if len(groups) == 0:
                empty_recordings.append(recording)
                continue

            ffmpeg_command = ["ffmpeg", "-y", "-i", in_file, "-filter_complex"]

            split_orth = segment.orth.split(" _ ")
            filter_commands = []

            for i, new_group in enumerate(groups[0]):
                command = "[0]atrim=%.3f:%.3f[g%i]" % (new_group[0], new_group[1], i)
                filter_commands.append(command)
            split_orth = split_orth[0].split(" ")
            count = 0
            if (self.silence_symbol != None):
                for i, grp in enumerate(groups[1]):
                    word_id = grp[0] + count
                    duration = (int(grp[1]) / (self.silence_symbol_duration / self.window_shift))
                    if (duration - math.floor(duration) < 0.5):
                        duration = math.floor(duration)
                    else:
                        duration = math.ceil(duration)
                    if duration != 0:
                        split_orth.insert(word_id, self.silence_symbol * duration)
                        count = count + 1
                segment.orth = " ".join(split_orth)

            filter_command = ";".join(filter_commands)
            filter_command += ";" + "".join(["[g%i]" % i for i in range(len(groups[0]))]) + "concat=n=%i:v=0:a=1[out]" % (
                len(groups[0]))

            ffmpeg_command += [filter_command, "-map", "[out]", target_file]

            print(" ".join(ffmpeg_command))
            ffmpeg_commands.append(ffmpeg_command)

            recording.audio = target_file

        def delete_recordings(c, recordings):
            for subcorpus in c.subcorpora:
                delete_recordings(subcorpus, recordings)
            for r in recordings:
                print("tried to delete empty recording %s" % r.name)
                c.recordings.remove(r)

        delete_recordings(c, empty_recordings)

        c.dump("temp_corpus.xml.gz")

        p = multiprocessing.Pool(processes=4)
        p.map(self.run_subprocess, ffmpeg_commands)

    def recover_duration(self):
        run_duration_recover("temp_corpus.xml.gz", tk.uncached_path(self.out_corpus))