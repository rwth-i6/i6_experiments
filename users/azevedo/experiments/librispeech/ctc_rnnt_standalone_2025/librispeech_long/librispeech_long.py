import os
import shutil
import subprocess

from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen
from pathlib import Path


class DownloadLibriSpeechLongCorpusJob(Job):
    """
    Download complete LibriSpeech-Long (test) corpus from
    https://github.com/google-deepmind/librispeech-long
    and checks for file integrity via md5sum

    To get the corpus metadata, use
    DownloadLibriSpeechMetadataJob

    self.out_corpus_folders[corpus_key] links to the root of <corpus_key>/speaker_id/chapter/*
    folder structure, where corpus_key in [dev | test]-[other | clean]
    """

    def __init__(self):
        """
        :param str corpus_key: corpus identifier, e.g. "dev-other"
        """
        self.corpus_keys = ["dev-clean", "dev-other", "test-clean", "test-other"]

        # self.out_corpus_folder = self.output_path("%s" % self.corpus_key)
        self.out_corpus_folders = {
            corpus_key: self.output_path("%s" % corpus_key) for corpus_key in self.corpus_keys
        }

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        subprocess.check_call(["wget", "https://storage.googleapis.com/librispeech_long/v0_1.tar.gz"])

        # verify download with md5sum
        # TODO: change md5sum.txt path (ideally use wget from some trusted server)
        shutil.copy("/work/asr3/hilmes/azevedo/librispeech-long/md5sum.txt", ".")
        subprocess.check_call(["md5sum", "--status", "-c", "md5sum.txt"])

        subprocess.check_call(
            [
                "tar",
                "-xf",
                "v0_1.tar.gz",
                "-C",
                ".",
            ]
        )
        self._move_files()
        os.unlink("v0_1.tar.gz")
        shutil.rmtree("librispeech-long")

    def _move_files(self):
        for corpus_key, out_folder in self.out_corpus_folders.items():
            shutil.move("librispeech-long/%s" % corpus_key, out_folder.get_path())


class LibriSpeechLongCreateBlissCorpusJob(Job):
    """
    Creates a Bliss corpus from a LibriSpeech-Long corpus folder using the speaker information in addition

    Outputs a single bliss .xml.gz file
    """

    def __init__(self, corpus_folder, speaker_metadata):
        """
        :param Path corpus_folder: Path to a LibriSpeech-Long corpus folder
        :param Path speaker_metadata: Path to SPEAKER.TXT file from the MetdataJob (out_speakers)
        """
        self.corpus_folder = corpus_folder
        self.speaker_metadata = speaker_metadata

        self.out_corpus = self.output_path("corpus.xml.gz")

        self._speakers = {}  # dict(key: id, value: [sex, subset, min, name]
        self._transcripts = []  # [dict(name, chapter, segment, orth, path)]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self._get_speakers()
        self._get_transcripts()

        c = corpus.Corpus()
        c.name = os.path.basename(self.corpus_folder.get_path())

        used_speaker_ids = set()  # store which speakers are used

        for transcript in self._transcripts:
            name = "{0}-{1}-{2:04d}".format(transcript["speaker_id"], transcript["chapter"], transcript["segment"])
            recording = corpus.Recording()
            recording.name = name
            recording.speaker_name = transcript["speaker_id"]
            recording.audio = "{0}/{1}_{2:04d}.flac".format(transcript["path"], transcript["chapter"], transcript["segment"])

            used_speaker_ids.add(transcript["speaker_id"])

            segment = corpus.Segment()
            segment.name = name
            segment.start = 0
            segment.end = float("inf")
            segment.orth = transcript["orth"].strip()

            recording.segments.append(segment)
            c.recordings.append(recording)

        for speaker_id, speaker_info in sorted(self._speakers.items()):
            if speaker_id not in used_speaker_ids:
                continue
            speaker = corpus.Speaker()
            speaker.name = speaker_id
            speaker.attribs["gender"] = "male" if speaker_info[0] == "M" else "female"
            c.add_speaker(speaker)

        c.dump(self.out_corpus.get_path())

    def _get_speakers(self):
        """
        Extract the speakers from the SPEAKERS.TXT file
        """
        with uopen(self.speaker_metadata, "r") as speakersfile:
            for line in speakersfile:
                if line[0] == ";":
                    continue
                procline = list(map(str.strip, line.split("|")))
                self._speakers[int(procline[0])] = [
                    procline[1],
                    procline[2],
                    float(procline[3]),
                    procline[4],
                ]

    def _get_transcripts(self):
        """
        Traverse the folder structure and search for the *.txt files and read the content
        """
        for dirpath, dirs, files in sorted(os.walk(self.corpus_folder.get_path(), followlinks=True)):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                with uopen(os.path.join(dirpath, file), "r") as transcription:
                    line = transcription.readlines()
                    # using `wc **/*.txt` we verified that always one line (no \n) in txt files
                    assert len(line) == 1

                    # dirpath/file = librispeech-long/<corpus_name>/<speaker_id>/<chapter>/<chapter>_<segment>.txt
                    stem = Path(file).stem.split("_")
                    transcript = {
                        "speaker_id": int(Path(dirpath).parent.stem),
                        "chapter": int(stem[0]),
                        "segment": int(stem[1]),
                        "orth": line[0],
                        "path": dirpath,
                    }
                    self._transcripts.append(transcript)
