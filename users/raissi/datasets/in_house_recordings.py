__all__ = [
    "CreateUkrainianBlissCorpusFromApptekDataJob",
]

import glob
import json
import os
import re
import shutil
import string

from typing import List, Optional, Tuple, Union

from sisyphus import *
from sisyphus import gs, tk, Path

# -------------------- Recipes --------------------

import i6_core.lib.corpus as corpus
import i6_core.util as util


class CreateRawTrialogBlissCorpusJob(Job):
    def __init__(
        self,
        audio_dir: str,
        json_dir: str,
        corpus_language: Union[str, List[str], Tuple[str]],
        audio_format: str = "wav",
        corpus_name: str = "corpus",
    ):
        self.audio_dir = audio_dir
        self.json_dir = json_dir
        self.corpus_language = (corpus_language,) if isinstance(corpus_language, str) else corpus_language
        self.audio_format = audio_format
        self.corpus_name = corpus_name

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        corpus_file = "corpus.xml.gz"
        if os.path.isfile(corpus_file):
            os.remove(corpus_file)

        audio_pattern = os.path.join(self.audio_dir, "*wav")
        json_pattern = os.path.join(self.json_dir, "*.json")

        audios = glob.glob(audio_pattern)
        transcriptions = glob.glob(json_pattern)

        corpus_out = corpus.Corpus()

        for trans_path in sorted(transcriptions):
            name = os.path.basename(trans_path).split(".")[0]
            with util.uopen(trans_path, "rt") as tp:
                trans = json.load(tp)

            for tier in trans["contains"]:
                label = tier["label"]
                label = label.split("_")

                speaker_id = label[2]
                gender = label[3]
                lang = label[4]
                native = label[5]

                assert speaker_id.startswith(("interp", "patient", "doctor"))
                assert gender in ("male", "female"), (name, label, gender)
                assert lang in ("de", "ukr", "ru")

                if label[0] == "R20221026":
                    if "doctor" in speaker_id:
                        assert label[6] == "ch1"
                        channel_id = 1
                    else:
                        assert label[6] == "ch2"
                        channel_id = 2

                elif label[0] == "R20221103":
                    if "doctor" in speaker_id:
                        assert label[6] == "ch1"
                        channel_id = 1
                    else:
                        assert label[6] == "ch2"
                        channel_id = 2

                else:
                    if "interpret" in speaker_id:
                        assert label[6] == "ch1"
                        channel_id = 1
                    else:
                        assert "patient" in speaker_id or "doctor" in speaker_id
                        channel_id = 2

                recording_name = f"{name}_CH{channel_id}.wav"
                recording_path = ("/").join([audio_path, recording_name])
                assert os.path.exists(audio_path)

                # create recording
                recording = corpus.Recording()
                recording.name = recording_name
                recording.audio = recording_path

                speaker = corpus.Speaker()
                speaker.name = speaker_id
                speaker.attribs["gender"] = gender
                speaker.attribs["native-language"] = "yes" if native == "native" else "no"

                recording.speakers[speaker.name] = speaker
                corpus_out.speakers[speaker.name] = speaker

                for j, e_ in enumerate(tier["first"]["items"]):
                    (start, end) = [float(v) for v in e_["target"]["id"].split("#t=")[-1].split(",")]
                    assert end > start
                    trnscrpt = e_["body"]["value"]

                    if trnscrpt == "":
                        continue
                    segment_id = e_["id"].split("#")[1]
                    segment_id = int(re.sub(r"[A-Za-z]", "", segment_id))
                    segment = corpus.Segment()
                    segment.name = f"{recording_name}_{segment_id}"
                    segment.start = start
                    segment.end = end
                    segment.track = segment_id
                    segment.orth = trnscrpt
                    segment.speaker_name = speaker.name

                    recording.add_segment(segment)

                corpus_out.add_recording(recording)

        shutil.move(corpus_file, self.out_corpus.get_path())
