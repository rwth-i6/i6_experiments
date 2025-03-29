import json
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from sisyphus import *
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

Path = setup_path(__package__)



Alignment = List[Dict[str, Union[str, int]]]



class BPEToWordAlignmentsJob(Job):
    """
    """

    def __init__(
        self,
        alignment_path: tk.Path,
        labels_path: tk.Path,
    ) -> None:
        """
        :param alignment_path: path to BPE alignment
        :param labels_path: path to dict containing BPE-token to index mappings
        """

        self.alignment_path = alignment_path
        self.labels_path = labels_path

        self.word_alignments = self.output_path("word_alignments.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.alignment_path) as f:
            token_als: Dict[str, Alignment] = json.load(f)

        with open(self.labels_path) as f:
            # skipping "{", "'<s>': 0," and "}"
            to_labels = [line.split(":")[0][1:-1] for line in f.readlines()[2:-1]]
        print(to_labels)

        word_als = {utt_id: self.to_word_level(token_als[utt_id], to_labels) for utt_id in token_als}

        with open(self.word_alignments, "w") as f:
            json.dump(word_als, f, indent=4)   

    @staticmethod
    def to_word_level(al: Alignment, labels: Dict[int, str]) -> Alignment:
        """
        BPE token-level alignments to word-level alignments.
        """

        word_al = []
        curr_word = ""
        curr_start = -1
        curr_end = -1

        for frame in al:
            token = labels[frame["token"]]

            # start or continue a multi-token word
            if token.endswith("@@"):
                # first token of a word
                if not curr_word:
                    curr_start = frame["start"]
                curr_word += token[:-2]  # remove "@@"
                curr_end = frame["end"]
            # end of word
            else:
                # end of single-token word
                if curr_word == "":
                    curr_start = frame["start"]

                curr_word += token
                
                word_al.append({
                    "word": curr_word,
                    "start": curr_start,
                    "end": frame["end"]
                })

                curr_word = ""
                curr_start = -1
                curr_end = -1

        # if for some reason bpe sequence ends with "@@"
        if curr_word:
            word_al.append({
                "word": curr_word,
                "start": curr_start,
                "end": curr_end
            })

        return word_al



class LatencyJob(Job):
    """
    Calculates the latency of a hyp alignment w.r.t. a reference alignment.
    """
    
    def __init__(
        self,
        ref_path: tk.Path,
        hyp_path: tk.Path,
    ) -> None:
        """
        :param ref_path: path to BPE alignment reference (e.g. CTC forced alignment)
        :param hyp_path: path to alignment hypothesis (e.g. obtained from beam search)
        """

        self.ref_path = ref_path
        self.hyp_path = hyp_path

        self.out_path = self.output_path("latency_statistics.csv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        records = []

        with open(self.ref_path) as f_ref, open(self.hyp_path) as f_hyp:
            refs: Dict[str, Alignment] = json.load(f_ref)
            hyps: Dict[str, Alignment] = json.load(f_hyp)

        assert hyps.keys() <= refs.keys(), "All utterance IDs in hyp must be in ref."

        for utt_id in hyps:
            ref_al, hyp_al = refs[utt_id], hyps[utt_id]

            # check if ref sequence equals hyp sequence
            if " ".join(t["word"] for t in ref_al) != " ".join(t["word"] for t in hyp_al):
                print(f"[!] SEQUENCE {utt_id} DOESN'T MATCH!")
                continue
            
            # get all alignment time deviations
            for pos, (r, h) in enumerate(zip(ref_al, hyp_al)):
                records.append({
                    "utt_id": utt_id, 
                    "position": pos,
                    "start_diff": h["start"] - r["start"],
                    "end_diff": h["end"] - r["end"],
                })
            print(f"> ADDED {len(ref_al)} diffs.")
        
        #
        # save latency data to csv
        #
        dtype_mapping = {
            'utt_id': 'category',       # IDs are repeated
            'position': 'uint16',       # positions < 65k
            'start_diff': 'int16',      # prolly no diff > 65k
            'end_diff': 'int16'         # prolly no diff > 65k
        }
        df = pd.DataFrame(records).astype(dtype_mapping)
        df.to_csv(self.out_path, index=False)