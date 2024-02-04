import os
import sys
import argparse
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_ctc_alignments(
    # alignment_hdf_file,
    allophone_file,
    state_tying_file,
    silence_phone,
    min_segment_length=0,
    max_segment_length=30,
    num_segments=10,
):
    """
    Only guaranteed to work for monophones without left/right context
    For general allophones with context, change the code based on allophone_file and state_tying_file
    """
    frame_duration = 40  ## hardcoded for now, 4x-subsampling

    allophones = set()
    with open(allophone_file, "r") as alf:
        next(alf)
        for allophone in alf:
            allophone = allophone.strip()
            allophones.add(allophone)

    phone2idx = {}
    idx2phone = {}
    with open(state_tying_file, "r") as stf:
        for line in stf:
            allophone, idx = line.split()

            allophone_ = allophone.split(".")[0]
            assert allophone_ in allophones, f"Unknown allophone {allophone_}"

            # Note
            phone = allophone.split("{#+#}")[0]
            idx = int(idx)

            phone2idx[phone] = idx
            idx2phone[idx] = phone

    assert silence_phone in phone2idx, f"Unknown silence_phone {silence_phone}"
    silence_idx = phone2idx[silence_phone]

    num_plotted_segments = 0

    dataset.init_seq_order()
    for curr_seq_idx in range((dataset.get_total_num_seqs())):
        dataset.load_seqs(curr_seq_idx, curr_seq_idx + 1)
        curr_alignment = dataset.get_data(curr_seq_idx, "data")
        curr_seq_length = dataset.get_estimated_seq_length(curr_seq_idx)
        curr_seq_tag = dataset.get_tag(curr_seq_idx)

        segment_duration = curr_seq_length * frame_duration / 10  ## num frames
        if segment_duration / 100 < min_segment_length or segment_duration / 100 > max_segment_length:
            print(f"Skipping sequence {curr_seq_tag} with length {segment_duration / 100}...")
            continue

        # Plot is saved into the corresponding seq_tag folder
        os.makedirs(f"{curr_seq_tag.replace('/', '.')}", exist_ok=True)

        # Labels with repetitions
        repeated_labels = [idx2phone[curr_alignment[i]] for i in range(len(curr_alignment))]

        repeat_flag = True  ## whether repeating the same phone is allowed
        curr_pos = 1  ## track position in alignment
        labels = [silence_phone]  ## labels to plot
        curr_alignment_plot = []  ## positions to plot

        for i in range(len(curr_alignment)):
            if repeated_labels[i] == silence_phone:
                repeat_flag = True
                curr_alignment_plot += [silence_idx] * 4
            else:
                if repeat_flag:
                    repeat_flag = False
                    labels.append(repeated_labels[i])
                    curr_alignment_plot += [curr_pos] * 4
                    curr_pos += 1
                else:
                    if repeated_labels[i] == repeated_labels[i - 1]:
                        curr_pos -= 1
                    else:
                        labels.append(repeated_labels[i])
                    curr_alignment_plot += [curr_pos] * 4
                    curr_pos += 1

        # Plot
        time_values = np.arange(0, segment_duration, frame_duration / 40)

        fig, ax = plt.subplots(figsize=(18, 12))

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)

        # Linear range might overcompensate
        if len(time_values) != len(curr_alignment_plot):
            time_values = time_values[:-1]

        ax.scatter(
            time_values, curr_alignment_plot, c="blue", s=2, marker="s",
        )

        ax.set_xlabel("Num frames (10 ms)")
        ax.set_ylabel("Labels")
        ax.set_title(f"Sequence Tag: {curr_seq_tag}")
        fig.savefig(f"{curr_seq_tag.replace('/', '.')}/alignment.pdf")
        plt.close()

        ## Account for number of segments plotted so far
        num_plotted_segments += 1
        print(f"Accumulated {num_plotted_segments} for plotting...")
        if num_plotted_segments >= num_segments > 0:
            return


def init(hdf_file, seq_list_filter_file):
    rnn.init_better_exchook()
    rnn.init_thread_join_hack()
    dataset_dict = {
        "class": "HDFDataset",
        "files": [hdf_file],
        "use_cache_manager": True,
        "seq_list_filter_file": seq_list_filter_file,
    }

    rnn.init_config(config_filename=None, default_config={"cache_size": 0})
    global config
    config = rnn.config
    config.set("log", None)
    global dataset
    dataset = rnn.init_dataset(dataset_dict)
    rnn.init_log()
    print("Returnn segment-alignments starting up...", file=rnn.log.v2)
    rnn.returnn_greeting()
    rnn.init_faulthandler()
    rnn.init_config_json_network()


def main():
    arg_parser = argparse.ArgumentParser(description="Plot alignment visualisations.")
    arg_parser.add_argument(
        "alignment_hdf_file", help="alignment hdf file, contains 'inputs', 'seqLengths' and 'seqTags'",
    )
    arg_parser.add_argument("allophones", help="allophones file")
    arg_parser.add_argument("state_tying", help="state tying file")

    # Index of [SILENCE] (opt. <blank>) is extracted from the state-tying
    arg_parser.add_argument("silence_phone", help="silence phone symbol", default="[SILENCE]", type=str)
    arg_parser.add_argument("returnn_root", help="path to RETURNN root")

    # Impose restrictions onto the length of the segment, in seconds
    arg_parser.add_argument(
        "--min_segment_length", help="min length of segments in seconds", default=0, type=float,
    )
    arg_parser.add_argument(
        "--max_segment_length", help="max length of segments in seconds", default=30, type=float,
    )
    # Impose restriction onto the segments considered
    arg_parser.add_argument(
        "--num_segments", help="how many segments to plot, set -1 to plot all", default=10, type=int,
    )
    arg_parser.add_argument(
        "--segment_whitelist", help="whitelisted segments to plot", default=None, type=Optional[List[str]],
    )

    args = arg_parser.parse_args()
    sys.path.insert(0, args.returnn_root)

    kwargs = {
        "min_segment_length": args.min_segment_length,
        "max_segment_length": args.max_segment_length,
        "num_segments": args.num_segments,
    }

    global rnn
    import returnn.__main__ as rnn

    init(args.alignment_hdf_file, args.segment_whitelist)
    plot_ctc_alignments(
        # args.alignment_hdf_file,
        args.allophones,
        args.state_tying,
        args.silence_phone,
        **kwargs,
    )
    rnn.finalize()


if __name__ == "__main__":
    main()
