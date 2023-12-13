import argparse
import json
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# This works if blank_idx == 0
def calc_ctc_segment_stats(num_labels, blank_idx=0):
    dataset.init_seq_order()

    # HDF Dataset Utterance (=Sequence) Iteration
    # In the end, total_num_seq := (curr_seq_idx + 1)
    curr_seq_idx = 0

    # blank and non-blank frame cnt (across utterances)
    num_blank_frames, num_label_frames = 0, 0

    # start-blank, end-blank, mid-blank (and non-blank)
    # individual counts (per sequence) and number of such segments
    # start_blank_cnt, mid_blank_cnt, end_blank_cnt = 0, 0, 0
    # start_blank_seg_cnt, end_blank_seg_cnt = 0, 0

    # Maximal segment duration
    max_seg_len = 0

    # Counter for segment duration indexed by label idx
    label_seg_lens = Counter()
    # Counter for number of label idx occurrences
    label_seg_freq = Counter()
    # Counter for number label segment duration occurrences
    label_seg_len_freq = Counter()

    # # Counter for blank segments duration
    # blank_seg_len = Counter()
    # # Counter for non-blank segments duration
    # non_blank_seg_len = Counter()

    while dataset.is_less_than_num_seqs(curr_seq_idx):
        dataset.load_seqs(curr_seq_idx, curr_seq_idx + 1)
        data = dataset.get_data(curr_seq_idx, "data")

        # Filter [blank] only (no [SILENCE])
        blanks_idx = np.where(data == blank_idx)[0]
        labels_idx = np.where(data != blank_idx)[0]
        labels = data[labels_idx]

        # Counts for [blank] and labels (non-blanks)
        num_blank_frames += len(blanks_idx)
        num_label_frames += len(labels_idx)

        # If there are only blanks, skip the current sequence
        if len(labels_idx) == 0:
            curr_seq_idx += 1
            continue

        # Frame duration between non-blank indices
        curr_seq_seg_len = np.diff(labels_idx, prepend=-1)

        # Update current max segment duration per sequence
        curr_seq_max_seg_len = np.max(curr_seq_seg_len)
        if curr_seq_max_seg_len > max_seg_len:
            max_seg_len = curr_seq_max_seg_len

        # Update number of label idx occurrences
        label_seg_freq.update(labels)

        # Update number of label segment duration occurrences
        label_seg_len_freq.update(curr_seq_seg_len)

        # Update durations of label segments
        for label, seg_len in zip(labels, curr_seq_seg_len):
            label_seg_lens.update({label: seg_len})

        # # Compute initial, intermediate, final blank frame counts
        # first_non_blank, last_non_blank = labels_idx[0], labels_idx[-1]
        # if first_non_blank > 0:
        #     start_blank_cnt += labels_idx[0]
        #     start_blank_seg_cnt += 1
        # if last_non_blank < len(data) - 1:
        #     end_blank_cnt += len(data) - labels_idx[-1] - 1
        #     end_blank_seg_cnt += 1
        # mid_blank_cnt = np.count_nonzero(data == 0) - start_blank_cnt - end_blank_cnt

        curr_seq_idx += 1

    # total_seg_cnt = sum(label_seg_freq.values())  # == num_label_frames

    # label_seg_lens holds the corresponding segment lengths
    total_seg_len_1 = sum(label_seg_lens.values())
    total_seg_len_2 = sum([k * v for (k, v) in label_seg_len_freq.items()])
    assert total_seg_len_1 == total_seg_len_2

    total_seg_len = total_seg_len_1

    # Mean label length per segment
    mean_label_seg_len = total_seg_len / num_label_frames
    # Mean label length per sequence
    mean_label_seq_len = total_seg_len / curr_seq_idx  # =: num_seqs

    # Mean duration of segment for each label
    mean_label_seg_lens = {}

    for idx in range(num_labels):
        if label_seg_freq[idx] == 0:
            mean_label_seg_lens[idx] = 0
        else:
            mean_label_seg_lens[idx] = label_seg_lens[idx] / label_seg_freq[idx]

    # Mean sequence length
    mean_seq_len = (num_blank_frames + num_label_frames) / curr_seq_idx

    # Length statistics of label segments
    num_seg_lt2 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 2
        ]
    )
    num_seg_lt4 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 4
        ]
    )
    num_seg_lt8 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 8
        ]
    )
    num_seg_lt16 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 16
        ]
    )
    num_seg_lt32 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 32
        ]
    )
    num_seg_lt64 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 64
        ]
    )
    num_seg_lt128 = sum(
        [
            count
            for label_seg_len, count in label_seg_len_freq.items()
            if label_seg_len < 128
        ]
    )

    # Overview of computed statistics
    filename = "statistics.txt"
    with open(filename, "w+") as f:
        f.write("\tNon-silence: \n")
        f.write("\t\tMean length per segment: %f \n" % mean_label_seg_len)
        f.write("\t\tMean length per sequence: %f \n" % mean_label_seq_len)
        f.write("\t\tNum segments: %f \n" % num_label_frames)
        f.write("\t\tPercent segments shorter than x frames: \n")
        f.write("\t\tx = 2: %f \n" % (num_seg_lt2 / num_label_frames))
        f.write("\t\tx = 4: %f \n" % (num_seg_lt4 / num_label_frames))
        f.write("\t\tx = 8: %f \n" % (num_seg_lt8 / num_label_frames))
        f.write("\t\tx = 16: %f \n" % (num_seg_lt16 / num_label_frames))
        f.write("\t\tx = 32: %f \n" % (num_seg_lt32 / num_label_frames))
        f.write("\t\tx = 64: %f \n" % (num_seg_lt64 / num_label_frames))
        f.write("\t\tx = 128: %f \n" % (num_seg_lt128 / num_label_frames))

        f.write("\n")
        f.write("Overall maximum segment length: %d \n" % max_seg_len)
        f.write("\n\n")

        f.write("Sequence statistics: \n\n")
        f.write("\tMean length: %f \n" % mean_seq_len)
        f.write("\tNum sequences: %f \n" % curr_seq_idx)

    filename = "mean_label_seq_len.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_label_seq_len)))

    filename = "mean_label_seg_len.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_label_seg_len)))

    filename = "mean_label_seg_lens.json"
    with open(filename, "w+") as f:
        json.dump(mean_label_seg_lens, f)

    # Histogram for label segment lengths
    hist_data = [
        item
        for label_seg_len, count in label_seg_len_freq.items()
        for item in [label_seg_len] * count
    ]
    plt.hist(hist_data, bins=40, range=(0, 40))
    ax = plt.gca()
    quantiles = [np.quantile(hist_data, q) for q in [0.90, 0.95, 0.99]]
    for n, q in zip([90, 95, 99], quantiles):
        # Write quantiles to files
        with open("quantile_%s" % n, "w+") as f:
            f.write(str(q))
        ax.axvline(q, color="r")
    plt.savefig("labels_histogram.pdf")
    plt.close()


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
    print("Returnn segment-statistics starting up...", file=rnn.log.v2)
    rnn.returnn_greeting()
    rnn.init_faulthandler()
    rnn.init_config_json_network()


def main():
    arg_parser = argparse.ArgumentParser(description="Calculate alignment statistics.")
    arg_parser.add_argument(
        "hdf_file",
        help="hdf file which contains the extracted alignments of some corpus",
    )
    arg_parser.add_argument(
        "--seq-list-filter-file", help="whitelist of sequences to use", default=None
    )
    arg_parser.add_argument(
        "--blank-idx", help="the blank index in the alignment", default=0, type=int
    )
    arg_parser.add_argument(
        "--num-labels", help="the total number of labels in the alignment", type=int
    )
    arg_parser.add_argument("--returnn-root", help="path to RETURNN root")

    args = arg_parser.parse_args()
    sys.path.insert(0, args.returnn_root)

    global rnn
    import returnn.__main__ as rnn

    init(args.hdf_file, args.seq_list_filter_file)
    calc_ctc_segment_stats(args.num_labels, args.blank_idx)
    rnn.finalize()


if __name__ == "__main__":
    main()
