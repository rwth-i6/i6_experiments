import sys
import json
import gzip
import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# kwargs['xlabel']: Optional[str]
# kwargs['ylabel']: Optional[str]
# kwargs['percentiles']: Optinal[List[int]]


def num_bin_heuristic(num_values):
    """ Try to partition the data into 20-30 bins. """
    num_values_rounded = np.round(num_values / 5) * 5

    min_remainder, num_bins = 30, 0
    for n in np.arange(20, 31):
        if min_remainder > num_values_rounded % n:
            min_remainder = num_values_rounded % n
            num_bins = n

    return num_bins


def make_histogram(counter, alias, **kwargs):
    # Data
    data = [*counter.elements()]

    max_num_bins = num_bin_heuristic(len(counter))
    num_bins = min(len(counter) + 1, max_num_bins)
    custom_binning = len(counter) < max_num_bins

    # TODO: Add more kwargs for histogram settings, etc.

    # Differentiate for smaller and larger number of bins
    bins = max_num_bins
    xticks = None
    if custom_binning:
        bins = np.arange(1, num_bins + 1) - 0.5
        xticks = np.arange(1, num_bins)

    # General
    hist, _, _ = plt.hist(
        data, bins=bins, alpha=0.75, range=(1, len(counter) + 2), ec="black",
    )
    plt.xticks(xticks, fontsize=8)
    plt.xlabel(kwargs.get("xlabel", ""))
    plt.ylabel(kwargs.get("ylabel", ""))

    # Percentiles
    percentiles = kwargs.get("percentiles", None)
    if percentiles is not None:
        for percentile in percentiles:
            perc = np.percentile(data, percentile)
            plt.axvline(x=perc, color="red")
            plt.text(
                perc,
                max(hist) * 1.05,
                str(percentile),
                color="red",
                ha="center",
                va="bottom",
            )

    # Save and close
    plt.savefig(f"{alias}.pdf")
    plt.close()


# Too specific, make more general
def make_plot(counter, alias, **kwargs):
    data = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    total_cnt = 0
    indices, counts, cum_counts = [], [], []

    for idx, (_, cnt) in enumerate(data, start=1):
        indices.append(idx)
        counts.append(cnt)

        total_cnt += cnt
        cum_counts.append(total_cnt)

    # plt.figure(figsize=(18, 12))  # fixed
    plot = plt.plot(
        indices,
        counts,
        alpha=0.75,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        linestyle="-",
        color="royalblue",
    )

    # Adjust for percentiles' indices
    ax = plt.gca()
    yticks_diff = ax.get_yticks()[1] - ax.get_yticks()[0]
    plt.ylim(-0.5 * yticks_diff)
    bottom_loc = -0.4 * yticks_diff

    # Percentiles
    cumulative_fraction = (np.array(cum_counts) / total_cnt) * 100
    percentiles = kwargs.get("percentiles", None)
    if percentiles is not None:
        for percentile in percentiles:
            percentile_idx = np.argmax(cumulative_fraction >= percentile)
            plt.axvline(x=percentile_idx + 1, color="r")
            plt.text(
                percentile_idx + 1,
                max(counts) * 1.05,
                str(percentile),
                color="r",
                ha="center",
                va="bottom",
            )
            plt.text(
                percentile_idx + 5 * len(str(percentile_idx)),  # 15-20
                bottom_loc,
                str(percentile_idx),
                color="r",
                ha="left",
                fontsize=8,
            )

    plt.xlabel(kwargs.get("xlabel", ""))
    plt.ylabel(kwargs.get("ylabel", ""))
    plt.savefig(f"{alias}.pdf")
    plt.close()


def calc_bpe_statistics(bliss_lexicon, transcription, **kwargs):
    try:
        bliss_lexicon = gzip.open(bliss_lexicon, "r")
    except Exception:
        pass

    tree = ET.parse(bliss_lexicon)
    root = tree.getroot()

    # If present, omit [blank], [SILENCE], [UNKNOWN] etc.
    bpe_tokens = tree.findall(
        './/phoneme-inventory/phoneme[variation="context"]/symbol'
    )
    bpe_vocab = [bpe_token.text for bpe_token in bpe_tokens]

    # Average number of symbols that comprise a BPE token
    # num_bpe_tokens = len(bpe_vocab), avoid redundant computations
    total_num_symbols = 0
    symbols_per_token = Counter()
    num_symbols_per_token = Counter()

    for bpe_token in bpe_vocab:
        curr_num_symbols = len(bpe_token.replace("@@", ""))
        symbols_per_token.update({bpe_token: curr_num_symbols})
        num_symbols_per_token.update([curr_num_symbols])
        total_num_symbols += curr_num_symbols

    # Mean (token-level statistics)
    mean_num_symbols_per_token = total_num_symbols / len(bpe_vocab)
    filename = "mean_num_symbols_per_token.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_num_symbols_per_token)))

    # # Histogram (token-level statistics)
    # data = [*num_symbols_per_token.elements()]
    # num_bins = len(num_symbols_per_token.keys()) + 1  # capture full range of values
    #
    # # General
    # plt.hist(data, bins=np.arange(1, num_bins+1) - 0.5, alpha=0.75, range=(1, num_bins+1), ec='black')
    # plt.xticks(np.arange(1,num_bins), fontsize=8)
    # plt.xlabel("Number of symbols per BPE token")
    # plt.ylabel("Number of occurrences")
    #
    # # 95-th quantile w/ disambiguation
    # percentile = np.percentile(data, 95)
    # plt.axvline(x=percentile, color='red')
    # plt.text(percentile, max(hist) + 40, '95', color='red', ha='center', va='bottom')
    #
    # plt.savefig("num_symbols_per_token_histogram.pdf")
    # plt.close()

    # Histogram (token-level statistics)
    kwargs.update(
        {
            "xlabel": "Number of symbols per BPE token",
            "ylabel": "Number of occurrences",
            "percentiles": [95],
        }
    )
    make_histogram(num_symbols_per_token, "num_symbols_per_token_histogram", **kwargs)

    # --------------------------------- #

    # Word-level statistics - iterate over tokenization-dict
    tokenization = {}
    # Take care to omit special lemmata
    for lemma in root.findall(".//lemma"):
        if not lemma.attrib:
            orth = lemma.find(".//orth")
            phon = lemma.find(".//phon")

            # Only works for unique <phon> sequences
            if orth is not None and phon is not None:
                tokenization[orth.text] = phon.text.split()

    total_num_tokens = 0  # num_words = len(tokenization), avoid redundant computations
    token_count_per_vocab = Counter()
    num_tokens_per_word = Counter()

    for word, subwords in tokenization.items():
        curr_num_tokens = len(subwords)
        num_tokens_per_word.update([curr_num_tokens])
        total_num_tokens += curr_num_tokens
        for subword in subwords:
            token_count_per_vocab.update([subword])

    # Means (word-level/vocab-level statistics)
    mean_num_token_per_word = total_num_tokens / len(tokenization)  # number of words
    filename = "mean_num_token_per_word.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_num_token_per_word)))

    mean_token_count_per_vocab = total_num_tokens / len(
        token_count_per_vocab
    )  # number of BPE subwords
    filename = "mean_token_count_per_vocab.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_token_count_per_vocab)))

    # Dump BPE token frequency (vocabulary)
    filename = "token_count_per_vocab.json"
    with open(filename, "w+") as f:
        json.dump(token_count_per_vocab, f, indent=0)

    # Visualisations (word-level statistics)
    kwargs.update(
        {
            "xlabel": "Number of BPE tokens per word",
            # 'ylabel': "Number of occurrences",
            "percentiles": [95],
        }
    )
    make_histogram(num_tokens_per_word, "num_token_per_word_histogram", **kwargs)

    kwargs.update(
        {
            "xlabel": "BPE tokens' counts (vocabulary)",
            # 'ylabel': "Number of occurrences",
            "percentiles": [90, 95, 99],
        }
    )
    make_plot(token_count_per_vocab, "token_count_per_vocab_plot", **kwargs)

    # --------------------------------- #

    # Sequence-level statistics - iterate over transcription (list of text sequences)
    total_num_tokens = 0
    total_num_sequences = 0
    oov_words = Counter()
    token_count_per_corpus = Counter()
    num_tokens_per_sequence = Counter()

    with open(transcription, "rt") as f:
        for sequence in f:
            curr_num_tokens = 0

            for word in sequence.split():
                subwords = tokenization.get(word, [])
                if subwords:
                    curr_num_tokens += len(subwords)

                    for subword in subwords:
                        token_count_per_corpus.update([subword])
                else:
                    oov_words.update([word])

            num_tokens_per_sequence.update(
                [curr_num_tokens]
            )  # num_tokens per given sequence
            total_num_tokens += curr_num_tokens
            total_num_sequences += 1

    # Means (transcription-level statistics)
    mean_num_token_per_sequence = total_num_tokens / total_num_sequences
    filename = "mean_num_token_per_sequence.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_num_token_per_sequence)))

    mean_token_count_per_corpus = total_num_tokens / len(
        token_count_per_corpus
    )  # number of BPE subwords
    filename = "mean_token_count_per_corpus.txt"
    with open(filename, "w+") as f:
        f.write(str(float(mean_token_count_per_corpus)))

    # Dump BPE token frequency (corpus)
    filename = "token_count_per_corpus.json"
    with open(filename, "w+") as f:
        json.dump(token_count_per_corpus, f, indent=0)

    # Dump OOV words (words without BPE-tokenization in lexicon)
    filename = "oov_words.json"
    with open(filename, "w+") as f:
        json.dump(oov_words, f, indent=0)

    # Visualisations (transcription-level statistics)
    kwargs.update(
        {
            "xlabel": "Number of BPE tokens per sequence",
            # 'ylabel': "Number of occurrences",
            "percentiles": [95],
        }
    )
    make_histogram(
        num_tokens_per_sequence, "num_token_per_sequence_histogram", **kwargs
    )

    kwargs.update(
        {
            "xlabel": "BPE tokens' counts (corpus)",
            # 'ylabel': "Number of occurrences",
            "percentiles": [90, 95, 99],
        }
    )
    make_plot(token_count_per_corpus, "token_count_per_corpus_plot", **kwargs)

    # --------------------------------- #

    filename = "bpe_statistics.txt"
    with open(filename, "w+") as f:
        f.write("BPE STATISTICS:\n")
        f.write(
            f"\t Mean number of symbols per BPE token: {mean_num_symbols_per_token}.\n\n"
        )
        f.write(f"\t Mean number of BPE tokens per word: {mean_num_token_per_word}.\n")
        f.write(
            f"\t Mean number of BPE tokens per sequence: {mean_num_token_per_sequence}.\n\n"
        )
        f.write(
            f"\t Mean count of BPE tokens in vocabulary: {mean_token_count_per_vocab}.\n"
        )
        f.write(
            f"\t Mean count of BPE tokens in corpus: {mean_token_count_per_corpus}.\n"
        )


def main():
    arg_parser = argparse.ArgumentParser(
        description="Calculate BPE subword statistics."
    )
    arg_parser.add_argument(
        "bliss_lexicon", help="Bliss lexicon with word-to-tokenization correspondence."
    )
    arg_parser.add_argument(
        "transcription", help="Corpus text corresponding to a Bliss corpus."
    )
    args = arg_parser.parse_args()

    # TODO
    hist_kwargs = {"xlabel": None, "ylabel": None, "percentile": [95]}
    calc_bpe_statistics(args.bliss_lexicon, args.transcription, **hist_kwargs)


if __name__ == "__main__":
    main()
