import numpy as np

from sisyphus import Job, Task, Path


class GetTokenizedTextStatisticsJob(Job):
    def __init__(
        self,
        text_file: Path,
    ):
        self.text_file = text_file

        self.out_statistics = self.output_path("statistics")
        self.out_seq_len_histogram_png = self.output_path("seq_len_histogram.png")
        self.out_seq_len_histogram_pdf = self.output_path("seq_len_histogram.pdf")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        import gzip
        import matplotlib.pyplot as plt

        opener = gzip.open if self.text_file.get_path().endswith(".gz") else open

        seq_lens = []
        with opener(self.text_file.get_path(), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                seq_lens.append(len(tokens))

        seq_lens = np.array(seq_lens)
        num_seqs = len(seq_lens)
        total_tokens = int(np.sum(seq_lens))

        with open(self.out_statistics.get_path(), "w+") as stat_file:
            stat_file.write("Statistics\n")
            stat_file.write(f"Number of sequences: {num_seqs}\n")
            stat_file.write(f"Total tokens: {total_tokens}\n")
            stat_file.write(f"Max tokens/seq: {int(np.max(seq_lens))}\n")
            stat_file.write(f"Min tokens/seq: {int(np.min(seq_lens))}\n")
            stat_file.write(f"Mean tokens/seq: {np.mean(seq_lens):.2f}\n")
            stat_file.write(f"Std tokens/seq: {np.std(seq_lens):.2f}\n")

        plt.hist(seq_lens, bins=max(1, num_seqs // 50))
        plt.xlabel("Tokens per sequence")
        plt.ylabel("Number of sequences")
        plt.title("Histogram of sequence lengths (tokens)")
        plt.grid()
        plt.savefig(self.out_seq_len_histogram_png.get_path())
        plt.savefig(self.out_seq_len_histogram_pdf.get_path())
