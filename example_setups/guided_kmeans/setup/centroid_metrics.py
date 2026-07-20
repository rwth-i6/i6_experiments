__all__ = ["CentroidCosineSimilarityJob", "PhonemeL1DistanceJob", "AverageTotalScoreJob"]

from sisyphus import tk, Job, Task


class CentroidCosineSimilarityJob(Job):
    # Compute mean pairwise cosine similarity of cluster centroids (excluding diagonal)

    def __init__(self, centroids_path: tk.Path):
        self.centroids_path = centroids_path
        self.out_mean_cos_sim = self.output_var("mean_cos_sim")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np

        c = np.load(str(self.centroids_path))
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        valid = (norms > 0).flatten()
        cn = np.where(norms > 0, c / np.where(norms > 0, norms, 1.0), 0.0)
        cos = cn @ cn.T
        cos[~valid, :] = np.nan
        cos[:, ~valid] = np.nan
        np.fill_diagonal(cos, np.nan)
        self.out_mean_cos_sim.set(float(np.nanmean(cos)))


class PhonemeL1DistanceJob(Job):
    # Compute L1 distance of recognition phoneme distribution to unigram priors

    def __init__(self, statistics_path: tk.Path, epoch: int, priors_path: tk.Path | str):
        self.statistics_path = statistics_path
        self.epoch = epoch
        self.priors_path = priors_path
        self.out_l1_dist = self.output_var("l1_dist")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        counts: dict[str, int] = {}
        with open(str(self.priors_path)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count_str, phoneme = line.split("\t")
                counts[phoneme] = int(count_str)
        total_prior = sum(counts.values())
        priors = {p: c / total_prior for p, c in counts.items()}

        with open(str(self.statistics_path)) as f:
            stats = json.load(f)

        epoch_key = str(self.epoch)
        if epoch_key not in stats:
            self.out_l1_dist.set(float("nan"))
            return

        epoch_counts: dict[str, int] = stats[epoch_key]["absolute_phoneme_counts"]
        total_epoch = sum(epoch_counts.get(p, 0) for p in priors)
        if total_epoch == 0:
            self.out_l1_dist.set(float("nan"))
            return

        l1 = sum(
            abs(epoch_counts.get(p, 0) / total_epoch - priors[p])
            for p in priors
        )
        self.out_l1_dist.set(l1)


class AverageTotalScoreJob(Job):
    # Extract average_total_score for one recognition pass from epoch_statistics.json

    def __init__(self, statistics_path: tk.Path, epoch: int):
        self.statistics_path = statistics_path
        self.epoch = epoch
        self.out_avg_total_score = self.output_var("avg_total_score")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        with open(str(self.statistics_path)) as f:
            stats = json.load(f)

        epoch_key = str(self.epoch)
        if epoch_key not in stats:
            self.out_avg_total_score.set(float("nan"))
            return

        raw = stats[epoch_key].get("average_total_score", float("nan"))
        self.out_avg_total_score.set(round(float(raw), 1))
