import os

from sisyphus import tk

from .. import audio


from i6_core.datasets.librispeech import DownloadLibriSpeechCorpusJob
from sisyphus import Job, Task

class CombineLibriSpeechJob(Job):
    def __init__(self, corpora_paths):
        self.corpora_paths = corpora_paths
        self.out_corpus_folder = self.output_path("train-other-960")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self.out_corpus_folder.mkdir()
        for corpus_path in self.corpora_paths:
            corpus_dir = str(corpus_path)
            for item in os.listdir(corpus_dir):
                src = os.path.join(corpus_dir, item)
                dst = os.path.join(self.out_corpus_folder.get_path(), item)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

def remove_silences_from_audio(librispeech_key: str):
    if librispeech_key == "train-other-960":
        parts = ["train-clean-100", "train-clean-360", "train-other-500"]
        paths = [DownloadLibriSpeechCorpusJob(p).out_corpus_folder for p in parts]
        audio_dir = CombineLibriSpeechJob(paths).out_corpus_folder
    else:
        audio_dir = DownloadLibriSpeechCorpusJob(librispeech_key).out_corpus_folder

    return audio.remove_silences_from_audio(
        audio_dir=audio_dir,
        concurrent=10,
    )


def get_featurized_audio(
    librispeech_key: str,
    existing_clusters: dict = None,
    existing_pca: dict = None,
    dump_hdf_concurrent: int = 10,
    featurize_concurrent: int = 10,
    remove_cluster_repetitions: bool = True,
):
    rem_audio_manifest = remove_silences_from_audio(librispeech_key)

    return audio.featurize_audio(
        librispeech_key=librispeech_key,
        input_audio_manifests=rem_audio_manifest,
        existing_clusters=existing_clusters,
        existing_pca=existing_pca,
        dump_hdf_concurrent=dump_hdf_concurrent,
        featurize_concurrent=featurize_concurrent,
        remove_cluster_repetitions=remove_cluster_repetitions,
    )
