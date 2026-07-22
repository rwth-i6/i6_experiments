import os

from sisyphus import tk

from .. import audio


def remove_silences_from_audio(librispeech_key: str):
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))

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
