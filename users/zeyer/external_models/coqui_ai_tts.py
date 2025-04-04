"""
Coqui-ai TTS: https://github.com/coqui-ai/TTS
own fork: https://github.com/albertz/coqui-ai-tts
"""

from __future__ import annotations
import os
import sys
import functools
from typing import Optional, TypeVar, Iterator

_my_dir = os.path.dirname(__file__)
_base_dir = functools.reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"
_returnn_dir = os.path.dirname(_base_dir) + "/tools/returnn"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.external_models"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        if _returnn_dir not in sys.path:
            sys.path.append(_returnn_dir)


_setup()


from sisyphus import Job, Task, Path, tk


def py():
    """
    demo to run this directly with Sisyphus
    """
    path = download_model("tts_models/multilingual/multi-dataset/your_tts")
    tk.register_output("external_models/coqui_ai_tts/your_tts", path)


def download_model(model_name: str, *, tts_repo_dir: Optional[Path] = None):
    """
    :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
    :param tts_repo_dir: if not specified, uses :func:`get_default_tts_repo_dir`
    """
    if tts_repo_dir is None:
        tts_repo_dir = get_default_tts_repo_dir()
    download_job = DownloadModelJob(model_name=model_name, tts_repo_dir=tts_repo_dir)
    return download_job.out_tts_data_dir


def get_default_tts_repo_dir() -> Path:
    """
    :return: upstream, via :func:`get_upstream_tts_git_repo`
    """
    return get_upstream_tts_git_repo()


def get_upstream_tts_git_repo() -> Path:
    """
    :return: upstream, via :class:`CloneGitRepositoryJob`, from https://github.com/coqui-ai/TTS.git
    """
    from i6_core.tools.git import CloneGitRepositoryJob

    clone_job = CloneGitRepositoryJob(
        "https://github.com/coqui-ai/TTS.git", commit="dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
    )
    return clone_job.out_repository


class DownloadModelJob(Job):
    """
    Downloads a model via the Coqui-ai TTS api.
    See: https://github.com/coqui-ai/TTS?tab=readme-ov-file#-python-api

    To use it later, set the ``TTS_HOME`` env var to ``out_tts_data_dir.get_path()``.
    Also, set the env var ``COQUI_TOS_AGREED=1``.

    Thus, it requires the TTS repo (with TTS source code; e.g. via :func:`get_default_tts_repo_dir`),
    and then will use that Python API.
    Thus, it requires the TTS dependencies: https://github.com/coqui-ai/TTS/blob/dev/requirements.txt
    Specifically, that should be:
    - torch
    - coqpit
    - trainer (might need --ignore-requires-python)
    - tqdm
    - pysbd
    - mutagen
    - pandas
    - anyascii
    - inflect
    - bangla
    - bnnumerizer
    - bnunicodenormalizer
    - gruut
    - jamo
    - jieba
    - pypinyin
    """

    def __init__(self, *, model_name: str, tts_repo_dir: Path):
        """
        :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
        :param tts_repo_dir:
        """
        super().__init__()
        self.model_name = model_name
        self.tts_repo_dir = tts_repo_dir
        self.out_tts_data_dir = self.output_path("tts_home", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import tempfile
        import shutil
        from sisyphus import gs

        sys.path.insert(0, self.tts_repo_dir.get_path())

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            print("using temp-dir:", tmp_dir)

            os.environ["TTS_HOME"] = tmp_dir
            os.environ["COQUI_TOS_AGREED"] = "1"

            from TTS.api import TTS

            tts = TTS(model_name=self.model_name, progress_bar=False)
            assert str(tts.manager.output_prefix) == tmp_dir + "/tts"
            assert os.path.isdir(tts.manager.output_prefix)
            dir_content = os.listdir(tts.manager.output_prefix)
            print(".../tts dir content:", dir_content)
            assert dir_content  # non-empty
            shutil.copytree(tts.manager.output_prefix, self.out_tts_data_dir.get_path() + "/tts")


class ExtractVocabFromModelJob(Job):
    """
    Extracts the vocab (as line-based txt file) from a TTS model.
    """

    def __init__(self, *, model_name: str, tts_model_dir: Path, tts_repo_dir: Path):
        """
        :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
        :param tts_model_dir: the ``out_tts_data_dir`` from a :class:`DownloadModelJob` job
        :param tts_repo_dir: the TTS repo dir, e.g. from :func:`get_default_tts_repo_dir`
        """
        super().__init__()
        self.model_name = model_name
        self.tts_model_dir = tts_model_dir
        self.tts_repo_dir = tts_repo_dir
        self.out_vocab_file = self.output_path("vocab.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os

        sys.path.insert(0, self.tts_repo_dir.get_path())
        os.environ["TTS_HOME"] = self.tts_model_dir.get_path()
        os.environ["COQUI_TOS_AGREED"] = "1"

        print("Importing TTS...")

        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.tts.models.vits import Vits  # just for typing, just as an example

        def _disallowed_create_dir_and_download_model(*_args, **_kwargs):
            raise RuntimeError(
                f"Disallowed create_dir_and_download_model({_args}, {_kwargs}),"
                f" model {self.model_name} not found, model dir {self.tts_model_dir} not valid? "
            )

        # patch to avoid any accidental download
        assert hasattr(ModelManager, "create_dir_and_download_model")
        ModelManager.create_dir_and_download_model = _disallowed_create_dir_and_download_model

        print("Loading TTS model...")
        tts = TTS(model_name=self.model_name, progress_bar=False)

        tts_model: Vits = tts.synthesizer.tts_model  # typing Vits is just an example

        with open(self.out_vocab_file.get_path(), "w", encoding="utf8") as f:
            for v in tts_model.tokenizer.characters.vocab:
                f.write(f"{v}\n")


def _demo():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/your_tts")
    arg_parser.add_argument("--model-dir", default="output/external_models/coqui_ai_tts/your_tts")
    arg_parser.add_argument("--tts-repo-dir")
    arg_parser.add_argument("--device", default="cuda")
    arg_parser.add_argument("--language")
    arg_parser.add_argument("--use-mean-emb", action="store_true")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()

    print(f"{os.path.basename(__file__)} demo, {args=}")

    import random
    import torch
    import numpy as np

    dev = torch.device(args.device)
    rnd = random.Random(args.seed)
    torch.random.manual_seed(rnd.randint(0, 2**32 - 1))

    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass

    model_name = args.model_name
    model_dir = args.model_dir
    assert os.path.exists(model_dir), f"model dir does not exist: {model_dir}"
    assert os.path.exists(model_dir + "/tts"), f"model dir does not exist: {model_dir}/tts"

    tts_repo_dir = args.tts_repo_dir
    if not tts_repo_dir:
        tts_repo_dir = get_default_tts_repo_dir().get_path()
        print("Using default TTS repo dir:", tts_repo_dir)

    sys.path.insert(0, tts_repo_dir)
    os.environ["TTS_HOME"] = model_dir
    os.environ["COQUI_TOS_AGREED"] = "1"

    print("Importing TTS...")

    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import synthesis
    from TTS.tts.utils.text.cleaners import multilingual_cleaners
    from TTS.utils.audio.numpy_transforms import save_wav
    from TTS.tts.models.vits import Vits

    def _disallowed_create_dir_and_download_model(*_args, **_kwargs):
        raise RuntimeError(
            f"Disallowed create_dir_and_download_model({_args}, {_kwargs}),"
            f" model {model_name} not found, model dir {model_dir} not valid? "
        )

    # patch to avoid any accidental download
    assert hasattr(ModelManager, "create_dir_and_download_model")
    ModelManager.create_dir_and_download_model = _disallowed_create_dir_and_download_model

    print("Loading TTS model...")
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.to(dev)

    # See tts.tts() func to how it generates audio.
    # This is a high-level wrapper, but we want to call it more directly to be able to use it in batch mode.

    tts_model: Vits = tts.synthesizer.tts_model  # typing Vits is just an example
    tts_config = tts.synthesizer.tts_config
    sample_rate = tts.synthesizer.output_sample_rate

    print(f"{type(tts_model) = }")

    speakers = tts.speakers
    print(f"speakers: {speakers}")
    speaker = None
    speaker_id = None
    speaker_embedding = None
    if tts_model.speaker_manager:
        speaker_id = rnd.randint(0, tts_model.speaker_manager.num_embeddings - 1)
        print("random speaker id:", speaker_id)
        emb = list(tts_model.speaker_manager.embeddings.values())[speaker_id]
        speaker = emb["name"]
        if args.use_mean_emb:
            speaker_embedding = tts_model.speaker_manager.get_mean_embedding(speaker)
        else:
            speaker_embedding = emb["embedding"]  # [emb_dim]
        speaker_embedding = np.array(speaker_embedding, dtype=np.float32)
    print(f"speaker: {speaker!r}")

    print("emb_g:", tts_model.emb_g if hasattr(tts_model, "emb_g") else None)  # speaker embedding
    print(f"{tts_config.use_d_vector_file = }")
    if tts_model.speaker_manager:
        print(f"{len(tts_model.speaker_manager.embeddings) = }")
        # print("embedding keys:", list(tts_model.speaker_manager.embeddings.keys()))
        # for emb in tts_model.speaker_manager.embeddings.values():
        #     print(f"  name={emb['name']!r}, keys:{list(emb.keys())}, dim:{len(emb['embedding'])}")

    language = args.language
    if tts_model.language_manager:
        languages = tts_model.language_manager.language_names
        print(f"{len(languages) = }, {languages = }")
        if language is None:
            language = rnd.choice(languages)
            print("Picked random language:", language)
        else:
            assert language in languages, f"language {language} not in {languages}"
    else:
        print("No language manager")

    print(f"{hasattr(tts_model, "synthesize") = }")  # if False, then it will call the synthesis func
    assert not hasattr(tts_model, "synthesize")  # we assume that we can call the synthesis func
    print(f"{sample_rate = }")

    # could also take this from config test_sentences
    text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

    # This is maybe Vits specific:
    print(f"{tts_model.length_scale = }")
    tts_model.length_scale = 1.0  # overwrite (testing...)
    print(f"{tts_model.inference_noise_scale = }")
    print(f"{tts_model.inference_noise_scale_dp = }")

    # Test high-level API.
    wav = tts.tts(text, speaker=speaker, language=language)
    print("tts() output:", type(wav))
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    assert isinstance(wav, np.ndarray)
    print("shape:", wav.shape)
    save_wav(wav=wav, path="demo.wav", sample_rate=sample_rate)

    # Generate another one. How much randomness?
    wav = tts.tts(text, speaker=speaker, language=language)
    print("tts() output:", type(wav))
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    assert isinstance(wav, np.ndarray)
    print("shape:", wav.shape)
    save_wav(wav=wav, path="demo2.wav", sample_rate=sample_rate)

    if tts_model.speaker_manager:
        if tts_config.use_d_vector_file:
            speaker_id = None  # the model expects speaker_embedding only, not speaker_id
        else:
            speaker_embedding = None  # speaker_id is used

    # Now do the synthesis directly, also including batching.
    # See also TTS.utils.synthesizer.synthesis().

    # outputs = synthesis(
    #     model=tts_model,
    #     text=txt,
    #     CONFIG=tts_config,
    #     use_cuda=use_cuda,
    #     speaker_id=speaker_id,
    #     style_wav=style_wav,
    #     style_text=style_text,
    #     use_griffin_lim=use_gl,
    #     d_vector=speaker_embedding,
    #     language_id=language_id,
    # )
    # waveform = outputs["wav"]

    print(f"{tts_model.tokenizer = }")  # <TTS.tts.utils.text.tokenizer.TTSTokenizer ...>
    print(f"{tts_model.tokenizer.characters = }")  # YourTTS: <TTS.tts.models.vits.VitsCharacters ...>
    print(f"{tts_model.tokenizer.text_cleaner = }")  # YourTTS: func multilingual_cleaners
    print(f"{tts_model.tokenizer.use_phonemes = }")  # YourTTS: False
    print(f"{tts_model.tokenizer.add_blank = }")  # YourTTS: True
    print(f"{tts_model.tokenizer.characters.blank = }")  # YourTTS: '<BLNK>'
    print(f"{tts_model.tokenizer.use_eos_bos = }")  # YourTTS: False

    import tempfile

    vocab_file = tempfile.NamedTemporaryFile(suffix=".vocab.txt", mode="w", encoding="utf8", delete_on_close=False)
    corpus_file = tempfile.NamedTemporaryFile(suffix=".corpus.txt", mode="w", encoding="utf8", delete_on_close=False)

    for v in tts_model.tokenizer.characters.vocab:
        vocab_file.write(f"{v}\n")
    vocab_file.close()

    # convert text to sequence of token IDs
    # For YourTTS specifically, using VitsCharacters:
    # _vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
    # _char_to_id = {char: idx for idx, char in enumerate(_vocab)}
    # tokenizer.text_to_ids:
    #         if self.text_cleaner is not None:  # -- done in LmDataset preprocessing
    #             text = self.text_cleaner(text)
    #         if self.use_phonemes:  # -- not needed, assert False. could be done in LmDataset preprocessing
    #             text = self.phonemizer.phonemize(text, separator="", language=language)
    #         text = self.encode(text)  # char_to_id. can also remap in model if necessary
    #         if self.add_blank:  # -- can be done in model
    #             text = self.intersperse_blank_char(text, True)
    #         if self.use_eos_bos:  # -- not needed, assert False. could be done in model
    #             text = self.pad_with_bos_eos(text)

    # Multiple texts, to test batching.
    texts = [
        text,
        "I am a very long text, and I am not sure if it will be cut off or not.",
        "Hello world, this is a test.",
        "RETURNN is a framework for building neural networks. It is very flexible and powerful.",
    ]
    for t in texts:
        corpus_file.write(f"{t}\n")
    corpus_file.close()

    from returnn.datasets.lm import LmDataset

    ds = LmDataset(
        corpus_file=corpus_file.name,
        orth_vocab={
            "class": "CharacterTargets",
            "vocab_file": vocab_file.name,
            "unknown_label": None,
        },
        orth_post_process=multilingual_cleaners,
        dtype="int32",
    )
    ds.initialize()

    from returnn.torch.data.returnn_dataset_wrapper import ReturnnDatasetIterDataPipe
    from returnn.torch.data.pipeline import BatchingIterDataPipe, collate_batch

    ds_ = ReturnnDatasetIterDataPipe(ds)
    ds_ = BatchingIterDataPipe(ds_, batch_size=None, max_seqs=None)
    batch = next(iter(ds_))
    batch_ = collate_batch(batch)
    print("batch entries:", list(batch_.keys()))
    assert len(batch) == batch_["data"].shape[0] == len(texts)
    batch_size = len(batch)

    # Just checking our LmDataset preprocessing to the tokenizer.text_to_ids.
    for b, t in enumerate(texts):
        ref = torch.tensor(tts_model.tokenizer.text_to_ids(t, language=language), dtype=torch.int32)
        ds_seq_len = batch_["data:seq_len"][b]
        ds_seq = batch_["data"][b][:ds_seq_len]
        assert ds_seq_len == len(ref) and np.array_equal(ref, ds_seq)

    text_inputs = batch_["data"].to(dev)  # [B, T_in]
    print(f"{text_inputs.shape = }")
    text_inputs_lens = batch_["data:seq_len"].to(dev)  # [B]
    speaker_id = torch.tensor(speaker_id, device=dev).expand(batch_size) if speaker_id is not None else None  # [B]
    speaker_embedding = torch.tensor(speaker_embedding, device=dev, dtype=torch.float32) if speaker_embedding is not None else None
    speaker_embedding = speaker_embedding.expand(batch_size, -1) if speaker_embedding is not None else None  # [B, emb_dim]
    language_id = torch.tensor(tts_model.language_manager.name_to_id[language], device=dev).expand(batch_size) if language else None  # [B]

    outputs = tts_model.inference(
        text_inputs,
        aux_input={
            "x_lengths": text_inputs_lens,
            "speaker_ids": speaker_id,
            "d_vectors": speaker_embedding,
            "language_ids": language_id,
        },
    )
    print(outputs)

    model_outputs = outputs["model_outputs"]  # for Vits: [B, 1, T_wav]
    y_mask = outputs["y_mask"]  # before the final waveform_decoder
    print(f"{model_outputs.shape = }")
    print(f"{y_mask.shape = }")  # [B, 1, T_wav]
    y_mask = y_mask.squeeze(1)  # [B, T_wav]
    y_lens = torch.sum(y_mask.to(torch.int32), dim=1)  # [B]
    print(f"{y_lens = }")

    # convert outputs to numpy. select first batch
    model_outputs = model_outputs[0].cpu().squeeze().numpy()  # [T_wav]
    assert model_outputs.ndim == 1
    wav = model_outputs
    save_wav(wav=wav, path="demo3.wav", sample_rate=sample_rate)

    # if hasattr(tts_model, "waveform_decoder"):
    #     print("waveform_decoder:", tts_model.waveform_decoder)

    # Get generic output sizes.
    from sympy.utilities.lambdify import lambdify

    # noinspection PyProtectedMember
    from torch._subclasses.fake_tensor import FakeTensorMode

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    # noinspection PyProtectedMember
    from torch._dynamo.source import ConstantSource

    shape_env = ShapeEnv(duck_shape=False)
    with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env):
        # Assuming that waveform_decoder is from Vits.
        time_sym = shape_env.create_symbol(100, ConstantSource("T"))
        time_sym_ = shape_env.create_symintnode(time_sym, hint=None)
        fake_in = torch.empty(1, tts_model.waveform_decoder.conv_pre.in_channels, time_sym_)
        if getattr(tts_model.waveform_decoder, "cond_layer", None):
            fake_g_in = torch.empty(1, tts_model.waveform_decoder.cond_layer.in_channels, time_sym_)
        else:
            fake_g_in = None

        out = tts_model.waveform_decoder(fake_in, g=fake_g_in)
        print(f"{out.shape = }")
        out_size = out.shape[-1]
        assert isinstance(out_size, torch.SymInt)
        out_sym = out_size.node.expr
        out_sym_lambda = lambdify(time_sym, out_sym)

    out_lens = out_sym_lambda(y_lens)
    print(f"{out_lens = }")


if __name__ == "__main__":
    _demo()
