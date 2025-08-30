import os
import subprocess as sp
from typing import Optional, Type
from sisyphus import Job, Task, tk
import logging
import shutil


class FairseqGenerateWav2VecUJob(Job):
    """
    Run the w2vu_generate.py script with specified configurations for Wav2Vec generation.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        task_data: Type[tk.Path],
        prepare_text: Type[tk.Path],
        checkpoints_path: Type[tk.Path],
        gen_subset: Optional[str] = None,
        config_dir: Optional[str] = None,
        config_name: Optional[str] = None,
        dict_phn_txt_path: Optional[tk.Path] = None,
        dict_wrd_txt_path: Optional[tk.Path] = None,
        extra_config: Optional[str] = None,
    ):
        """
        :param environment: Path to the Python virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param task_data: Path to the directory with features.
        :param checkpoints_path: Path to the GAN checkpoints.
        :param config_dir: Path to the configuration directory (default: "config/generate").
        :param config_name: Name of the Hydra configuration (default: "viterbi").
        :param gen_subset: Subset to generate (default: "valid").
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.task_data = task_data
        self.prepare_text = prepare_text
        self.checkpoints_path = checkpoints_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.gen_subset = gen_subset
        self.dict_phn_txt_path = dict_phn_txt_path
        self.dict_wrd_txt_path = dict_wrd_txt_path
        self.extra_config = extra_config if extra_config is not None else ""

        self.results_path = self.output_path("transcriptions", directory=True)
        self.transcription_generated = self.output_path(f"transcriptions/{gen_subset}.txt")

        # Resource requirements (adjust as needed)
        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 70}

    def tasks(self):
        yield Task("copy_dict_phn_txt", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def copy_dict_phn_txt(self):
        dict_phn_path = os.path.join(self.prepare_text.get_path(), "phones/dict.phn.txt")
        dest = os.path.join(self.task_data.get_path(), "dict.phn.txt")
        shutil.copy2(dict_phn_path, dest)
        logging.info(f"Coppied {dict_phn_path} to {dest}")

    def run(self):
        logging.info(self.gen_subset)
        if self.gen_subset == None:
            feat_subsets = []
            for file in os.listdir(self.task_data.get_path()):
                if file.endswith(".npy"):
                    feat_subsets.append(file[:-4])

            assert len(feat_subsets) > 0, "No features found for decoding"

            for s in feat_subsets:
                if s.endswith("valid"):
                    self.gen_subset = s
                    break

            if self.gen_subset == None:
                self.gen_subset = feat_subsets[0]

            logging.info(f"Found gen_subset: {self.gen_subset}, will be used for decoding")

        # Extract the best checkpoint path from the provided checkpoints_path, which could be a folder or a file.
        best_checkpoint_path = ""
        chckpt_found = False
        if os.path.isfile(self.checkpoints_path.get_path()):
            if self.checkpoints_path.get_path().endswith(".pt"):
                chckpt_found = True
                best_checkpoint_path = self.checkpoints_path
            else:
                raise ValueError(
                    "Expected a .pt file for checkpoint or a directory containing checkpoint_best.pt, but got an invalid file."
                )

        else:
            for root, dirs, files in os.walk(self.checkpoints_path.get_path()):
                for file in files:
                    if file == "checkpoint_best.pt":
                        chckpt_found = True
                        best_checkpoint_path = tk.Path(os.path.join(root, file))
        if not chckpt_found:
            raise FileNotFoundError("No checkpoint_best.pt found in the specified path.")
        logging.info(f"Using checkpoint: {best_checkpoint_path.get_path()}")

        if self.dict_phn_txt_path != None and self.dict_phn_txt_path != "":
            dest_phn = os.path.join(self.task_data.get_path(), os.path.basename(self.dict_phn_txt_path.get_path()))
            shutil.copy(self.dict_phn_txt_path.get_path(), dest_phn)
            logging.info(f"Copied dict_phn_txt_path to {dest_phn}")

        if self.dict_wrd_txt_path != None and self.dict_wrd_txt_path != "":
            dest_wrd = os.path.join(self.task_data.get_path(), os.path.basename(self.dict_wrd_txt_path.get_path()))
            shutil.copy(self.dict_wrd_txt_path.get_path(), dest_wrd)
            logging.info(f"Copied dict_wrd_txt_path to {dest_wrd}")

        if not self.config_dir:
            logging.warning("No config_dir provided, using default.")
            self.config_dir = os.path.join(
                self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/config/generate"
            )

        if not self.config_name:
            logging.warning("No config_name provided, using default.")
            self.config_name = "viterbi"

        # TODO: THIS SHOULD CHANGE DEPENDING ON THE CONFIG!! this is specific for the kaldi decoding
        self.extra_config += f""" \
            lexicon={os.path.join(self.prepare_text.get_path(), 'lexicon.lst')} \
            kaldi_decoder_config.hlg_graph_path={os.path.join(self.prepare_text.get_path(), 'fst/phn_to_words_sil/HLG.phn.kenlm.wrd.o40014.fst')} \
            kaldi_decoder_config.output_dict={os.path.join(self.prepare_text.get_path(), 'fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o40014.txt')} \
            """

        sh_call_str = ""
        if self.environment is not None:
            sh_call_str += f"export PYTHONNOUSERSITE=1 && source {self.environment.get_path()}/bin/activate && "

        sh_call_str = (
            sh_call_str
            + f""" \
            export HYDRA_FULL_ERROR=1 && \
            /opt/conda/bin/python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/w2vu_generate.py \
            --config-dir {self.config_dir} \
            --config-name {self.config_name} \
            fairseq.common_eval.path={best_checkpoint_path.get_path()} \
            fairseq.task.data={self.task_data.get_path()} \
            fairseq.dataset.gen_subset={self.gen_subset} \
            fairseq.common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            results_path={self.results_path.get_path()} \
            """
            + self.extra_config
        )

        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)
