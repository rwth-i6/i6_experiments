import io
import os
import subprocess as sp
from typing import Optional, Type
from sisyphus import Job, Task, tk
from sisyphus.job_path import AbstractPath, Path, Variable
import logging
import shutil

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import get_w2vu_librispeech_transcription, create_ctm_from_json

class FairseqGenerateWav2VecUJob(Job):
    """
    Run the w2vu_generate.py script with specified configurations for Wav2Vec generation.
    """

    def __init__(
        self,
        decoding_audio_name,
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
        :param decoding_audio_name: Name of the decoding audio
        :param environment: Path to the Python virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param task_data: Path to the directory with features.
        :param checkpoints_path: Path to the GAN checkpoints.
        :param config_dir: Path to the configuration directory (default: "config/generate").
        :param config_name: Name of the Hydra configuration (default: "viterbi").
        :param gen_subset: Subset to generate (default: "valid").
        """
        self.decoding_audio_name = decoding_audio_name
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

        self.transcription_generated = self.output_path(f"transcriptions/{self.gen_subset}.txt")

        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")

        self.out_label_path_dict = self.output_path(f"labels_path.json")

        self.rqmt = {"time": 1000, "cpu": 1, "mem": 32}
        if self.decoding_audio_name == "train-other-960":
            self.rqmt = {"time": 2000, "cpu": 1, "gpu": 1, "mem": 150}            
        

    def tasks(self):
        yield Task("copy_dict_phn_txt", mini_task=True)
        yield Task("run", rqmt=self.rqmt)
        yield Task("format_transcription", mini_task=True)

    def copy_dict_phn_txt(self):
        dict_phn_path = os.path.join(self.prepare_text.get_path(), "phones/dict.phn.txt")
        dest = os.path.join(self.task_data.get_path(), "dict.phn.txt")
        shutil.copy2(dict_phn_path, dest)
        logging.info(f"Coppied {dict_phn_path} to {dest}")

    def run(self):
        # Extract the best checkpoint path from the provided checkpoints_path, which could be a folder or a file.
        best_checkpoint_path = ""
        chckpt_found = False
        if isinstance(self.checkpoints_path, Variable):
            self.checkpoints_path = self.checkpoints_path.get()
        else:
            self.checkpoints_path = self.checkpoints_path.get_path() # asume its a tk.Path
        if os.path.isfile(self.checkpoints_path):
            if self.checkpoints_path.endswith(".pt"):
                chckpt_found = True
                best_checkpoint_path = self.checkpoints_path
            else:
                raise ValueError(
                    f"Expected a .pt file for checkpoint or a directory containing checkpoint_best.pt, but got an invalid file {self.checkpoints_path}."
                )

        else:
            for root, dirs, files in os.walk(self.checkpoints_path):
                for file in files:
                    if file == "checkpoint_best.pt":
                        chckpt_found = True
                        best_checkpoint_path = tk.Path(os.path.join(root, file))
        if not chckpt_found:
            raise FileNotFoundError("No checkpoint_best.pt found in the specified path.")
        logging.info(f"Using checkpoint: {self.checkpoints_path}")

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
            kaldi_decoder_config.hlg_graph_path={os.path.join(self.prepare_text.get_path(), 'fst/phn_to_words_sil/HLG.phn.kenlm.wrd.o4.fst')} \
            kaldi_decoder_config.output_dict={os.path.join(self.prepare_text.get_path(), 'fst/phn_to_words_sil/kaldi_dict.kenlm.wrd.o4.txt')} \
            viterbi_transcript="" \
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
            fairseq.common_eval.path={self.checkpoints_path} \
            fairseq.task.data={self.task_data.get_path()} \
            fairseq.dataset.gen_subset={self.gen_subset} \
            fairseq.common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            results_path={self.results_path.get_path()} \
            """
            + self.extra_config
        )

        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    def format_transcription(self):
        """
        Format the generated transcription to a more readable format.
        """
        import json
        import gzip

        hypothesis_formatted = {}

        manifest_path = self.task_data.get_path() + "/" + self.gen_subset + ".tsv"

        ids = []

        with open(manifest_path, "r") as manifest_file:
            lines = manifest_file.readlines()
    
            base = self.decoding_audio_name + '/'

            for line in lines[1:]:  # Skip header
                line = line.split(' ')
                line_id =  line[0]  # e.g. 71550/3703-71550-0004.flac

                slash_index = line_id.rfind('/')
                # find the last occurrence of ".flac"
                flac_index = line_id.rfind('.flac')
                
                # slice between them 3703-71550-0004
                line_id = line_id[slash_index + 1:flac_index]

                ids.append(base + line_id + '/' + line_id)

        transcriptions = []

        with open(self.transcription_generated.get_path(), "r") as gen_file:
            for line in gen_file:
                line = line.strip().replace('\n', '').upper()
                transcriptions.append(line)


        assert len(ids) == len(transcriptions), f"Length mismatch: {len(ids)} ids vs {len(transcriptions)} transcriptions"
        for i in range(len(ids)):
            hypothesis_formatted[ids[i]] = transcriptions[i]

        # save as gzipped JSON
        with gzip.open(self.transcription_generated_formatted, "wt", encoding="utf-8") as f:
            json.dump(hypothesis_formatted, f, ensure_ascii=False, indent=2)

        label_path_dict = {"path": {self.decoding_audio_name: self.transcription_generated_formatted.get_path()}, "score": -1, "epoch": -1}

        with open(self.out_label_path_dict.get_path(), "w") as f:
            f.write(json.dumps(label_path_dict))
            f.write("\n")


        create_ctm_from_json(
            self.transcription_generated_formatted.get_path(), self.transcription_generated_ctm.get_path()
        ) 


class FairseqKaldiDecodingJob(Job):
    """
    Run the w2vu_generate.py script with specified configurations for Wav2Vec generation.
    """

    def __init__(
        self,
        decoding_audio_name,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        task_data: Type[tk.Path],
        lexicon_lst_path: Type[tk.Path],
        hlg_graph_path: Type[tk.Path],
        kaldi_dict: Type[tk.Path],
        checkpoints_path: Type[tk.Path],
        gen_subset: Optional[str] = None,
        config_dir: Optional[str] = None,
        config_name: Optional[str] = None,
        dict_phn_txt_path: Optional[tk.Path] = None,
        extra_config: Optional[str] = None,
    ):
        """
        :param decoding_audio_name: Name of the decoding audio
        :param environment: Path to the Python virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param task_data: Path to the directory with features.
        :param checkpoints_path: Path to the GAN checkpoints.
        :param config_dir: Path to the configuration directory (default: "config/generate").
        :param config_name: Name of the Hydra configuration (default: "viterbi").
        :param gen_subset: Subset to generate (default: "valid").
        """
        self.decoding_audio_name = decoding_audio_name
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.task_data = task_data
        self.lexicon_lst_path = lexicon_lst_path
        self.hlg_graph_path = hlg_graph_path
        self.kaldi_dict = kaldi_dict
        self.checkpoints_path = checkpoints_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.gen_subset = gen_subset 
        self.extra_config = extra_config if extra_config is not None else ""

        self.results_path = self.output_path("transcriptions", directory=True)
        
        self.shard_id  = None
        if "shard_id=" in self.extra_config:
            shard_id_index = self.extra_config.index("shard_id=") + len("shard_id=")
            shard_id_end_index = self.extra_config.find(" ", shard_id_index)
            if shard_id_end_index == -1:
                shard_id_end_index = len(self.extra_config)
            else:
                self.shard_id = self.extra_config[shard_id_index:shard_id_end_index]+"_"

        self.num_shards = None
        if "num_shards=" in self.extra_config:
            num_shards_index = self.extra_config.index("num_shards=") + len("num_shards=")
            num_shards_end_index = self.extra_config.find(" ", num_shards_index)
            if num_shards_end_index == -1:
                num_shards_end_index = len(self.extra_config)
            self.num_shards = self.extra_config[num_shards_index:num_shards_end_index]

        self.manifest_path = tk.Path(self.task_data.get_path() + "/" + self.gen_subset + ".tsv")

        self.transcription_generated = self.output_path(f"transcriptions/{self.gen_subset}{self.shard_id if self.shard_id is not None else ''}.txt")
        self.transcription_generated_formatted = self.output_path(f"search_results.py.gz")
        self.transcription_generated_ctm = self.output_path(f"search_results.ctm")

        self.out_label_path_dict = self.output_path(f"labels_path.json")

        self.rqmt = {"time": 1000, "gpu": 1, "mem": 32}         
        

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
        yield Task("format_transcription", mini_task=True)

    def run(self):
        if isinstance(self.checkpoints_path, Variable):
            self.checkpoints_path = self.checkpoints_path.get()
        else:
            assert isinstance(self.checkpoints_path, tk.Path), "checkpoints_path should be of type tk.Path or Variable"
            self.checkpoints_path = self.checkpoints_path.get_path() # asume its a tk.Path

        logging.info(f"Using checkpoint: {self.checkpoints_path}")
        
        self.extra_config += f""" \
            lexicon={self.lexicon_lst_path.get_path()} \
            kaldi_decoder_config.hlg_graph_path={self.hlg_graph_path.get_path()} \
            kaldi_decoder_config.output_dict={self.kaldi_dict.get_path()} \
            viterbi_transcript="" \
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
            fairseq.common_eval.path={self.checkpoints_path} \
            fairseq.task.data={self.task_data.get_path()} \
            fairseq.dataset.gen_subset={self.gen_subset} \
            fairseq.common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            results_path={self.results_path.get_path()} \
            """
            + self.extra_config
        )

        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    def format_transcription(self):
        """
        Format the generated transcription to a more readable format.
        """
        import json
        import gzip

        hypothesis_formatted = {}

        self.manifest_path = self.task_data.get_path() + "/" + self.gen_subset + ".tsv"

        ids = []

        shard_id_int = int(self.shard_id[:-1]) if self.shard_id is not None else 0
        num_shards_int = int(self.num_shards) if self.num_shards is not None else 1

        with open(self.manifest_path, "r") as manifest_file:
            lines = manifest_file.readlines()
    
            base = self.decoding_audio_name + '/'

            lines = lines[1:]  # Skip header

            if num_shards_int > 1:
                lines = lines[shard_id_int::num_shards_int]

            for line in lines[:]:  # Skip header
                line = line.split(' ')
                line_id =  line[0]  # e.g. 71550/3703-71550-0004.flac

                slash_index = line_id.rfind('/')
                # find the last occurrence of ".flac"
                flac_index = line_id.rfind('.flac')
                
                # slice between them 3703-71550-0004
                line_id = line_id[slash_index + 1:flac_index]

                ids.append(base + line_id + '/' + line_id)

        transcriptions = []

        with open(self.transcription_generated.get_path(), "r") as gen_file:
            for line in gen_file:
                line = line.strip().replace('\n', '').upper()
                transcriptions.append(line)


        assert len(ids) == len(transcriptions), f"Length mismatch: {len(ids)} ids vs {len(transcriptions)} transcriptions"
        for i in range(len(ids)):
            hypothesis_formatted[ids[i]] = transcriptions[i]

        # save as gzipped JSON
        with gzip.open(self.transcription_generated_formatted, "wt", encoding="utf-8") as f:
            json.dump(hypothesis_formatted, f, ensure_ascii=False, indent=2)

        label_path_dict = {"path": {self.decoding_audio_name: self.transcription_generated_formatted.get_path()}, "score": -1, "epoch": -1}

        with open(self.out_label_path_dict.get_path(), "w") as f:
            f.write(json.dumps(label_path_dict))
            f.write("\n")


        create_ctm_from_json(
            self.transcription_generated_formatted.get_path(), self.transcription_generated_ctm.get_path()
        ) 

class ViterbiGenerateWav2VecUJob(FairseqGenerateWav2VecUJob):
    def __init__(
        self,
        decoding_audio_name,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        task_data: Type[tk.Path],
        prepare_text: Type[tk.Path],
        checkpoints_path: Type[tk.Path],
        gen_subset: Optional[str] = None,
        extra_config: Optional[str] = None,
    ):
        self.decoding_audio_name = decoding_audio_name
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.task_data = task_data
        self.prepare_text = prepare_text
        self.checkpoints_path = checkpoints_path
        self.gen_subset = gen_subset
        self.extra_config = extra_config if extra_config is not None else ""

        self.results_path = self.output_path("transcriptions", directory=True)
        
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

        self.transcription_generated = self.output_path(f"transcriptions/{self.gen_subset}.txt")

        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 32}
        if self.decoding_audio_name == "train-other-960":
            self.rqmt = {"time": 2000, "cpu": 1, "gpu": 1, "mem": 100}   

    def tasks(self):
        yield Task("copy_dict_phn_txt", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def copy_dict_phn_txt(self):
        dict_phn_path = os.path.join(self.prepare_text.get_path(), "phones/dict.phn.txt")
        dest = os.path.join(self.task_data.get_path(), "dict.phn.txt")
        shutil.copy2(dict_phn_path, dest)
        logging.info(f"Coppied {dict_phn_path} to {dest}")

    def run(self):
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

        self.config_dir = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/config/generate"
        )

        self.config_name = "viterbi"

        sh_call_str = ""
        if self.environment is not None:
            sh_call_str += f"export PYTHONNOUSERSITE=1 && source {self.environment.get_path()}/bin/activate && "

        self.extra_config += f""" \
            viterbi_transcript='' \
            """

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
