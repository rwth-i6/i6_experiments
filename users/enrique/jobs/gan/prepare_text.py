import os.path as osp
import os
import random
import subprocess as sp
from typing import List, Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
import numpy as np
from sisyphus import tools

class WordToLetterJob(Job):
    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}

    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        words_file: tk.Path,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.words_file = words_file
        self.letters_file = self.output_path("letters.txt")

    def tasks(self):
        yield Task("convert_words_to_letters", rqmt={"time": 1000, "cpu": 1, "mem": 16})

    def convert_words_to_letters(self):
        word_lines = []
        with open(self.words_file.get_path(), "r") as wf:
            word_lines = wf.readlines()

        assert len(word_lines) > 0, "Words file is empty."
        letter_lines = []
        for line in word_lines:
            word = line.strip()
            letters = " ".join(list(word))
            letter_lines.append(f"{letters}\n")

        with open(self.letters_file.get_path(), "w") as lf:
            lf.writelines(letter_lines)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


class FairseqG2PWordToPhnJob(Job):
    """
    Job to convert words to phonemes using a G2P model.
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < $target_dir/words.txt > $target_dir/phones.txt
    """

    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}

    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        words_file: tk.Path,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.words_file = words_file
        self.phn_file = self.output_path("phones.txt")

    def tasks(self):
        yield Task("convert_words_to_phonemes", rqmt={"time": 1000, "cpu": 1, "mem": 16})

    def env_call(self):
        """Prepare the fairseq_python_env activation call string."""
        env_call_str = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()}  &&  export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH && "
        )
        if self.fairseq_python_env:
            env_call_str += f"source {self.fairseq_python_env.get_path()}/bin/activate && "
        return env_call_str
    
    def convert_words_to_phonemes(self):
        script_path = os.path.join(
            self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py"
        )

        sh_call = (
            f"python {script_path} --compact < {self.words_file.get_path()} > {self.phn_file.get_path()}"
        )

        sp.run(self.env_call() + sh_call, shell=True, check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))
        
class CreateLexiconAndDictionaryJob(Job):
    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}
    __sis_hash_constants__ = {"label_type": ""}
    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        words_txt: Type[tk.Path],
        tokenized_words: Type[tk.Path],
        label_type: str = "",
        sil_token: str = "<SIL>",
        each_label_count_threshold: int = 0,
    ):
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.words_txt = words_txt
        self.tokenized_words = tokenized_words
        self.label_type = label_type
        self.silence_token = sil_token
        self.each_label_count_threshold = each_label_count_threshold

        # Outputs
        self.lexicon_lst = self.output_path("lexicon.lst")
        self.lexicon_filtered_lst = self.output_path("lexicon_filtered.lst")
        
        # Intermediate folder for dict generation
        self.dict_folder = self.output_path("dict_folder", directory=True)
        self.dict_txt = self.output_path("dict_folder/dict.txt")
        self.dict_labeltype_txt = self.output_path(f"dict_folder/dict.{label_type}.txt")

        self.rqmt = {"time": 3000, "cpu": 1, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        logging.info(f"Running command: {sh_call}") 
        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        # 1. Create raw lexicon
        sh_call = f"paste {self.words_txt.get_path()} {self.tokenized_words.get_path()} > {self.lexicon_lst.get_path()}"
        self.sh_call_with_environment(sh_call)

        # 2. Generate dictionary from phones
        # Note: preprocess.py writes dict.txt to the destdir
        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref {self.tokenized_words.get_path()} --only-source --destdir {self.dict_folder.get_path()} --thresholdsrc {str(self.each_label_count_threshold)} --padding-factor 1 --dict-only"
        self.sh_call_with_environment(sh_call)

        # 3. Filter lexicon based on the generated dictionary
        sh_call = f"python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d {self.dict_txt.get_path()} < {self.lexicon_lst.get_path()} > {self.lexicon_filtered_lst.get_path()}"
        self.sh_call_with_environment(sh_call)
        

        # cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
        # echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
        # Create final dictionary with silence token
        sh_call = f"cp {self.dict_txt.get_path()} {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

        sh_call = f"echo \"{self.silence_token} 0\" >> {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v
            if k in cls.__sis_hash_constants__:
                d[k] = cls.__sis_hash_constants__[k]

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))

class TokenizeWithLexiconAndSilenceJob(Job):
    """
    Job to phonemize text using a lexicon and optionally insert silence tokens.
    """

    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        text_file_path: Type[tk.Path],
        lexicon_lst: Type[tk.Path],
        insert_silence_between_words_prob: float = 0.25,
        silence_token: str = "<SIL>",  # only <SIL> is supported currently
        surround_with_silence: bool = True,
    ):
        assert silence_token == "<SIL>", "Currently, only '<SIL>' is supported as silence token."
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.text_file_path = text_file_path
        self.lexicon_lst = lexicon_lst
        self.insert_silence_between_words_prob = insert_silence_between_words_prob
        self.silence_token = silence_token
        self.surround_with_silence = surround_with_silence

        # Outputs
        self.tokenized_text_with_silence = self.output_path("tokenized_text_with_silence.txt")

        self.rqmt = {"time": 3000, "cpu": 1, "mem": 120}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        wrd_to_phn = {}
        lexicon_path = self.lexicon_lst.get_path()
        lines = open(self.text_file_path.get_path(), "r").readlines()
        # Load lexicon
        with open(lexicon_path, "r") as lf:
            for line in lf:
                items = line.rstrip().split()
                assert len(items) > 1, f"Invalid lexicon line: {line}"
                assert items[0] not in wrd_to_phn, f"Duplicate word: {items[0]}"
                wrd_to_phn[items[0]] = items[1:]

        outputs = []

        # Process input lines
        for line in lines:
            words = line.strip().split()

            # Skip lines with unknown words
            if not all(w in wrd_to_phn for w in words):
                continue

            phones = []
            if self.surround_with_silence:
                phones.append(self.silence_token)

            sample_sil_probs = None
            if self.insert_silence_between_words_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1)

            for i, w in enumerate(words):
                phones.extend(wrd_to_phn[w])
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < self.insert_silence_between_words_prob
                ):
                    phones.append(self.silence_token)

            if self.surround_with_silence:
                phones.append(self.silence_token)

            outputs.append(" ".join(phones))

        with open(self.tokenized_text_with_silence.get_path(), "w") as f:
            for line in outputs:
                f.write(line + "\n")
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))

class FairseqNormalizeTextAndCreateDictionary(Job):

    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        text_file_path: Type[tk.Path],
        language: str, # fr, pt, en
        lid_path: Type[tk.Path], # path to a fasttext model
        thresholdsrc: int = 0,
    ):
        
        self.fairseq_root = fairseq_root
        self.text_file_path = text_file_path
        self.language = language
        self.lid_path = lid_path
        self.fairseq_python_env = fairseq_python_env
        self.thresholdsrc = thresholdsrc

        self.output_dir = self.output_path("normalized_text", directory=True)
        self.words_txt = self.output_path("normalized_text/words.txt")
        self.dict_txt = self.output_path("normalized_text/dict.txt")
        self.normalize_text_lid = self.output_path("normalized_text/lm.upper.lid.txt")
        
        self.rqmt = {"time": 200, "cpu": 1, "mem": 16}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        sh_call = ['python', self.fairseq_root.get_path() + '/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py', '--lang', self.language, '--fasttext-model', self.lid_path.get_path(), '<', self.text_file_path.get_path(), '|', 'grep', '-v', "'\\-\\-\\-'", '>', self.normalize_text_lid.get_path()]
        self.sh_call_with_environment(" ".join(sh_call))
        sh_call = ['python', self.fairseq_root.get_path() + '/fairseq_cli/preprocess.py', '--dataset-impl', 'mmap', '--trainpref', self.normalize_text_lid.get_path(), '--only-source', '--destdir', self.output_dir.get_path(),"--thresholdsrc", str(self.thresholdsrc),  '--padding-factor', '1', '--dict-only']
        self.sh_call_with_environment(" ".join(sh_call))
        sh_call = ['cut', '-f1', "-d'", "'", self.dict_txt.get_path(), '|', 'grep', '-v', '-x', "'[[:punct:]]*'", '|', 'grep', '-Pv', "'\\d\\d\\d\\d\\d+'", '>', self.words_txt.get_path()]
        self.sh_call_with_environment(" ".join(sh_call))
    
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))

class FairseqPreprocessJob(Job):
   
    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}

    def __init__(
        self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        tokenized_text: Type[tk.Path],
        dict_labeltype_txt: Type[tk.Path],
        dataset_impl: str = "mmap",
    ):
        
        self.fairseq_root = fairseq_root
        self.tokenized_text = tokenized_text
        self.dict_labeltype_txt = dict_labeltype_txt
        self.fairseq_python_env = fairseq_python_env
        self.dataset_impl = dataset_impl

        self.labels_folder = self.output_path("labels_folder", directory=True)
        
        self.rqmt = {"time": 4242, "cpu": 1, "mem": 32}
    
    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def sh_call_with_environment(self, sh_call):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fairseq_root.get_path()}:" + env.get("PYTHONPATH", "")
        env["FAIRSEQ_ROOT"] = self.fairseq_root.get_path()
        env["PATH"] = f"{self.fairseq_python_env.get_path()}/bin:" + env["PATH"]

        sp.run(sh_call, env=env, check=True, shell=True)

    def run(self):
        sh_call = f"python {self.fairseq_root.get_path()}/fairseq_cli/preprocess.py --dataset-impl {self.dataset_impl} --trainpref {self.tokenized_text.get_path()} --workers 1 --only-source --destdir {self.labels_folder.get_path()} --srcdict {self.dict_labeltype_txt.get_path()}"
        self.sh_call_with_environment(sh_call)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))


def resolve_tk_paths(obj):
    # If it's a tk.Path object → replace it
    if isinstance(obj, tk.Path):
        return obj.get_path()
    
    # If dictionary → process values
    if isinstance(obj, dict):
        return {k: resolve_tk_paths(v, tk.Path) for k, v in obj.items()}
    
    # If list → process elements
    if isinstance(obj, list):
        return [resolve_tk_paths(x, tk.Path) for x in obj]
    
    # If tuple → return a new tuple
    if isinstance(obj, tuple):
        return tuple(resolve_tk_paths(x, tk.Path) for x in obj)
    
    # If set → return a new set
    if isinstance(obj, set):
        return {resolve_tk_paths(x, tk.Path) for x in obj}
    
    # Otherwise just return unchanged
    return obj

class TrainKenLMJob(Job):

    __sis_hash_exclude__ = {"kenlm_root"}

    def __init__(
        self,
        kenlm_root: Type[tk.Path],
        text_file_path: Type[tk.Path],
        lm_order: int,
        discount_fallback: bool=True,
        pruning: Optional[List[int]] = None,
        build_binary: bool = True,
    ):
        self.kenlm_root = kenlm_root
        self.text_file_path = text_file_path
        self.lm_order = lm_order
        self.discount_fallback = discount_fallback
        self.pruning = pruning
        self.build_binary = build_binary

        if pruning is None:
            self.output_arpa = self.output_path(f"kenlm.o{lm_order}.arpa")
            if build_binary:
                self.output_bin = self.output_path(f"kenlm.o{lm_order}.bin")
        else:
            assert len(pruning) == lm_order, "Pruning list length must match LM order"
            self.output_arpa = self.output_path(f"kenlm.o{lm_order}.pruned_{'_'.join(map(str, pruning))}.arpa")
            if build_binary:
                self.output_bin = self.output_path(f"kenlm.o{lm_order}.pruned_{'_'.join(map(str, pruning))}.bin")

        self.rqmt = {"time": 1000, "cpu": 1, "mem": 64}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def run(self):
        sh_call = f"{self.kenlm_root.get_path()}/lmplz -o {self.lm_order} {('--prune ' + ' '.join(map(str, self.pruning))) if not self.pruning is None else ' '} < {self.text_file_path.get_path()} --discount_fallback > {self.output_arpa.get_path()}"
        sp.run(sh_call, check=True, shell=True)

        if self.build_binary:
            sh_call = f"{self.kenlm_root.get_path()}/build_binary {self.output_arpa.get_path()} {self.output_bin.get_path()}"
            sp.run(sh_call, check=True, shell=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))
        


class CreateKaldiFSTJob(Job):

    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root", "kaldi_root"}

    def __init__(self,
        fairseq_root: Type[tk.Path],
        fairseq_python_env: Type[tk.Path],
        lm_arpa: Type[tk.Path],
        lexicon_lst: Type[tk.Path],
        data_dir: Type[tk.Path],
        label_type: str = "",
        kaldi_root: Type[tk.Path] = None,
        blank_symbol: str = "<SIL>",
        lg: str = "en",
    ):
        self.fairseq_root = fairseq_root
        self.fairseq_python_env = fairseq_python_env
        self.kaldi_root = kaldi_root
        self.lm_arpa = lm_arpa
        self.lexicon_lst = lexicon_lst
        self.data_dir = data_dir
        self.label_type = label_type
        self.blank_symbol = blank_symbol
        self.lg = lg

        self.fst_dir = self.output_path("fst", directory=True)
        self.out_hlg_graph = self.output_path(f"fst/HLG.{label_type}.{osp.splitext(osp.basename(lm_arpa))[0]}.fst")
        self.out_kaldi_dict = self.output_path(f"fst/kaldi_dict.{osp.splitext(osp.basename(lm_arpa))[0]}.txt")
        
        self.rqmt = {"time": 2000, "cpu": 1, "mem": 150}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
    
    def run(self):
        sh_call = (
            f"export FAIRSEQ_ROOT={self.fairseq_root.get_path()} "
            f"{(' && export KALDI_ROOT='+self.kaldi_root.get_path()) if self.kaldi_root is not None else ''} "
            f"&& export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH "
            f"&& source {self.fairseq_python_env.get_path()}/bin/activate "
            f"&& lg={self.lg} python "
            f"{self.fairseq_root.get_path()}/examples/speech_recognition/kaldi/kaldi_initializer.py "
            f"kaldi_root=$KALDI_ROOT "
            f"fst_dir={self.fst_dir.get_path()} "
            f"lm_arpa={self.lm_arpa.get_path()} "
            f"wav2letter_lexicon={self.lexicon_lst.get_path()} "
            f"data_dir={self.data_dir.get_path()} "
            f"in_labels={self.label_type} "
            f"\"blank_symbol='{self.blank_symbol}'\""
        )

        sp.run(sh_call, check=True, shell=True)
        
    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__))