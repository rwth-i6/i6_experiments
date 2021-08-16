from sisyphus import Job, Path


class ApplySequiturWithDict(Job):
    """
    Applies a sequitur model and a phoneme dictionary to a text-format corpus file.
    """

    def __init__(self, text_corpus, phon_dict, sequitur_model,
                 tts_toolchain_root=None, tts_toolchain_python_exe=None):
        """

        :param text_corpus: text file corpus
        :param phon_dict: CMU format phoneme dictionary
        :param sequitur_model: sequitur model file
        :param tts_toolchain_root: path to the tts_toolchain root directory
        """

        self.text_corpus_in = text_corpus
        self.phon_dict = phon_dict
        self.sequitur_model = sequitur_model
        self.tts_toolchain_root = tts_toolchain_root or gs.TTS_TOOLCHAIN_ROOT
        self.tts_toolchain_python_exe = tts_toolchain_python_exe or gs.TTS_TOOLCHAIN_PYTHON_EXE

        self.out = self.output_path("phoneme_text_corpus")

        self.rqmt = {'cpu': 1, 'mem': 8, 'time': 2}

    def tasks(self):
        yield Task('run', rqmt=self.rqmt)

    def run(self):
        env = os.environ
        env['_'] = tk.uncached_path(self.tts_toolchain_python_exe)
        call = [tk.uncached_path(self.tts_toolchain_python_exe),
                os.path.join(tk.uncached_path(self.tts_toolchain_root), "apply_sequitur.py"),
                tk.uncached_path(self.phon_dict),
                tk.uncached_path(self.sequitur_model),
                tk.uncached_path(self.text_corpus_in),
                tk.uncached_path(self.out)]
        subprocess.check_output(call, env=env)