from sisyphus import *

import os, re
import numpy as np
import subprocess as sp

class InspectTFCheckpointJob(Job):
    """
    Job that wraps returnn tf_inspect_checkpoint.py tool. Additionally parses the output and collects the
    successfully parsed tensors in a dict that is accessible as a sisyphus "Value" object.
    """

    __sis_hash_exclude__ = {"assert_tensors_parsed": None}

    def __init__(self, checkpoint, all_tensors=True, tensor_name=None, print_options=None, crnn_python_exe=None, crnn_root=None, assert_tensors_parsed=None):
        assert all_tensors == (tensor_name is None)
        self.tensors_parsed = assert_tensors_parsed or []
        self.crnn_python_exe   = crnn_python_exe if crnn_python_exe is not None else gs.CRNN_PYTHON_EXE
        self.crnn_root         = crnn_root       if crnn_root       is not None else gs.CRNN_ROOT
        self.checkpoint = checkpoint
        self.all_tensors = all_tensors
        self.tensor_name = tensor_name
        self.print_options = print_options
        self.tensors_raw = self.output_path("tensors.txt")
        self.tensors = self.output_var("tensors", pickle=True)
        self.rqmts = {"gpu": 1, "time": 0.1}
    
    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmts)
        yield Task('extract', mini_task=True)
    
    def run(self):
        args = [
            tk.uncached_path(self.crnn_python_exe),
            os.path.join(tk.uncached_path(self.crnn_root), 'tools/tf_inspect_checkpoint.py'),
            f'--file_name={self.checkpoint}'
        ]
        if self.all_tensors:
            args += ['--all_tensors']
        if self.tensor_name:
            args += [f'--tensor_name={self.tensor_name}']
        if self.print_options:
            args += [f'--print_options={self.print_options}']
        p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
        output_raw, err = p.communicate()
        output = output_raw.decode("utf-8")
        print(f"{output}")
        with open(self.tensors_raw.get_path(), "w") as f:
            f.write(output)

    @staticmethod
    def read_array(string):
        string = re.sub("\[\s+", "[", string.rstrip("\n").replace("\n", ","))
        string = re.sub("\s+\]", "]", string)
        # string = re.sub("\[\s+(.+)\s+\]", "[\1]", string.rstrip("\n").replace("\n", ","))
        parsed_string = re.sub("\s+", ",", string)
        parsed_string
        try:
            return np.array(
                eval(parsed_string)
            )
        except:
            print("Eval call failed on '{}'".format(parsed_string))
            return None
    
    def extract(self):
        prev_tensor_name = None
        buffer = ""
        tensors = {}
        with open(self.tensors_raw.get_path(), "r") as f:
            for line in f:
                if line.startswith("tensor_name:"):
                    if prev_tensor_name is not None:
                        tensors[prev_tensor_name] = InspectTFCheckpointJob.read_array(buffer)
                    prev_tensor_name = line.split(" ")[-1].rstrip("\n")
                    buffer = ""
                    continue
                buffer += line
            value = InspectTFCheckpointJob.read_array(buffer)
            tensors[prev_tensor_name] = InspectTFCheckpointJob.read_array(buffer)
        assert all(tensor_name in tensors and tensors[tensor_name] is not None for tensor_name in self.tensors_parsed)
        self.tensors.set(tensors)
            