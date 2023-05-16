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

    def __init__(
        self,
        checkpoint,
        # all_tensors=True,
        tensor_name=None,
        print_options=None,
        returnn_python_exe=None,
        returnn_root=None,
        assert_tensors_parsed=None,
        gpu=0,
    ):
        # assert all_tensors == (tensor_name is None)
        self.tensors_parsed = assert_tensors_parsed or []
        self.returnn_python_exe   = returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE
        self.returnn_root         = returnn_root       if returnn_root       is not None else gs.RETURNN_ROOT
        self.checkpoint = checkpoint
        self.all_tensors = False
        self.tensor_name = tensor_name
        self.print_options = print_options
        self.tensors_raw = self.output_path("tensors.txt")
        # self.tensors = self.output_var("tensors", pickle=True)
        self.out_tensor_file = self.output_path("var.txt")
        self.rqmts = {"gpu": 0, "time": 0.1}
    
    def tasks(self):
        mini_task = self.rqmts["gpu"] > 0
        yield Task('run', mini_task=mini_task, resume='run', rqmt=self.rqmts)
        yield Task('extract', resume='run', mini_task=True)
    
    def run(self):
        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), 'tools/tf_inspect_checkpoint.py'),
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
        string = string.strip().replace("\n", " ")
        string = re.sub("\[\s+", "[", string)
        string = re.sub("\s+\]", "]", string)
        parsed_string = re.sub("\s+", ",", string)
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
            copy = False
            for line in f:
                if line.startswith("tensor_name:"):
                    copy = True
                    continue
                elif line.startswith("mean:"):
                    break
                elif copy:
                    buffer += line
            value = InspectTFCheckpointJob.read_array(buffer)
            tensors[prev_tensor_name] = InspectTFCheckpointJob.read_array(buffer)
        assert all(tensor_name in tensors and tensors[tensor_name] is not None for tensor_name in self.tensors_parsed)
        np.savetxt(self.out_tensor_file.get_path(), value)
