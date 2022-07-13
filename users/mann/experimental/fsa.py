__all__ = ['FsaSearchBuilder']

from sisyphus import *

import re, os
import shutil
import recipe.sprint as sprint
import xml.etree.ElementTree as ET

class FsaSearchBuilder(sprint.SprintCommand, Job):

    dependencies = {
        "g": ["lexicon", "lm"],
        "l": ["lexicon"],
        "c": []
    }

    def __init__(self, csp, operations, input_fsas=None,
            allow_non_crossword_transitions=True, exploit_disambiguators=True,
            add_disambiguators=True, semiring="log", ignore_symbols=True,
            allow_nondet=False, mem=32, **extra_kwargs
        ):
        input_fsas = input_fsas or {}

        kwargs = locals()
        del kwargs['self']

        self.csp        = csp
        self.operations = operations

        self.config, self.post_config = self.create_config(**kwargs)

        self.log_file = self.log_file_output_path("build", csp, parallel=False)
        self.outputs = self.parse_outputs(operations)
        for output in self.outputs:
            setattr(self, output, self.output_path(f"{output}.fsa"))
        self.exe = self.select_exe(csp.fsa_search_builder_exe, 'fsa-search-builder')
        self.rqmt = { 'time': .5, 'cpu': 1, 'gpu': 0, 'mem': mem} #, 'engine': 'long' }

    
    def tasks(self):
        yield Task('create_files', mini_task=True)
        yield Task('run'         , resume='run', rqmt=self.rqmt)
    
    def create_files(self):
        self.write_config(self.config, self.post_config, "fsa-builder.config")
        self.write_run_script(self.exe, "fsa-builder.config", operations=self.operations)
    
    def run(self):
        self.run_script(1, self.log_file)
        for output in self.outputs:
            shutil.move(f"{output}.fsa", getattr(self, output).get_path())
    
    @classmethod
    def parse_outputs(cls, operations: str) -> list:
        prog = re.compile("write-fsa,write-(\w+)")
        matches = prog.findall(operations)

        prog = re.compile("write-fsa( |$)")
        out_match = prog.search(operations)
        if out_match is not None:
            matches.append("out")
        return matches
    
    @classmethod
    def parse_inputs(cls, operations: str) -> list:
        prog = re.compile("read-fsa,read-(\w+)")
        matches = prog.findall(operations)

        prog = re.compile("read-fsa( |$)")
        in_match = prog.search(operations)
        if in_match is not None:
            matches.append("in")
        return matches
    
    @classmethod
    def create_config(cls, csp, operations, input_fsas,
            allow_non_crossword_transitions, exploit_disambiguators,
            add_disambiguators, semiring, ignore_symbols,
            allow_nondet, **kwargs
        ):
        config, post_config = sprint.build_config_from_mapping(
            csp,
            {
                'lexicon'       : 'fsa-search-builder.model-combination.lexicon',
                'acoustic_model': 'fsa-search-builder.model-combination.acoustic-model',
                'language_model': 'fsa-search-builder.model-combination.lm'
            },
        )
        config.fsa_search_builder.build_l.lexicon_builder.add_disambiguators = add_disambiguators
        config.fsa_search_builder.build_c.context_builder.allow_non_crossword_transitions = allow_non_crossword_transitions
        config.fsa_search_builder.build_c.context_builder.exploit_disambiguators          = exploit_disambiguators

        config.fsa_search_builder.minl.semiring = semiring

        config['*'].compose.ignore_symbols = ignore_symbols

        if allow_nondet:
            config['*'].allow_nondet = True

        # add outputs
        for output in cls.parse_outputs(operations):
            prop = "write-fsa" if output == "out" else f"write-{output}"
            config.fsa_search_builder[prop].filename = f"{output}.fsa"

        # add inputs
        for input_name in cls.parse_inputs(operations):
            prop = "read-fsa" if input_name == "in" else f"read-{input_name}"
            config.fsa_search_builder[f"read-{input_name}"].filename = input_fsas[input_name]

        # add extra args
        config.fsa_search_builder = sprint.ConfigBuilder({})(**kwargs.pop("extra_args", {}))
        return config, post_config
    
    @staticmethod
    def write_run_script(exe, config, filename='run.sh', extra_code='', operations=''):
        """
        :param str exe:
        :param str config:
        :param str filename:
        :param str extra_code:
        """
        with open(filename, 'wt') as f:
            f.write("""\
#!/usr/bin/env bash
set -ueo pipefail

if [[ $# -gt 0 ]]; then
    TASK=$1;
    shift;
else
    echo "No TASK-id given";
    exit 1;
fi

if [ $# -gt 0 ]; then
    LOGFILE=$1;
    shift;
else
    LOGFILE=sprint.log
fi

%s

%s \\
    --config=%s \\
    --*.LOGFILE=$LOGFILE \\
    --operations="%s"
""" % (extra_code, exe, config, operations))
            os.chmod(filename, 0o755)

    @classmethod
    def hash(cls, kwargs):
        config, _ = cls.create_config(**kwargs)
        return super().hash(
            {
                'config': config,
                'operations': kwargs["operations"]
            }
        )


class NotFoundError(Exception):
    def __init__(self, *args, **kwargs):
        super(NotFoundError, self).__init__(*args, **kwargs)

def comfy_element(tag, attrib=None, text=None, **extra):
    if attrib is None: attrib = {}
    res = ET.Element(tag, attrib, **extra)
    if text is not None:
        res.text = str(text)
    return res

def element_with_children(tag, *children, attrib=None, text=None, **extra):
    res = comfy_element(tag, attrib, text, **extra)
    for child in children:
        res.append(child)
    return res

def find_symbol(token, element, substring=False):
    ftest = lambda x, y: x == y if not substring else x in y
    for child in element:
        if ftest(token, child.text):
            return child
            break
    else:
        return None

def find_silence_loop(states):
    for state in states:
        if state.attrib["id"] == "3636":
            print(state[1])
            for prop in state[1]:
                print(prop.tag, prop.text)
        if len(state) == 2 and any(prop.text == "207" and prop.tag == "in" for prop in state[1]):
            return state
    raise NotFoundError("Silence loop state not found")

def find_silence_loop(states, silence_id):
    for state in states:
        arcs = [el for el in state if el.tag == 'arc']
        if len(arcs) == 1 and arcs[0].attrib['target'] == state.attrib['id'] and arcs[0][0].text == str(silence_id):
            return state
    raise NotFoundError("Silence loop state not found")


from functools import wraps

def camelize(string):
    subwords = string.split("_")
    return "".join(map(lambda s: s.capitalize(), subwords))

def simple_job(output_name="out"):
    def decorator(func):
        name = camelize(func.__name__)
        class SimpleJob(Job):
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.out = self.output_path(output_name)

            def tasks(self):
                yield Task("run", mini_task=True)

            def run(self):
                func(self.out, *self.args, **self.kwargs)

        # lift to top level for pickling
        SimpleJob.__name__ = name
        SimpleJob.__qualname__ = name
        globals()[name] = SimpleJob
        @wraps(func)
        def job_caller(*args, **kwargs):
            j = SimpleJob(*args, **kwargs)
            return j.out
        return job_caller
    return decorator


@simple_job(output_name="fix")
def fix_loops(output, input_fsa, silence_id, loop_entrance_weight=0.0, loop_exit_weight=0.0):
    tree = ET.parse(input_fsa.get_path())
    _, *states = tree.getroot()
    loop_state = find_silence_loop(states, silence_id)
    loop_state_id = loop_state.attrib['id']
    states[0].append(
        element_with_children(
            "arc",
            comfy_element("in", text=silence_id),
            comfy_element("weight", text=loop_entrance_weight),
            target=loop_state_id
        )
    )
    loop_state.append(
        element_with_children(
            "arc",
            comfy_element("in", text=silence_id),
            comfy_element("weight", text=loop_exit_weight),
            target="0",
        )
    )
    tree.write(output.get_path())

@simple_job(output_name="mod.fsa")
def insert_silence(
        output,
        input_fsa,
        silence_token = "[SILENCE]",
        silence_initial_token = "[SILENCE]@i",
        ):
    tree = ET.parse(input_fsa.get_path())
    input_alphabet, output_alphabet, *states = tree.getroot()

    silence_input_id  = find_symbol(silence_initial_token, input_alphabet).attrib['index']
    silence_output_id = find_symbol(silence_token, output_alphabet, substring=True).attrib['index']

    for state in states:
        target_id = state.attrib['id']
        state.append(
            element_with_children(
                "arc",
                comfy_element("in", text=silence_input_id),
                comfy_element("out", text=silence_output_id),
                target=target_id,
            )
        )
    tree.write(output.get_path())


class FixLoopConnection(Job):
    def __init__(self, input_fsa, silence_id, loop_entrance_weight=0.0, loop_exit_weight=0.0):
        self.input_fsa = input_fsa
        self.silence_id = str(silence_id)
        self.weights = {
            "enter": str(loop_entrance_weight),
            "exit": str(loop_exit_weight)
        }
        self.out = self.output_path("out.fsa")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    @staticmethod
    def find_silence_loop(states):
        for state in states:
            if state.attrib["id"] == "3636":
                print(state[1])
                for prop in state[1]:
                    print(prop.tag, prop.text)
            if len(state) == 2 and any(prop.text == "207" and prop.tag == "in" for prop in state[1]):
                return state
        raise NotFoundError("Silence loop state not found")
    
    def run(self):
        tree = ET.parse(self.input_fsa.get_path())
        _, *states = tree.getroot()
        loop_state = FixLoopConnection.find_silence_loop(states)
        loop_state_id = loop_state.attrib['id']
        states[0].append(
            element_with_children(
                "arc",
                comfy_element("in", text=self.silence_id),
                comfy_element("weight", text=self.weights["enter"]),
                target=loop_state_id
            )
        )
        loop_state.append(
            element_with_children(
                "arc",
                comfy_element("in", text=self.silence_id),
                comfy_element("weight", text=self.weights["exit"]),
                target="0",
            )
        )
        tree.write(self.out.get_path())
