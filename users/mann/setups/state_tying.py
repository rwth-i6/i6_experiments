from sisyphus import *

import re
from abc import ABC, abstractmethod
from collections import namedtuple

ALLO_PROG = re.compile("(.+)\{(.+)\+(.+)\}(.*).([0-2]) (\d+)")

BaseAllophone = namedtuple(
    "Allphone",
    "phon prev next initial final state idx",
    defaults=[False, False, None, None]
)

class Allophone(BaseAllophone):
    """Represents a RASR allophone an implements
    some base functionality."""
    __slots__ = ()

    @classmethod
    def parse_line(cls, line: str):
        m = ALLO_PROG.match(line)
        phon, prev, nxt, flags, state, idx = m.groups()
        # return cls._make(m.groups())
        return cls(phon, prev, nxt, "@i" in flags, "@f" in flags, int(state), int(idx))
    
    @classmethod
    def from_phones(cls, phones):
        return cls(*phones)
    
    def write_phon(self):
        return "{}{{{}+{}}}{}".format(
            self.phon, self.prev, self.next, self.flags
        )
    
    def write(self):
        return "{}{{{}+{}}}{}.{} {}".format(
            self.phon, self.prev, self.next, self.flags, self.state, self.idx
        )

    @property
    def flags(self):
        flags = ""
        if self.initial:
            flags += "@i"
        if self.final:
            flags += "@f"
        return flags


def parse_line(line: str):
    m = ALLO_PROG.match(line)
    return Allophone._make(m.groups())

def write_allophone(allophone: Allophone):
    return "{}{{{}+{}}}{}.{} {}".format(*allophone)

class AbstractMakeStateTying(Job):
    def __init__(self, state_tying_file, num_states, non_phon_tokens=None):
        self.exceptions = non_phon_tokens or []
        self.num_states = num_states
        self.input_state_tying_file = state_tying_file
        self.state_tying = self.output_path("state-tying")
    
    def tasks(self):
        yield Task("run", mini_task=True)

    def is_exception(self, phon):
        return any(exc in phon for exc in self.exceptions)
    
    def run(self):
        raise NotImplementedError


class MakeEOWStateTying(Job):
    def __init__(self, state_tying_file, num_states, non_phon_tokens=None):
        self.exceptions = non_phon_tokens or []
        self.num_states = num_states
        self.input_state_tying_file = state_tying_file
        self.state_tying = self.output_path("state-tying")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        # incr_id = lambda si: str(int(si) + self.num_states)
        # incr_second = lambda t: (t[0], incr_id(t[1]))
        def transform(line):
            line = line.rstrip("\n")
            phon, idx = line.split(" ")
            if any(exc in phon for exc in self.exceptions) \
                or "@f" not in phon:
                return line
            new_idx = str(int(idx) + self.num_states)
            return " ".join([phon, new_idx])
        with open(self.state_tying.get(), "w") as fout:
            phone_and_id = [
                transform(line)
                for line in open(tk.uncached_path(self.input_state_tying_file), "r")
            ]
            
            fout.write("\n".join(phone_and_id))


class MakeDiphoneStateTying(AbstractMakeStateTying):
    def __init__(self, state_tying_file, num_states, hmm_partition, non_phon_tokens=None):
        super().__init__(state_tying_file, num_states, non_phon_tokens)
        self.hmm_partition = hmm_partition

    def extract_phons(self, allophones):
        phons = set()
        for allophone in allophones:
            phon = allophone.phon
            if any(exc in phon for exc in self.exceptions):
                continue
            phons.add(phon.split("{")[0])
        return list(phons)

    def run(self):
        print("Run")
        from operator import itemgetter as fget
        from itertools import tee
        import re
        allophones = map(
            parse_line,
            open(tk.uncached_path(self.input_state_tying_file), "r")
        )
        # all_phones = list(set(map(
        #         lambda phon: phon.split("{")[0],
        #         map(fget(0), phon_idxs)
        # )))
        first_run, second_run = tee(allophones)
        all_phons = self.extract_phons(first_run)
        all_ctx_tokens = all_phons + ["#"]
        num_phons = len(all_phons)
        num_ctxs = num_phons + 1
        exc_offset = num_phons * self.hmm_partition 
        def get_new_idx(allophone):
            phon, prev = allophone[:2]
            if self.is_exception(phon):
                idx = allophone.idx
                return int(idx) - exc_offset + num_phons * num_ctxs * self.hmm_partition
            hmm_id = int(allophone.state)
            return all_phons.index(phon) * num_ctxs * self.hmm_partition \
                + all_ctx_tokens.index(prev) * self.hmm_partition \
                + hmm_id
        with open(self.state_tying.get(), "w") as fout:
            fout.write(
                "\n".join(
                    write_allophone(allo._replace(idx=get_new_idx(allo)))
                    for allo in second_run
                )
            )



            


