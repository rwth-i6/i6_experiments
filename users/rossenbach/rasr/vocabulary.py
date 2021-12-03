from sisyphus import Job, Task

from i6_core.util import uopen


class GenerateLabelFileFromStateTying(Job):

    def __init__(self, state_tying_file, add_eow=False, add_sow=False):
        """
        
        :param state_tying_file: 
        """
        self.state_tying_file = state_tying_file
        self.add_eow = add_eow
        self.add_sow = add_sow

        self.out_label_file = self.output_path("label_vocab_dict")


    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):

        state_tying_dict = {}
        with uopen(self.state_tying_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                tokens = line.split()
                assert len(tokens) >= 2, line
                state_tying_dict[tokens[0]] = ' '.join(tokens[1:])

        vocab_dict = {}
        max_id = 0
        for allo, index in state_tying_dict.items():
            phon = allo.split('{')[0]
            if not phon.startswith('[') and not phon.endswith(']'):
                # only eow for single-phon pronunciation, i.e. @i@f
                if self.add_eow and ('@f' in allo):
                    phon += '#'
                elif self.add_sow and ('@i' in allo):
                    phon = '#' + phon
            if phon in vocab_dict:
                assert vocab_dict[phon] == index, "index conflict for %s: %s vs. %s (%s %s)" %(phon, index, vocab_dict[phon], allo, index)
            else:
                vocab_dict[phon] = index
            if int(index) > max_id:
                max_id = int(index)
                
        with uopen(self.out_label_file, 'w') as f:
            for v in sorted(vocab_dict.keys()):
                f.write(v + ' '+vocab_dict[v] + '\n')
                
        print('number of classes:', len(set(vocab_dict.values())))
        assert len(set(vocab_dict.values())) == max_id+1, "expected number of classes %d" %(max_id+1)








