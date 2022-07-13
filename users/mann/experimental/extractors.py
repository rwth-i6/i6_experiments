__all__ = ['ExtractHost']

from sisyphus import *

import os
import re
import gzip
import json

import tabulate as tab
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict
from itertools import chain

from sisyphus import *
packagePath = setup_path(__package__)

import i6_core.lib.corpus as corpus_lib
import i6_core.lib.lexicon as lexicon
import i6_core.util as util
from i6_core.corpus.filter import FilterSegmentsByListJob

class ExtractHost(Job):

    def __init__(self, data, log_file_name='log.run.1', 
            sclite_file_name='sclite.dtl'):
        self.data             = data
        self.log_file_name    = log_file_name
        self.sclite_file_name = sclite_file_name
        self.headers          = ['Host', 'WERs']

        self.summary = self.output_path("summary.txt")

    def tasks(self):
        yield Task('run', resume='run', mini_task=True)

    def run(self):
        d = defaultdict(list)
        for work_dir, report_dir in self.data:
            log_file = work_dir + self.log_file_name
            sclite_file = os.path.join(tk.uncached_path(report_dir), self.sclite_file_name)
            d[self.host(log_file)].append(self.wer(sclite_file))
        
        rows = [[k, v.join(', ')] for k, v in d.items()]
        with open(self.summary.get_path(), 'wt') as f:
            f.write('Hi') #tabulate.tabulate(rows, headers=self.headers))
    
    @staticmethod
    def host(path):
        with open(path, 'rt') as f:
            for line in f:
                if "Execution host" in line:
                    return line.split(' ')[-1]
            return None

    @staticmethod
    def wer(path):
        regex = re.compile('^Percent Total Error *= *(\\d+\.\\d%).*')
        with open(path, 'rt') as f:
            for line in f:
                m = regex.match(line)
                if m is not None:
                    return m.group(1)
            return None


# get segment list with less then N characters
class FilterSegmentsByTranscriptionLength(Job):
  def __init__(self, corpus_path, max_length=50):
    self.corpus_path = corpus_path
    self.max_length  = max_length

    self.output_path = self.output_path('filtered.segments')

  def tasks(self):
    yield Task('run', resume='run', mini_task=True)

  def run(self):
    c = corpus_lib.Corpus()
    c.load(tk.uncached_path(self.corpus_path))

    filtered_segments = [segment.fullname() for segment in c.segments() if len(segment.orth) < self.max_length]

    with open(tk.uncached_path(self.output_path), 'w') as segment_file:
      segment_file.write('\n'.join(filtered_segments))


class GenericFilterCorpusJob(Job):
  def __init__(self, corpus_path: tk.Path, filter_expression: str):
    assert isinstance(filter_expression, str) and callable(eval(filter_expression)), \
        "Filter expression is not a string or does not evaluate to a callable object."

    self.corpus_path = corpus_path
    self.filter_expression = filter_expression

    self.segments = self.output_path('filtered.segments')

  def tasks(self):
    yield Task('run', resume='run', mini_task=True)

  def run(self):
    c = corpus_lib.Corpus()
    c.load(tk.uncached_path(self.corpus_path))
    fltr = eval(self.filter_expression)

    filtered_segments = filter(fltr, c.segments())

    with open(tk.uncached_path(self.segments), 'w') as segment_file:
      segment_file.write('\n'.join(filtered_segments))



class SegmentStatistics(Job):
  def __init__(self, corpus_paths, partition=[50, 100, 150, 200, 250, 300, 350]):
    self.corpus_paths = corpus_paths
    self.partition = partition

    self.plot_path = self.output_path('hist.plot.png')

  def tasks(self):
    yield Task('run', resume='run', mini_task=True)

  def run(self):
    lengths, labels = [], []
    for c, p in self.corpus_paths.items():
        corp = corpus_lib.Corpus()
        corp.load(tk.uncached_path(p))

        labels += [c] 
        lengths += [[len(segment.orth) for segment in corp.segments()]]
    # plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.hist(lengths, bins=self.partition, label=labels)
    ax.set_title('Segment Length Histogramm')
    ax.legend()
    ax.set_xlabel('Segment orth length')
    ax.set_ylabel('Count')
    plt.savefig(self.plot_path.get_path())

class FilterSegmentsByAlignmentFailures(FilterSegmentsByListJob):
    def __init__(self, single_segment_files, alignment_logs,
            # FilterSegmentByList arguments
            filter_list=None, invert_match=False):
        super().__init__(single_segment_files, filter_list, invert_match)
        assert isinstance(alignment_logs, (list, dict))
        if isinstance(alignment_logs, dict):
            alignment_logs = list(alignment_logs.values())
        # new inputs
        self.alignment_logs = alignment_logs
        # make filter list an output
        self.filter_list = self.output_path('bad.segments')

    def tasks(self):
        yield Task('bad', resume='bad', mini_task=True)
        yield Task('run', resume='run', mini_task=True)

    def bad(self):
        prog = re.compile("Alignment did not reach any final state in segment \\'(.+)\\'")
        segments = []
        for log in self.alignment_logs:
            log_path = log.get_path() if isinstance(log, Path) else log
            with gzip.open(log_path, 'rt') as log_file:
                for line in log_file:
                    m = prog.search(line)
                    segments += [m.group(1)] if m else []
        with open(self.filter_list.get_path(), 'w') as bad_segment_file:
            bad_segment_file.write('\n'.join(segments))


class LexiconToUpperCase(Job):
    def __init__(self, lexicon):
        self.in_lexicon = lexicon
        # self.transform = self.f if transform is None else transform 
        self.out_lexicon = self.output_path("out.lexicon.gz")
    
    def tasks(self):
        yield Task('run', resume='run', mini_task=True)

    def run(self):
        transform = lambda x: x.upper()
        lex = lexicon.Lexicon()
        lex.load(self.in_lexicon.get_path())
        out_lex = lexicon.Lexicon()
        for phon, var in lex.phonemes.items():
            out_lex.add_phoneme(transform(phon), var)
        for lemma in lex.lemma:
            lemma.orth = list(map(transform, lemma.orth))
            lemma.phon = list(map(transform, lemma.phon))
            out_lex.add_lemma(lemma)
        root = out_lex.to_xml()
        with util.uopen(self.out_lexicon, 'wt', encoding='utf-8') as f:
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            f.write(xmlstr)


class TransformLexicon(Job):
    def __init__(self, lexicon: lexicon.Lexicon, transform: str):
        assert isinstance(transform, str) and callable(eval(transform)), \
            "Transform is not a string or does not evaluate to a callable object."
        self.in_lexicon = lexicon
        self.transform_string = transform
        # self.transform = self.f if transform is None else transform 
        self.out_lexicon = self.output_path("out.lexicon.gz")
    
    def tasks(self):
        yield Task('run', resume='run', mini_task=True)

    def run(self):
        transform = eval(self.transform_string)
        transform_lex = lambda s: " ".join(map(transform, s.split(" ")))
        lex = lexicon.Lexicon()
        lex.load(self.in_lexicon.get_path())
        out_lex = lexicon.Lexicon()
        for phon, var in lex.phonemes.items():
            out_lex.add_phoneme(transform(phon), var)
        for lemma in lex.lemma:
            # lemma.orth = list(map(transform, lemma.orth))
            phon = []
            for p in lemma.phon:
                p.content = transform_lex(p.content)
                if all(ph.content != p.content for ph in phon):
                    phon.append(p)
            # lemma.phon = list(map(transform_lex, lemma.phon))
            lemma.phon = phon

            out_lex.add_lemma(lemma)
        root = out_lex.to_xml()
        with util.uopen(self.out_lexicon, 'wt', encoding='utf-8') as f:
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            f.write(xmlstr)


class FilterCorpusByAlignmentFailures(Job):
    def __init__(self, corpus_path, alignment_logs):
        # inputs
        self.corpus_path    = corpus_path
        self.alignment_logs = alignment_logs

        # outputs
        self.bad_segments      = self.output_path('bad.segments')
        # self.filtered_segments = self.output_path('filtered.segments')
        self.out_corpus        = self.output_path('corpus.xml')

    def tasks(self):
        yield Task('bad', resume='bad', mini_task=True)
        yield Task('run', resume='run', mini_task=True)

    def run(self):
        c = corpus_lib.Corpus()
        c.load(tk.uncached_path(self.corpus_path))

        bad_segment_list = [line.rstrip() for line in open(self.bad_segments.get_path(), 'r')]
        for rec in c.all_recordings():
            rec.segments = [segment.fullname() for segment in rec.segments 
                            if segment.fullname() not in bad_segment_list]
        
        c.dump(tk.uncached_path(self.out_corpus))

        """
        with open(tk.uncached_path(self.filtered_segments), 'w') as segment_file:
            segment_file.write('\n'.join(segments))
        """
    
    def bad(self):
        prog = re.compile("Alignment did not reach any final state in segment \\'(.+)\\'")
        segments = []
        for log in self.alignment_logs:
            log_path = log.get_path() if isinstance(log, Path) else log
            with gzip.open(log_path, 'rt') as log_file:
                for line in log_file:
                    m = prog.search(line)
                    segments += [m.group(1)] if m else []
        with open(self.bad_segments.get_path(), 'w') as bad_segment_file:
            bad_segment_file.write('\n'.join(segments))
    

class LearningRates(Job):
    def __init__(self, learning_rate_file):
        self.learning_rate_file = learning_rate_file
        self.learning_rates = self.output_var('learning_rates')
    
    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        def EpochData(learningRate, error):
            return {'learning_rate': learningRate, 'error': error}

        with open(self.learning_rate_file.get_path(), 'r') as lr_file:
            data = eval(lr_file.read())

        epochs = list(sorted(data.keys()))
        self.learning_rates.set([data[epoch]['learning_rate'] for epoch in epochs])

class ExtractStateTyingStats(Job):
    def __init__(self, state_tying_file):
        self.state_tying_file = state_tying_file
        self.num_states = self.output_var("num_states")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        with open(tk.uncached_path(self.state_tying_file), "r") as f:
            lines = f.readlines()
            self.num_states.set(
                max(map(lambda l: int(l.rstrip("\n").split(" ")[-1]), lines)) + 1
            )

class AlignmentWrapper:
    def __init__(self, alignment_bundle):
        import recipe.sprint as sprint
        if isinstance(alignment_bundle, sprint.FlagDependentFlowAttribute):
            alignment_bundle = alignment_bundle.alternatives['bundle']
        self.single_alignment_caches = {
            task_id: Path(alignment_cache.rstrip('\n'))
            for task_id, alignment_cache in enumerate(open(alignment_bundle.get_path(), 'r').readlines())
        }


class TpdSummary(Job):
    def __init__(self, tdps):
        self.tdps    = tdps
        self.summary = self.output_path('tdp.summary')

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        tdp_bundle = {
            key: json.load(open(tdp_file.get_path(), 'r'))
            for key, tdp_file in self.tdps.items()
        }
        transitions = ['loop', 'forward', 'skip']
        transitions = [
            '.'.join([s, transition]) 
            for s in ['*', 'sil'] for transition in transitions
        ]
        headers = ['alignment'] + transitions
        data = [
            [key] + [trans_dict[s][str(i)] for s in ['phon', 'si'] for i in range(3)]
            for key, trans_dict in tdp_bundle.items()
        ]
        with open(self.summary.get_path(), 'w') as out:
            out.write(tab.tabulate(data, headers=headers, tablefmt='presto'))


class AlignmentScore(Job):
    def __init__(self, alignment_logs, concurrent):
        self.alignment_logs = alignment_logs
        self.concurrent = concurrent
        self.signature = ['segment_count', 'avg_score', 'score', 'frame_count']
        self.stats = {i: self.output_var('alignment.score.{}'.format(i)) for i in range(1, self.concurrent+1)}
        self.total_score = self.output_var('alignment.score.total')
    
    def tasks(self):
        yield Task('run', mini_task=True, args=range(1, self.concurrent+1))
        yield Task('sum', mini_task=True)
    
    def run(self, task_id):
        alignment_log_file = self.alignment_logs[task_id].get_path()
        stats = {'segment_count': 0, 'avg_score': 0, 'score': 0, 'frame_count': 0}
        tree = ET.parse(gzip.open(alignment_log_file, 'rb'))
        root = tree.getroot()
        for segment in root.iter('segment'):
            al_stats = segment.find('alignment-statistics')
            stats['segment_count'] += 1
            stats['frame_count'] += int(al_stats.find('frames').text)
            score = al_stats.find('score')
            stats['score'] += float(score.find('total').text)
            stats['avg_score'] += float(score.find('avg').text)
        self.stats[task_id].set(stats)
    
    def sum(self):
        stats = {'segment_count': 0, 'avg_score': 0, 'score': 0, 'frame_count': 0}
        for i in range(1, self.concurrent+1):
            cstats = self.stats[i].get()
            for s in self.signature:
                stats[s] += cstats[s]
        avg_stats = {
            'segment': stats['avg_score'] / stats['segment_count'],
            'frame': stats['score'] / stats['frame_count']
        }
        self.total_score.set(avg_stats)


class SummaryJob(Job):
    def __init__(self, value_bundle, header, columns=None, variable_key=None, latex=False):
        assert all(isinstance(value, (tk.Variable, dict)) for value in value_bundle.values())
        self.value_bundle = value_bundle
        self.header = header
        self.columns = columns
        self.variable_key = variable_key
        self.summary = self.output_path('summary.txt')
        if latex:
            self.latex = self.output_path('latex_summary.txt')
    
    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        if not self.columns:
            first_value = next(iter(self.value_bundle.values()))
            if isinstance(first_value, tk.Variable):
                first_value = first_value.get()
            self.columns = first_value.keys()
        self.columns = columns = list(self.columns)

        headers = [self.header] + self.columns
        def row_to_values(row):
            if isinstance(row, tk.Variable):
                value = row.get()
                return [value[c] for c in columns]
            if self.variable_key is not None:
                return [row[c].get()[self.variable_key] for c in columns]
            return [self.get_wer(row[c]) for c in columns]
        data = [
            [key] + row_to_values(row) for key, row in self.value_bundle.items()
        ]
        with open(self.summary.get_path(), 'w') as summary_file:
            summary_file.write(tab.tabulate(data, headers=headers, tablefmt='presto'))
        
        if hasattr(self, "latex"):
            with open(self.latex.get_path(), 'w') as summary_file:
                summary_file.write(tab.tabulate(data, headers=headers, tablefmt='latex'))
    
    @staticmethod
    def get_wer(variable):
        value = variable.get()
        if isinstance(value, str) and "UNFINISHED VARIABLE" in value:
            wer_path    = value.split(" ")[-1][:-1]
            report_path = os.path.join(os.path.dirname(wer_path), "reports/sclite.dtl")
            return SummaryJob.wer(report_path)
        return value

    @staticmethod
    def wer(path):
        regex = re.compile('^Percent Total Error *= *(\\d+\.\\d%).*')
        with open(path, 'rt') as f:
            for line in f:
                m = regex.match(line)
                if m is not None:
                    return m.group(1)
        return None