from sisyphus import *

import sys, os
import matplotlib.pyplot as plt
from itertools import filterfalse
from tabulate import tabulate

import recipe.lib.sprint_cache as sc
from sisyphus import *


class AlignmentStatisticsJob(Job):

    def __init__(self, alignment_bundles):
        self.alignment_bundles = alignment_bundles
        self.statistics_file = self.output_path("statistics.txt")
        self.statistics = self.output_var("statistics")
    
    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        headers = ["alignment", "prepended silence", "appended silence", "total silence"]
        data = [[k] + list(get_stats(alignment.get_path(), '', decimals=6)) 
                for k, alignment in self.alignment_bundles.items()]
        table = tabulate(data, headers=headers)
        with open(self.statistics_file.get_path(), 'w') as stat_file:
            stat_file.write(table)


class MultipleEpochAlignmentStatisticsJob(Job):

    def __init__(self, alignment_bundles):
        self.alignment_bundles = alignment_bundles
        self.statistics = self.output_path("statistics.txt")
    
    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        epochs = sorted(self.alignment_bundles.keys())
        e = epochs[0]
        ks = self.alignment_bundles[e].keys()
        headers = ["Alignment \\ Epoch"] + epochs
        data = [[k] + [get_stats(self.alignment_bundles[epoch][k].get_path(), '', decimals=6)[2]
                       for epoch in epochs] for k in ks]
        table = tabulate(data, headers=headers, tablefmt='presto')
        with open(self.statistics.get_path(), 'w') as stat_file:
            stat_file.write(table)


class HammingDistance(Job):
    """ Compute Hamming distance between two alignments. """

    def __init__(self, alignment1, alignment2, allophones):
        # self.alignments = [alignment1, alignment2]
        self.alignments = [
            al.get_path() if isinstance(al, tk.Variable) else al
            for al in [alignment1, alignment2]
        ]
        self.allophones = allophones
        # outputs 
        self.distance = self.output_var("distance")
    
    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        archives = [sc.FileArchiveBundle(al.get_path()) for al in self.alignments]
        for archive in archives: archive.setAllophones(self.allophones)
        fnames = archives[0].files
        total_dist = total_frames = 0
        for f in fnames:
            if f.endswith('attribs'): continue
            alignments = [archive.read(f, "align_raw") for archive in archives]
            total_frames += len(alignments[0])
            total_dist += sum(a1 != a2 for a1, a2 in zip(*alignments))
            # total_dist += len(set(alignments[0]) - set(alignments[1]))
        res = {
            "total_dist"   : total_dist, 
            "total_frames" : total_frames, 
            "relative_dist": float(total_dist) / total_frames}
        self.distance.set(res)


def get_alignments(fpath):
    alignments = open(fpath, 'r').read().split('\n')[:-1]
    return alignments


def get_segment_string(alignments):
    border_indicator = "flow-attribute"
    segment_string = ""
    for alignment in alignments:
        with open(alignment, 'r', encoding='latin-1') as file:
            for line in file:
                if border_indicator in line:
                    segment_string += "s"
                if "[SILENCE]" in line:
                    segment_string += "0"
                elif border_indicator not in line:
                    segment_string += "1"
    return segment_string


def compute_stats(segment_string, normalization='segment'):
    segments = segment_string.split("s")
    segments = list(filterfalse(lambda x: x == '', segments))
    num_segments = float(len(segments))
    num_frames = sum(map(len, segments))

    # define methods for getting zeros in on segment
    get_prepended_zero_string = lambda segment : len(segment.split('1')[0]) #/ float(len(segment))
    get_appended_zero_string  = lambda segment : len(segment.split('1')[-1]) #/ float(len(segment))
    get_segment_zero_string   = lambda segment : segment.count('0')
    get_prepended_zero_ratio  = lambda segment : len(segment.split('1')[0]) / float(len(segment))
    get_appended_zero_ratio   = lambda segment : len(segment.split('1')[-1]) / float(len(segment))
    get_segment_zero_ratio    = lambda segment : segment.count('0') / float(len(segment))

    # normalize on each segment or on full alignment
    if normalization == 'segment':
        # map to segment list
        num_prepended_zeros = sum(map(get_prepended_zero_ratio, segments)) / num_segments 
        num_appended_zeros  = sum(map(get_appended_zero_ratio , segments)) / num_segments
        silence_ratio       = sum(map(get_segment_zero_ratio  , segments)) / num_segments
    else:
        # get zeros total
        total_prepended_zeros = sum(map(get_prepended_zero_string, segments))
        total_appended_zeros  = sum(map(get_appended_zero_string , segments))
        total_zeros           = sum(map(get_segment_zero_string  , segments))
        
        # normalize on total number of frames
        num_prepended_zeros = total_prepended_zeros / float(num_frames)
        num_appended_zeros  = total_appended_zeros  / float(num_frames)
        silence_ratio       = total_zeros           / float(num_frames)
        

    return num_prepended_zeros, num_appended_zeros, silence_ratio


def get_stats(fpath, normalization='segment', decimals=3):
    alignments = get_alignments(fpath)
    segment_string = get_segment_string(alignments)

    # compute statistics and round
    stats = compute_stats(segment_string, normalization=normalization) 
    round_stats = tuple(map(lambda x: round(x, decimals), stats))

    return round_stats


def print_summary(rel_path, prefix, title=None, er_iter=None):
    if title:
        print(title)

    if not er_iter:
        er_iter = {
                1 : [1, 2, 3, 4],
                2 : [1, 2, 3],
                4 : [1]
                }

    headers = ["iteration", "prepended zeros", "appended zeros", "total zeros"]


    for er, iters in er_iter.items():

        print("Epoch-per-Realignment-Ratio = {}".format(er))

        table = []

        for i in iters:
            # build name
            fname = prefix
            if er > 1:
                fname += "_ER-{}".format(er)
            fname += "_iter-{}".format(i)

            row = [i] + list(get_stats(rel_path, fname, "total"))
            table += [row]

        print(tabulate(table, headers=headers, tablefmt='latex'))
        print("")


def print_altered_summary(rel_path, prefix, title=None, er_iter=None):
    if title:
        print(title)

    if not er_iter:
        er_iter = {
                1 : [1, 2, 3, 4],
                2 : [1, 2, 3],
                4 : [1]
                }

    headers = ["iteration", "between sentences", "inside sentences", "total"]


    for er, iters in er_iter.items():

        print("Epoch-per-Realignment-Ratio = {}".format(er))

        table = []

        for i in iters:
            # build name
            fname = prefix
            if er > 1:
                fname += "_ER-{}".format(er)
            fname += "_iter-{}".format(i)

            fractions = get_stats(rel_path, fname, "total")
            altered_fractions = (fractions[0] + fractions[1], 
                    fractions[2] - fractions[0] - fractions[1], 
                    fractions[2]) 

            row = [i] + list(altered_fractions)
            table += [row]

        print(tabulate(table, headers=headers, tablefmt='latex'))
        print("")


def print_single(fnames):
    # convert to list if single tuple
    if not isinstance(fnames, list):
        fnames = [fnames]

    headers = ["iteration", "prepended zeros", "appended zeros", "total zeros"]
    table = []
    for fname, tag in fnames:
        row = [tag] + list(get_stats('.', fname, normalization="total"))
        table += [row]
        
    print(tabulate(table, headers=headers, tablefmt='latex'))
    print("")


def print_altered_single(fnames):
    # convert to list if single tuple
    if not isinstance(fnames, list):
        fnames = [fnames]

    headers = ["iteration", "between sentences", "inside sentences", "total"]
    table = []
    for fname, tag in fnames:
        fractions = get_stats('.', fname, "total")
        altered_fractions = (fractions[0] + fractions[1], 
                fractions[2] - fractions[0] - fractions[1], 
                fractions[2]) 

        row = [tag] + list(altered_fractions)
        table += [row]
        
    print(tabulate(table, headers=headers, tablefmt='latex'))
    print("")
