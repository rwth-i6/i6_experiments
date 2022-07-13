#!/u/mann/bin/python3.6
import sys, os
import copy
import matplotlib
import pickle

matplotlib.use('Tkagg')

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.widgets import Slider
from itertools import product


# define nomenclature
den_prefix = "den"
target_prefix = "target"
infix = "_train-clean-100_"

segment_file = "reduced.train.segments"

tdpf_array = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]

"""
with open(segment_file, 'r') as sfile:
    tag_space = [line.split("train-clean-100/")[1].replace("/", "_")[:-1] + ".npy"
            for line in sfile]
"""


comparisons = {
        "adam" : {"spf" : tdpf_array, "sif": tdpf_array},
        "eve" : dict(
                spf_array = [0.5],
                sif_array = np.arange(0.2, 0.5, 0.01)   
            ),
        "cain" : dict(
                spf_array = [0.5],
                sif_array = [0.5 * (1 - 2**(-n)) for n in range(1, 11)]
                
            ),
        "edith" : dict(
                am = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
            ),
        "esther" : dict(
                epoch = list(range(20))
            ),
        }

def load_by_params(am=0.001, tdp=1.0, prior=1.0, spf=0.5, sif=0.5, batch_idx=4):
    name = "am-{}_tdp-{}_prior-{}".format(am, tdp, prior)
    name += "_spf-{}_sif-{}".format(spf, sif)
    tag = "alignment.dump.{}.0".format(batch_idx)
    fname = os.path.join('.', 'dump', name, tag)
    return np.loadtxt(fname)


def load_comparison(name, spf_array=[0.5], sif_array=[0.5], cache_dir="plot_cache", recache=False):
    print(name)
    cache_name = "fdata.{}.pickle".format(name)
    cache_path = os.path.join(cache_dir, cache_name)

    if not os.path.exists(cache_path) or recache:
        # init data dictionary
        fdata = { key : dict.fromkeys(sif_array) for key in spf_array }

        print("start loading files...")
        total_files = len(spf_array)*len(sif_array)
        sys.stdout.write("Loading {} / {}".format(0, total_files))
        sys.stdout.flush()
        #fdata[sample_tdp][sample_tdp] = load_by_params(spf=sample_tdp, sif=sample_tdp)
        for count, (spf, sif) in enumerate(product(spf_array, sif_array)):
            sys.stdout.write("\rLoading {} / {}".format(count + 1, total_files))
            sys.stdout.flush()
            data = load_by_params(spf=spf, sif=sif)
            data[:,2] = np.exp(-data[:,2])
            fdata[spf][sif] = data

        print("\nDone.")
        print("Cache data...")
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(fdata, cache_file)
        print("Done.")
    else:
        print("Existing cache detected. Load data from cache")
        with open(cache_path, 'rb') as cache_file:
            fdata = pickle.load(cache_file)
        print("Done.")

    return fdata


def plot(fdata, spf_array=[0.5], sif_array=[0.5], title="missing title"):
    # plot data
    fig = plt.figure(title)
    ax = plt.subplot(1, 1, 1)

    #ax.set_aspect('auto')
    fig.subplots_adjust(bottom=0.25)

    # create custom colour map
    my_cmap = copy.copy(matplotlib.cm.get_cmap('hot'))
    my_cmap.set_bad((0,0,0))

    # plot first data set
    spf_init = spf_array[0]
    sif_init = sif_array[0]
    data = fdata[spf_init][sif_init]
    l = ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
    ax.set_title("speech.forward={} / sil.forward={}".format(spf_init, sif_init))

    # make the slider
    axcolor = 'lightgoldenrodyellow'
    axspf = plt.axes([0.25, 0.05, 0.65, 0.03]) #, facecolor=axcolor)
    axsif = plt.axes([0.25, 0.15, 0.65, 0.03]) #, facecolor=axcolor)

#    for ax, ticks in zip([axspf, axsif], [spf_array, sif_array]):
    axspf.xaxis.set_visible(True)
    axspf.set_xticklabels(spf_array)
    axsif.xaxis.set_visible(True)
    axsif.set_xticklabels(sif_array)

    sspf = Slider(axspf, 'spe.fwd', 0, len(spf_array), valinit=0, valfmt='%d', valstep=1, facecolor=None, color=None)
    ssif = Slider(axsif, 'sil.fwd', 0, len(sif_array), valinit=0, valfmt='%d', valstep=1, facecolor=None, color=None)

    # call back function
    def update(val):
        spf = spf_array[int(sspf.val)]
        sif = sif_array[int(ssif.val)]
        data = fdata[spf][sif]
        ax.cla()
        ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
        ax.set_title("speech.forward={} / sil.forward={}".format(spf, sif))
        fig.canvas.draw()

    # connect callback to slider   
    sspf.on_changed(update)
    ssif.on_changed(update)

    plt.show()


def load_am_comparison(name, am_array=[0.001], spf_array=[0.5], sif_array=[0.5], cache_dir="plot_cache", recache=False):
    print(name)
    cache_name = "fdata.{}.pickle".format(name)
    cache_path = os.path.join(cache_dir, cache_name)

    if not os.path.exists(cache_path) or recache:
        # init data dictionary
        fdata = dict.fromkeys(am_array)

        print("start loading files...")
        total_files = len(am_array)
        sys.stdout.write("Loading {} / {}".format(0, total_files))
        sys.stdout.flush()
        #fdata[sample_tdp][sample_tdp] = load_by_params(spf=sample_tdp, sif=sample_tdp)
        for count, am in enumerate(am_array):
            sys.stdout.write("\rLoading {} / {}".format(count + 1, total_files))
            sys.stdout.flush()
            data = load_by_params(am=am)
            data[:,2] = np.exp(-data[:,2])
            fdata[am] = data

        print("\nDone.")
        print("Cache data...")
        with open(cache_path, 'wb') as cache_file:
            pickle.dump(fdata, cache_file)
        print("Done.")
    else:
        print("Existing cache detected. Load data from cache")
        with open(cache_path, 'rb') as cache_file:
            fdata = pickle.load(cache_file)
        print("Done.")

    return fdata


def plot_am(fdata, am_array=[0.001], spf_array=[0.5], sif_array=[0.5], title="missing title"):
    # plot data
    fig = plt.figure(title)
    ax = plt.subplot(1, 1, 1)

    #ax.set_aspect('auto')
    fig.subplots_adjust(bottom=0.15)

    # create custom colour map
    my_cmap = copy.copy(matplotlib.cm.get_cmap('hot'))
    my_cmap.set_bad((0,0,0))

    # plot first data set
    am_init = am_array[0]
    data = fdata[am_init]
    l = ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
    ax.set_title("am_scale = {}".format(am_init))

    # make the slider
    axcolor = 'lightgoldenrodyellow'
    axam = plt.axes([0.25, 0.05, 0.65, 0.03]) #, facecolor=axcolor)

#    for ax, ticks in zip([axspf, axsif], [spf_array, sif_array]):
    axam.xaxis.set_visible(True)
    axam.set_xticklabels(spf_array)

    sam = Slider(axam, 'am scale', 0, len(am_array), valinit=0, valfmt='%d', valstep=1, facecolor=None, color=None)

    # call back function
    def update(val):
        am = am_array[int(sam.val)]
        data = fdata[am]
        ax.cla()
        ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
        ax.set_title("am_scale = {}".format(am))
        fig.canvas.draw()

    # connect callback to slider   
    sam.on_changed(update)

    plt.show()


class Monolith:

    def __init__(self, comparison):
        self.comparison = comparison
        self.sig = comparison.keys()

        # init dict over tuples of keys
        arrays = [self.comparison[key] for key in self.sig]
        self._keys = list(product(*arrays))
        #for key in self._keys:
            #print(key)
        self.size = len(list(self._keys))
        self.f_data = dict.fromkeys(self._keys)


    def from_dict(self, dictionary):
        """ Get data from dict. """
        self.f_data = dictionary


    def to_dict(self):
        """ Return data as dict. """
        return self.f_data

    def get(self, *args, **kwargs):
        """ Getter taking in dict of symbols and values. """
        keys = self._get_key_from_kwargs(args, kwargs)
        try:
            return self.f_data[keys]
        except KeyError:
            raise KeyError("Value not set for keys {}".format(keys))


    def set(self, value, *args, **kwargs):
        """ Setter taking in dict of symbols and values. """
        self.f_data[self._get_key_from_kwargs(args, kwargs)] = value


    def _get_key_from_kwargs(self, args, kwargs):
        # prefer args if given
        if len(args) >= len(self.sig):
            return tuple(args)
        # one arg might be tuple or list then try to use
        if len(args) == 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                if len(args[0]) == len(self.sig):
                    return tuple(args[0])
        # if not then incompatible
        if len(args) > 1:
            raise KeyError("Given positional arguments are incompatible with signature")
        # check correct kwargs given
        if not set(self.sig).issubset(kwargs):
            raise TypeError("Expected kwargs {}".format(self.sig))
        # assign keys
        return tuple(kwargs[s] for s in self.sig)


    def __iter__(self):
        """ Return iterator over all keys that returns a dictionary with the signature and
        the key values. """
        return (dict(zip(self.sig, keys)) for keys in self._keys)



class Adam(Monolith):

    def __init__(self, name, comparison, dump_path_string, 
            base_dir='.',
            cache_dir='.', recache=False):
        super().__init__(comparison)

        # names and caches
        self.name = name
        self.cache_name = "fdata.{}.pickle".format(self.name)
        self._cache_dir = cache_dir
        self.cache_path = os.path.join(self._cache_dir, self.cache_name)
        self.dump_path_string = dump_path_string
        self._base_dir = base_dir

        # set default values and update
        self.default_config = dict( 
                am = '0.001',
                tdp = '1.0',
                prior = '1.0',
                spf = '0.5',
                sif = '0.5'
            )
        self.default_config.update(**comparison)
        self._load_comparison(recache=recache)


    def _load_dump_from_file(self, dump_file):
        return np.loadtxt(dump_file) 


    def _get_dump_file_path(self, **kwargs):
        return os.path.join(self._base_dir, self.dump_path_string.format(**kwargs))


    def _get_dump_files_iter(self):
        return (self._get_dump_file_path(**kwargs) for kwargs in self)


    def _get_dump_files_and_keys_iter(self):
        return ((self._get_dump_file_path(**kwargs), kwargs) for kwargs in self)


    def _load_data_from_cache(self, cache_path):
        with open(cache_path, 'rb') as cfile:
            self.f_data = pickle.load(cfile)


    def _write_data_to_cache(self, fdata, cache_path):
        with open(cache_path, 'wb') as cfile:
            pickle.dump(fdata, cfile)

    
    def _load_comparison(self, recache=False):
        print(self.name)
        if not os.path.exists(self.cache_path) or recache:
            print("start loading files...")
            total_files = self.size
            sys.stdout.write("Loading {} / {}".format(0, total_files))
            sys.stdout.flush()
            for count, (dump_file, kwargs) in enumerate(self._get_dump_files_and_keys_iter()):
                sys.stdout.write("\rLoading {} / {}".format(count + 1, total_files))
                sys.stdout.flush()
                # load data
                data = self._load_dump_from_file(dump_file)
                data[:,2] = np.exp(-data[:,2])
                self.set(data, **kwargs)

            print("\nDone.")
            print("Cache data...")
            self._write_data_to_cache(self.f_data, self.cache_path)
            print("Done.")
        else:
            print("Existing cache detected. Load data from cache")
            self._load_data_from_cache(self.cache_path)
            print("Done.")


    def plot(self, title_string="no title", save=None):
        # plot data
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        #ax.set_aspect('auto')
        fig.subplots_adjust(bottom=0.25)

        # create custom colour map
        my_cmap = copy.copy(matplotlib.cm.get_cmap('hot'))
        my_cmap.set_bad((0,0,0))

        # plot first data set
        key_init = next(iter(self))
        data = self.get(**key_init)
        l = ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
        ax.set_title(title_string.format(**key_init))

        # make the slider axes objects
        axcolor = 'lightgoldenrodyellow'
        axes = {s : plt.axes([0.25, 0.05+0.10*i, 0.65, 0.03]) 
                for i, s in enumerate(self.sig)}

        # construct array of sliders
        sliders = {
                s : Slider(ax, s, 0, len(self.comparison[s])-1, 
                    valinit=0, valfmt='%d', valstep=1, facecolor=None, color=None)
                for s, ax in axes.items()
                }

        # call back function
        def update(val):
            kwargs = {s : self.comparison[s][int(slider.val)] 
                    for s, slider in sliders.items()}

            data = self.get(**kwargs)
            ax.cla()
            ax.scatter(data[:,0], data[:,1], c=data[:,2], marker='.', linewidths=0, cmap='cool')
            ax.set_title(title_string.format(**kwargs))
            fig.canvas.draw()

        # connect callback to slider   
        for slider in sliders.values():
            slider.on_changed(update)

        if save:
            plt.savefig(save)
        else:
            plt.show()


if __name__ == "__main__":
    # construct container
    a = Adam("epoch32", {'idx': list(range(11))}, 
            "dump_baseline_bw_epoch32_debug/alignment.dump.{idx}.0",
            base_dir="output/config_04_am_warmup/", 
            cache_dir="plots/cache",
            recache=False)

    a.plot(title_string="idx = {idx}")

