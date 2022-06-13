import errno
import gc
import glob
import shutil
import tempfile

import numpy as np
import os
import subprocess
import soundfile
import sys
import multiprocessing

from sisyphus import Job, Task, tk

from i6_core.lib import corpus as bliss_corpus


class PhaseReconstructor():

    def __init__(self, out_folder,
                 backend,
                 sample_rate,
                 window_shift,
                 window_size,
                 n_fft,
                 iterations,
                 preemphasis,
                 file_format,
                 corpus_format):
        """

        :param str out_folder:
        :param str backend:
        :param int sample_rate:
        :param float window_shift:
        :param float window_size:
        :param int n_fft:
        :param int iterations:
        :param float preemphasis:
        :param str file_format:
        :param str|None corpus_format:
        """
        self.out_folder = out_folder
        self.backend = backend
        self.sample_rate = sample_rate
        self.window_shift = window_shift
        self.window_size = window_size
        self.n_fft = n_fft
        self.iterations = iterations
        self.preemphasis = preemphasis
        self.file_format = file_format
        self.corpus_format = corpus_format


        self.reconstruct_function = None
        if self.backend in ['legacy', 'numpy']:
            self.reconstruct_function = self.griffin_lim
        elif self.backend in ['librosa']:
            self.reconstruct_function = self.librosa_griffin_lim
        else:
            assert False, "invalid backend: %s" % self.backend

        assert self.file_format in ['wav', 'ogg'], "invalid file format: %s" % self.file_format


    def inv_preemphasis(self, wav):
        """
        :param np.array wav:
        :param float k:
        :return:
        """
        from scipy import signal
        return signal.lfilter([1], [1, -self.preemphasis], wav)

    def _istft(self, lin_spec):
        import librosa
        return librosa.istft(stft_matrix=lin_spec,
                             hop_length=int(self.sample_rate*self.window_shift),
                             win_length=int(self.sample_rate*self.window_size))

    def _stft(self, waveform):
        import librosa
        return librosa.stft(y=waveform,
                            n_fft=self.n_fft,
                            hop_length=int(self.sample_rate*self.window_shift),
                            win_length=int(self.sample_rate*self.window_size),
                            pad_mode='constant')

    def griffin_lim(self, spectrogram):
        """
        old librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        """
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
        complex_spectrogram = np.abs(spectrogram).astype(np.complex)
        waveform = self._istft(complex_spectrogram * angles)
        for i in range(self.iterations):
            angles = np.exp(1j * np.angle(self._stft(waveform)))
            waveform = self._istft(complex_spectrogram * angles)
        return waveform

    def librosa_griffin_lim(self, spectrogram):
        import librosa
        return librosa.griffinlim(spectrogram,
                                  n_iter=self.iterations,
                                  hop_length=int(self.sample_rate*self.window_shift),
                                  win_length=int(self.sample_rate*self.window_size),)


    def convert(self, data_tuple):
        """
        perform the conversion, possibly multithreaded

        :param tuple(str, np.array) data_tuple:
        :return:
        """
        tag = data_tuple[0]
        print("reconstructing phase for %s" % tag)
        lin_spec = data_tuple[1]

        # in compliance with the bliss format, create folders if tag is seperated by slashes
        if "/" in tag:
            tag_split = tag.split("/")
            folder = "/".join(tag_split[:-1])
            print("create folder %s" % folder)
            mkdir_p(self.out_folder + "/" + folder)

        # if
        spec_width = int(self.n_fft/2)
        if lin_spec.shape[0] == spec_width:
            lin_spec = np.pad(lin_spec, ((1,0),(0,0)), mode='constant', constant_values=0)
        elif lin_spec.shape[0] == spec_width + 1:
            lin_spec = lin_spec[0, :] = 0
        else:
            assert False, "invalid feature shape %i in data, n_fft/2 is %i" % (lin_spec.shape[0], spec_width)

        waveform = self.reconstruct_function(lin_spec)

        if self.preemphasis != 0:
            waveform = self.inv_preemphasis(waveform)

        if self.file_format == "wav":
            path = os.path.join(self.out_folder, "%s.wav" % tag)
            save_wav(waveform, path, self.sample_rate)
        elif self.file_format == "ogg":
            path = os.path.join(self.out_folder, '%s.ogg' % tag)

            save_ogg(waveform, path, self.sample_rate)
            data, sr = soundfile.read(path)

            target_len = self.window_shift * (lin_spec.shape[1] - 2)
            file_len = (len(data) / float(self.sample_rate))

            error = False
            if sr != self.sample_rate:
                print("invalid sample rate in %s" % path, file=sys.stderr)
                error = True
            elif file_len < target_len:
                print("invalid length %f vs %f in %s" % (target_len, file_len, path), file=sys.stderr)
                error = True

            assert not error, "wav file could not be converted to a correct .ogg file"
            assert os.path.exists(path)
        else:
            assert False

        if self.corpus_format == "bliss":
            recording = bliss_corpus.Recording()
            corpus_name, recording_name, segment_name = tag.split("/")
            recording.name = recording_name
            recording.audio = path
            segment = bliss_corpus.Segment()
            segment.name = segment_name
            segment.start = 0
            segment.end = float(len(waveform)) / float(self.sample_rate)
            recording.add_segment(segment)
            return recording
        elif self.corpus_format == "json":
            return NotImplementedError
        else:
            return path


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def save_wav(wav, path, sr):
    from scipy.io import wavfile
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def save_ogg(wav, path, sr):
    """

    :param wav:
    :param path:
    :param sr:
    :return:
    """
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    p1 = subprocess.Popen(["ffmpeg", "-y", "-f", "s16le", "-ar", "%i" % sr, "-i", "pipe:0", path],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE)
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()


class HDFPhaseReconstruction(Job):

    def __init__(self, hdf_file, backend, iterations, sample_rate, window_shift, window_size, preemphasis, file_format,
                 time_rqmt=8, mem_rqmt=8, cpu_rqmt=4):
        """

        :param tk.Path hdf_file:
        :param str backend:
        :param int iterations:
        :param int sample_rate:
        :param float window_shift:
        :param float window_size:
        :param str file_format:
        """
        self.hdf_file = hdf_file
        self.backend = backend
        self.iterations = iterations
        self.sample_rate = sample_rate
        self.window_shift = window_shift
        self.window_size = window_size
        self.preemphasis = preemphasis
        self.file_format = file_format

        self.out_folder = self.output_path("corpus", directory=True)
        self.out_corpus = self.output_path("corpus/corpus.xml.gz")

        self.rqmt = {'time': time_rqmt, 'mem': mem_rqmt, 'cpu': cpu_rqmt}

    def tasks(self):
        yield Task('run', rqmt=self.rqmt)


    def run(self):
        import h5py

        temp_dir = tempfile.TemporaryDirectory(prefix="hdf_reconstruction_")
        ref_linear_data = h5py.File(self.hdf_file.get_path(), 'r')
        rl_inputs = ref_linear_data['inputs']
        rl_tags = ref_linear_data['seqTags']
        rl_lengths = ref_linear_data['seqLengths']

        n_fft = rl_inputs[0].shape[0]*2
        print("N_FFT from HDF: % i" % n_fft)

        converter = PhaseReconstructor(out_folder=temp_dir.name,
                                       backend=self.backend,
                                       sample_rate=self.sample_rate,
                                       window_shift=self.window_shift,
                                       window_size=self.window_size,
                                       n_fft=n_fft,
                                       iterations=self.iterations,
                                       preemphasis=self.preemphasis,
                                       file_format=self.file_format,
                                       corpus_format="bliss")

        corpus = bliss_corpus.Corpus()


        # H5py has issues with multithreaded loading, so buffer 512 spectograms
        # single threaded and then distribute to the workers for conversion

        p = multiprocessing.Pool(self.rqmt['cpu'])

        loaded_spectograms = []
        offset = 0
        for tag, length in zip(rl_tags, rl_lengths):
            tag = tag if isinstance(tag, str) else tag.decode()
            loaded_spectograms.append((tag, np.asarray(rl_inputs[offset:offset + length[0]]).T))
            offset += length[0]
            if len(loaded_spectograms) > 512:
                recordings = p.map(converter.convert, loaded_spectograms)

                for recording in recordings:
                    corpus.add_recording(recording)

                # force gc for minimal memory requirement
                del loaded_spectograms
                gc.collect()
                loaded_spectograms = []

        # process rest in the buffer
        if len(loaded_spectograms) > 0:
            recordings = p.map(converter.convert, loaded_spectograms)
            # put all recordings to the corpus
            for recording in recordings:
                corpus.add_recording(recording)

        corpus.name = tag.split("/")[0]
        corpus.dump("corpus.xml")
        replacement_string = "s:%s:%s:g" % (temp_dir.name, self.out_folder.get_path())
        subprocess.call(["sed", "-i", replacement_string, "corpus.xml"])
        subprocess.call(["gzip", "corpus.xml"])
        shutil.move("corpus.xml.gz", self.out_corpus.get_path())

        for path in glob.glob(temp_dir.name + "/*"):
            shutil.move(path, self.out_folder.get_path())

    @classmethod
    def hash(cls, kwargs):
        kwargs.pop('time_rqmt')
        kwargs.pop('mem_rqmt')
        return super().hash(kwargs)