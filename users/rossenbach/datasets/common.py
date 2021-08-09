import copy

from sisyphus import Path

from i6_core.meta.system import CorpusObject


class DatasetGroup():
    """
    This class manages a group of datasets with the same preprocessing status
    This means that in any pipeline all datasets should behave the same.
    """

    def __init__(self, name):
        """
        :param str name: for debugging purposes
        """
        # any bliss datasets
        self._corpus_objects = {}  # type: dict[str, CorpusObject]
        self._zip_datasets = {}  # type: dict[str, Path]

        self._segmented_datasets = {}  # type: dict[str, tuple[str, Path]]

        self._file_format = None  # type: str|None
        self._name = name

    def add_corpus_object(self, name, corpus_object):
        """
        :param str name:
        :param CorpusObject corpus_object:
        :return:
        """
        assert name not in self._corpus_objects.keys()
        assert name not in self._segmented_datasets.keys()

        # set file format if not set yet
        if self._file_format:
            assert corpus_object.audio_format == self._file_format
        else:
            self._file_format = corpus_object.audio_format

        self._corpus_objects[name] = corpus_object

    def add_zip_dataset(self, name, zip_dataset):
        """
        :param str name:
        :param Path zip_dataset:
        :return:
        """
        assert name not in self._zip_datasets.keys()
        assert name not in self._segmented_datasets.keys()
        self._zip_datasets[name] = zip_dataset

    def add_segmented_dataset(self, segmented_dataset_name, original_dataset_name, segment_file):
        """
        :param str segmented_dataset_name: the name of the new segmented corpus
        :param str original_dataset_name: the name of the original corpus, used to get the bliss or zip file
        :param Path segment_file: plain segment file
        :return:
        """
        assert ((original_dataset_name in self._corpus_objects)
            or (original_dataset_name in self._zip_datasets))
        assert segmented_dataset_name not in self._segmented_datasets
        self._segmented_datasets[segmented_dataset_name] = (original_dataset_name, segment_file)

    def copy_segmented_datasets_from_group(self, tts_dataset_group):
        """

        :param DatasetGroup tts_dataset_group:
        :return:
        """
        for segmented_name, (original_name, segment_file) in tts_dataset_group._segmented_datasets.items():
            assert original_name in self._corpus_objects
            self.add_segmented_dataset(segmented_name, original_name, segment_file)

    def get_segmented_corpus_object(self, name):
        """
        IMPORTANT! returns only a copy

        :param str name:
        :return: a tuple of a copied corpus object and an optional segment file
        :rtype tuple(CorpusObject, Path|None)
        """
        if name in self._segmented_datasets:
            dataset_name, segment_file = self._segmented_datasets[name]
        else:
            dataset_name = name
            segment_file = None
        return copy.deepcopy(self._corpus_objects[dataset_name]), segment_file

    def get_segmented_zip_dataset(self, name):
        """
        :param str name:
        :return: a tuple of a zip dataset path and an optional segment file
        :rtype tuple(Path, Path|None)
        """
        if name in self._segmented_datasets:
            dataset_name, segment_file = self._segmented_datasets[name]
        else:
            dataset_name = name
            segment_file = None
        return self._zip_datasets[dataset_name], segment_file

    def __str__(self):
        string = "<DatasetGroup name='%s'>\n" % self._name
        string += "corpus objects: %s" % str(self._corpus_objects.keys())
        string += "segmented datasets: %s" % str([(s, t) for s, (t, _) in self._segmented_datasets])
        string += "</DatasetGroup>"
        return string

