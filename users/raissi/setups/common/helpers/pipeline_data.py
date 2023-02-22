__all__ = ["ContextEnum",
           "ContextMapper",
           "PipelineStages",
           "LabelInfo",
           "RasrFeatureToHDF",
           "RasrFeatureAndAlignmentToHDF", 
           "RasrFeatureAndAlignmentWithDenseAndCARTStateTyingsToHDF"]

from sisyphus import *
from i6_core.lib.rasr_cache import FileArchive, FileArchiveBundle

import h5py
import itertools as it
import numpy as np
from enum import Enum

from IPython import embed

Path = setup_path(__package__)


class ContextEnum(Enum):
    """
    These are the implemented models. The string value is the one used in the feature scorer of rasr, except monophone
    """

    monophone = "monophone"
    mono_state_transition = "monophone-delta"
    diphone = "diphone"
    diphone_state_transition = "diphone-delta"
    triphone_symmetric = "triphone-symmetric"
    triphone_forward = "triphone-forward"
    triphone_backward = "triphone-backward"
    tri_state_transition = "triphone-delta"


class ContextMapper:
    def __init__(self):
        self.contexts = {
            1: "monophone",
            2: "diphone",
            3: "triphone-symmetric",
            4: "triphone-forward",
            5: "triphone-backward",
            6: "triphone-delta",
            7: "monophone-delta",
            8: "diphone-delta",
        }

    def get_enum(self, contextTypeId):
        return self.contexts[contextTypeId]


class LabelInfo:
    def __init__(
        self,
        n_states_per_phone,
        n_contexts,
        ph_emb_size,
        st_emb_size,
        state_tying,
        state_tying_file=None,
        n_cart_labels=None,
        sil_id=None,
        use_word_end_classes=True,
        use_boundary_classes=False,
        add_unknown_phoneme=True,
        use_minimum_duration=False,
    ):
        self.n_states_per_phone = n_states_per_phone
        self.n_contexts = n_contexts
        self.sil_id = sil_id
        self.ph_emb_size = ph_emb_size
        self.st_emb_size = st_emb_size
        self.state_tying = state_tying
        self.use_word_end_classes = use_word_end_classes
        self.use_boundary_classes = use_boundary_classes
        self.add_unknown_phoneme = add_unknown_phoneme
        self.use_minimum_duration = use_minimum_duration

        if state_tying == 'cart':
            assert state_tying_file is not None, 'for cart state tying you need a file'
            assert n_cart_labels is not None, 'for cart you need to set number of cart labels'
            self.n_cart_labels = n_cart_labels
            self.state_tying_file = state_tying_file

    def get_n_of_dense_classes(self):
        n_contexts = self.n_contexts
        if not self.add_unknown_phoneme:
            n_contexts+=1
        return self.n_states_per_phone * (n_contexts**3) * (1 + int(self.use_word_end_classes))

    def get_n_state_classes(self):
        if self.state_tying == 'cart':
            assert self.n_cart_labels is not None
            return self.n_cart_labels
        return self.n_states_per_phone * self.n_contexts * (1 + int(self.use_word_end_classes))



class PipelineStages:
    def __init__(self, alignment_keys):
        self.names = dict(
            zip(alignment_keys, [self._get_context_dict(k) for k in alignment_keys])
        )

    def _get_context_dict(self, align_k):
        return {
            "mono": f"mono-from-{align_k}",
            "mono-delta": f"mono-delta-from-{align_k}",
            "di": f"di-from-{align_k}",
            "di-delta": f"di-delta-from-{align_k}",
            "tri": f"tri-from-{align_k}",
            "tri-delta": f"tridelta-from-{align_k}",
            "cart": f"cart-from-{align_k}",
        }

    def get_name(self, alignment_key, context_type):
        return self.names[alignment_key][context_type]



class RasrFeatureToHDF(Job):

  def __init__(self, feature_caches):
    self.feature_caches = feature_caches
    self.hdf_files = [self.output_path('data.hdf.%d' % d, cached=False) for d in range(len(feature_caches))]
    self.rqmt = {'cpu': 1, 'mem': 8, 'time': 1.0}

  def tasks(self):
    yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

  def run(self, task_id):

    seq_names = []
    string_dt = h5py.special_dtype(vlen=str)

    feature_path = self.feature_caches[task_id - 1]
    if isinstance(feature_path, tk.Path):
        feature_path = feature_path.get_path()
    feature_cache = FileArchive(feature_path)
    out = h5py.File(self.hdf_files[task_id - 1].get_path(), 'w')

    # root
    streams_group = out.create_group('streams')

    # first level
    feature_group = streams_group.create_group('features')
    feature_group.attrs['parser'] = 'feature_sequence'

    # second level
    feature_data = feature_group.create_group('data')

    for file in feature_cache.ft:
      info = feature_cache.ft[file]
      if info.name.endswith('.attribs'):
        continue
      seq_names.append(info.name)

      # features
      times, features = feature_cache.read(file, 'feat')
      feature_data.create_dataset(seq_names[-1].replace('/', '\\'), data=features)

    out.create_dataset('seq_names', data=[s.encode() for s in seq_names], dtype=string_dt)


class RasrFeatureAndAlignmentToHDF(Job):
    def __init__(self, feature_caches, alignment_caches, allophones, state_tying):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.state_tying = state_tying
        self.hdf_files = [self.output_path('data.hdf.%d' % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {'cpu': 1, 'mem': 8, 'time': 0.5}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        num_classes = 0
        for line in open(self.state_tying.get_path(), 'rt'):
            if not line.startswith('#'):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        string_dt = h5py.special_dtype(vlen=str)
        state_tying = dict(
            (k, int(v)) for l in open(self.state_tying.get_path()) for k, v in [l.strip().split()[0:2]])

        feature_cache = FileArchive(self.feature_caches[task_id - 1].get_path())
        alignment_cache = FileArchive(
            self.alignment_caches[min(task_id - 1, len(self.alignment_caches) - 1)].get_path())
        alignment_cache.setAllophones(self.allophones.get_path())

        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), 'w')

        # root
        streams_group = out.create_group('streams')

        # first level
        feature_group = streams_group.create_group('features')
        feature_group.attrs['parser'] = 'feature_sequence'

        alignment_group = streams_group.create_group('alignment')
        alignment_group.attrs['parser'] = 'sparse'
        alignment_group.create_dataset('feature_names', data=[b'label_%d' % l for l in range(num_classes + 1)],
                                       dtype=string_dt)

        # second level
        feature_data = feature_group.create_group('data')
        alignment_data = alignment_group.create_group('data')

        for file in feature_cache.ft:
            info = feature_cache.ft[file]
            if info.name.endswith('.attribs'):
                continue

            seq_names.append(info.name)

            # features
            times, features = feature_cache.read(file, 'feat')
            feature_data.create_dataset(seq_names[-1].replace('/', '\\'), data=features)

            # alignment
            alignment = alignment_cache.read(file, 'align')

            targets = []

            alignmentStates = ['%s.%d' % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]

            for allophone in alignmentStates:
                targets.append(state_tying[allophone])

            alignment_data.create_dataset(seq_names[-1].replace('/', '\\'), data=targets)

        out.create_dataset('seq_names', data=[s.encode() for s in seq_names], dtype=string_dt)


class RasrFeatureAndAlignmentWithRandomAllophonesToHDF(Job):

    def __init__(self, feature_caches, alignment_caches, allophones, dense_tying, cart_tying, label_info):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.dense_tying = dense_tying
        self.cart_tying = cart_tying
        self.dense_label_info = label_info
        self.hdf_files = [self.output_path('data.hdf.%d' % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {'cpu': 1, 'mem': 8, 'time': 0.5}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        import itertools as it
        import random
        random.seed(42)

        cart_to_dense_dict = self.get_dense_cluster_dict(self.cart_tying, self.dense_tying)
        cart_tying = dict((k, int(v)) for l in open(cart_st_path.get_path()) for k, v in [l.strip().split()[0:2]])

        #allophone and state_tying are generally output of a job, alignment and features might be strings
        feature_path = self.feature_caches[task_id - 1]
        if isinstance(feature_path, tk.Path):
            feature_path = feature_path.get_path()
        feature_cache = FileArchive(feature_path)

        alignment_path = self.alignment_caches[min(task_id - 1, len(self.alignment_caches) - 1)]
        if isinstance(alignment_path, tk.Path):
            alignment_path = alignment_path.get_path()
        alignment_cache = FileArchive(alignment_path)
        alignment_cache.setAllophones(self.allophones.get_path())


        string_dt = h5py.special_dtype(vlen=str)
        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), 'w')

        #num_classes for each label
        n_center_state_classes = self.dense_label_info.n_contexts * self.dense_label_info.n_states_per_phone
        n_contexts = self.dense_label_info.n_contexts

        # root
        streams_group = out.create_group('streams')

        # first level
        feature_group = streams_group.create_group('features')
        feature_group.attrs['parser'] = 'feature_sequence'

        futureLabel_group = streams_group.create_group('futureLabel')
        futureLabel_group.attrs['parser'] = 'sparse'
        futureLabel_group.create_dataset('feature_names',
                                         data=[b'label_%d' % l for l in range(n_contexts)],
                                         dtype=string_dt)

        centerStateLabel_group = streams_group.create_group('centerState')
        centerStateLabel_group.attrs['parser'] = 'sparse'
        centerStateLabel_group.create_dataset('feature_names',
                                              data=[b'label_%d' % l for l in range(n_center_state_classes)],
                                              dtype=string_dt)

        pastLabel_group = streams_group.create_group('pastLabel')
        pastLabel_group.attrs['parser'] = 'sparse'
        pastLabel_group.create_dataset('feature_names',
                                       data=[b'label_%d' % l for l in range(n_contexts)],
                                       dtype=string_dt)

        # second level
        feature_data = feature_group.create_group('data')
        futureLabel_data = futureLabel_group.create_group('data')
        centerstateLabel_data = centerStateLabel_group.create_group('data')
        pastLabel_data = pastLabel_group.create_group('data')

        for file in feature_cache.ft:
            info = feature_cache.ft[file]
            if info.name.endswith('.attribs'):
                continue

            seq_names.append(info.name)

            # features
            times, features = feature_cache.read(file, 'feat')
            feature_data.create_dataset(seq_names[-1].replace('/', '\\'), data=features)

            # alignment
            alignment = alignment_cache.read(file, 'align')
            aligned_allophones = ['%s.%d' % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]
            cart_targets = [cart_tying[allo] for allo in aligned_allophones]



            #optimize the calculation by grouping
            futureLabel_strings = []
            centerState_strings = []
            pastLabel_strings = []
            for k_cart, g in it.groupby(cart_targets):
                segLen = len(list(g))
                k = random.choice(cart_to_dense_dict[k_cart])
                f, c, l = self.get_target_labels_from_dense(k)
                futureLabel_strings = futureLabel_strings + [f] * segLen
                centerState_strings = centerState_strings + [c] * segLen
                pastLabel_strings   = pastLabel_strings + [l] * segLen

            #initialize last level data
            futureLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=futureLabel_strings)
            centerstateLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=centerState_strings)
            pastLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=pastLabel_strings)

        out.create_dataset('seq_names', data=[s.encode() for s in seq_names], dtype=string_dt)

    def get_dense_cluster_dict(self, cart_st_path, dense_st_path):
        cart_tying = dict((k, int(v)) for l in open(cart_st_path.get_path()) for k, v in [l.strip().split()[0:2]])
        dense_tying = dict((k, int(v)) for l in open(dense_st_path.get_path()) for k, v in [l.strip().split()[0:2]])

        allophone_clusters = {}
        for k, v in cart_tying.items():
            dense_label = dense_tying[k]
            if v not in allophone_clusters:
                allophone_clusters[v] = []
            allophone_clusters[v].append(dense_label)

        return allophone_clusters, num_classes


    def get_target_labels_from_dense(self, dense_label):
        import numpy as np
        n_contexts = self.dense_label_info.n_contexts

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)

        return futureLabel, centerState, pastLabel


class RasrFeatureAndAlignmentWithDenseAndCARTStateTyingsToHDF(Job):

    def __init__(self, feature_caches, alignment_caches, allophones, dense_tying, cart_tying, label_info):
        self.feature_caches = feature_caches
        self.alignment_caches = alignment_caches
        self.allophones = allophones
        self.dense_tying = dense_tying
        self.cart_tying = cart_tying
        self.dense_label_info = label_info
        self.hdf_files = [self.output_path('data.hdf.%d' % d, cached=False) for d in range(len(feature_caches))]
        self.rqmt = {'cpu': 1, 'mem': 8, 'time': 0.5}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, (len(self.feature_caches) + 1)))

    def run(self, task_id):
        dense_tying, _ = self.get_tying_and_num_classes(self.dense_tying)
        cart_tying, cart_num_classes = self.get_tying_and_num_classes(self.cart_tying)

        #allophone and state_tying are generally output of a job, alignment and features might be strings
        feature_path = self.feature_caches[task_id - 1]
        if isinstance(feature_path, tk.Path):
            feature_path = feature_path.get_path()
        feature_cache = FileArchive(feature_path)

        alignment_path = self.alignment_caches[min(task_id - 1, len(self.alignment_caches) - 1)]
        if isinstance(alignment_path, tk.Path):
            alignment_path = alignment_path.get_path()
        alignment_cache = FileArchive(alignment_path)
        alignment_cache.setAllophones(self.allophones.get_path())


        string_dt = h5py.special_dtype(vlen=str)
        seq_names = []
        out = h5py.File(self.hdf_files[task_id - 1].get_path(), 'w')

        #num_classes for each label
        n_center_state_classes = self.dense_label_info.n_contexts * self.dense_label_info.n_states_per_phone
        n_contexts = self.dense_label_info.n_contexts

        # root
        streams_group = out.create_group('streams')

        # first level
        feature_group = streams_group.create_group('features')
        feature_group.attrs['parser'] = 'feature_sequence'

        cartLabel_group = streams_group.create_group('cartLabel')
        cartLabel_group.attrs['parser'] = 'sparse'
        cartLabel_group.create_dataset('feature_names',
                                        data=[b'label_%d' % l for l in range(cart_num_classes)],
                                        dtype=string_dt)

        futureLabel_group = streams_group.create_group('futureLabel')
        futureLabel_group.attrs['parser'] = 'sparse'
        futureLabel_group.create_dataset('feature_names',
                                         data=[b'label_%d' % l for l in range(n_contexts)],
                                         dtype=string_dt)

        centerStateLabel_group = streams_group.create_group('centerState')
        centerStateLabel_group.attrs['parser'] = 'sparse'
        centerStateLabel_group.create_dataset('feature_names',
                                              data=[b'label_%d' % l for l in range(n_center_state_classes)],
                                              dtype=string_dt)

        pastLabel_group = streams_group.create_group('pastLabel')
        pastLabel_group.attrs['parser'] = 'sparse'
        pastLabel_group.create_dataset('feature_names',
                                       data=[b'label_%d' % l for l in range(n_contexts)],
                                       dtype=string_dt)

        # second level
        feature_data = feature_group.create_group('data')
        cartLabel_data = cartLabel_group.create_group('data')
        futureLabel_data = futureLabel_group.create_group('data')
        centerstateLabel_data = centerStateLabel_group.create_group('data')
        pastLabel_data = pastLabel_group.create_group('data')

        for file in feature_cache.ft:
            info = feature_cache.ft[file]
            if info.name.endswith('.attribs'):
                continue

            seq_names.append(info.name)

            # features
            times, features = feature_cache.read(file, 'feat')
            feature_data.create_dataset(seq_names[-1].replace('/', '\\'), data=features)

            # alignment
            alignment = alignment_cache.read(file, 'align')
            aligned_allophones = ['%s.%d' % (alignment_cache.allophones[t[1]], t[2]) for t in alignment]

            cart_targets  = [cart_tying[allo] for allo in aligned_allophones]
            dense_targets = [dense_tying[allo] for allo in aligned_allophones]

            #optimize the calculation by grouping
            futureLabel_strings = []
            centerState_strings = []
            pastLabel_strings = []
            for k, g in it.groupby(dense_targets):
                segLen = len(list(g))
                f, c, l = self.get_target_labels_from_dense(k)
                futureLabel_strings = futureLabel_strings + [f] * segLen
                centerState_strings = centerState_strings + [c] * segLen
                pastLabel_strings   = pastLabel_strings + [l] * segLen

            #initialize last level data
            cartLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=cart_targets)
            futureLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=futureLabel_strings)
            centerstateLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=centerState_strings)
            pastLabel_data.create_dataset(seq_names[-1].replace('/', '\\'), data=pastLabel_strings)

        out.create_dataset('seq_names', data=[s.encode() for s in seq_names], dtype=string_dt)


    def get_tying_and_num_classes(self, state_tying_path):
        num_classes = 0
        for line in open(state_tying_path.get_path(), 'rt'):
            if not line.startswith('#'):
                num_classes = max(num_classes, int(line.strip().split()[1]))

        state_tying = dict(
            (k, int(v)) for l in open(state_tying_path.get_path()) for k, v in [l.strip().split()[0:2]])

        return state_tying, num_classes


    def get_target_labels_from_dense(self, dense_label):
        import numpy as np
        n_contexts = self.dense_label_info.n_contexts

        futureLabel = np.mod(dense_label, n_contexts)
        popFutureLabel = np.floor_divide(dense_label, n_contexts)

        pastLabel = np.mod(popFutureLabel, n_contexts)
        centerState = np.floor_divide(popFutureLabel, n_contexts)

        return futureLabel, centerState, pastLabel
