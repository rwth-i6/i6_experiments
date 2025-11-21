from returnn.util.basic import NumbersDict
from returnn.torch.data.pipeline import BatchingIterDataPipe


# noinspection PyAbstractClass
class AlternateBatchingIterDataPipe(BatchingIterDataPipe):
    # noinspection PyShadowingNames
    def __iter__(self):
        """
        :return: generator providing batches in the form of lists of sequences, where each sequence is a dict
          data_key -> data_array.
        :rtype: Iterable[list[dict[str, numpy.ndarray]]]
        """
        current_batch_asr = []
        current_batch_asr_max_sequence_lengths = NumbersDict(0)  # data_key -> length of longest seq in batch
        current_batch_text = []
        current_batch_text_max_sequence_lengths = NumbersDict(0)  # data_key -> length of longest seq in batch

        for data_dict in self._dataset:
            max_seqs = self._parse_max_seqs(self._max_seqs, data_dict=data_dict)
            max_batch_size = self._parse_batch_size(self._max_batch_size, data_dict=data_dict)
            assert isinstance(max_seqs, int) and max_seqs > 0
            assert isinstance(max_batch_size, NumbersDict) and max_batch_size.min_value() > 0

            if len(current_batch_asr) >= max_seqs or len(current_batch_text) >= max_seqs:
                if current_batch_asr:
                    yield current_batch_asr
                if current_batch_text:
                    yield current_batch_text
                current_batch_asr = []
                current_batch_asr_max_sequence_lengths = NumbersDict(0)
                current_batch_text = []
                current_batch_text_max_sequence_lengths = NumbersDict(0)

            is_asr = data_dict["asr"].shape[0] > 0
            if is_asr:
                current_batch = current_batch_asr
                current_max_sequence_lengths = current_batch_asr_max_sequence_lengths
            else:
                current_batch = current_batch_text
                current_max_sequence_lengths = current_batch_text_max_sequence_lengths

            # Note: This assumes all data has time as first dimension. Currently we can't know better..
            sequence_lengths = NumbersDict(
                {data_key: data.shape[0] for data_key, data in data_dict.items() if data.shape}
            )

            max_sequence_lengths_if_included = NumbersDict.max([current_max_sequence_lengths, sequence_lengths])
            batch_size_if_included = max_sequence_lengths_if_included * (len(current_batch) + 1)  # including padding

            if current_batch and batch_size_if_included.any_compare(max_batch_size, (lambda a, b: a > b)):
                if current_batch_asr:
                    yield current_batch_asr
                if current_batch_text:
                    yield current_batch_text
                current_batch_asr = []
                current_batch_asr_max_sequence_lengths = NumbersDict(0)
                current_batch_text = []
                current_batch_text_max_sequence_lengths = NumbersDict(0)

                if is_asr:
                    current_batch_asr.append(data_dict)
                    current_batch_asr_max_sequence_lengths = sequence_lengths
                else:
                    current_batch_text.append(data_dict)
                    current_batch_text_max_sequence_lengths = sequence_lengths
            else:
                current_batch.append(data_dict)
                if is_asr:
                    current_batch_asr_max_sequence_lengths = max_sequence_lengths_if_included
                else:
                    current_batch_text_max_sequence_lengths = max_sequence_lengths_if_included

        if current_batch_asr:
            yield current_batch_asr
        if current_batch_text:
            yield current_batch_text
