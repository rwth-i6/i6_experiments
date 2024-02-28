from sisyphus import Job, tk, Task

import h5py
import numpy as np


class FilterEmptySequencesJob(Job):
    def __init__(self, dataset_hdf: tk.Path) -> None:
        self.dataset_hdf = dataset_hdf

        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        hdf_file = h5py.File(self.dataset_hdf)

        seqTags = hdf_file["seqTags"][:]
        seqLengths = hdf_file["seqLengths"][:]

        valid_indices = np.where(seqLengths[:, 0] > 0)[0]
        filtered_seqTags = seqTags[valid_indices]
        filtered_seqLengths = seqLengths[valid_indices]

        def copy_data_group(src, key, dest):
            src_data = src[key]
            if isinstance(src_data, h5py.Dataset):
                dest.create_dataset(key, data=src_data[:])
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
            if isinstance(src_data, h5py.Group):
                dest.create_group(key)
                for attr_key, attr_val in src_data.attrs.items():
                    dest[key].attrs[attr_key] = attr_val
                for sub_key in src_data:
                    copy_data_group(src_data, sub_key, dest[key])

        out_hdf = h5py.File(self.out_hdf, "w")

        for attr_key, attr_val in hdf_file.attrs.items():
            out_hdf.attrs[attr_key] = attr_val

        copy_data_group(hdf_file, "inputs", out_hdf)
        out_hdf.create_dataset("seqTags", data=filtered_seqTags)
        for attr_key, attr_val in hdf_file["seqTags"].attrs.items():
            out_hdf["seqTags"].attrs[attr_key] = attr_val
        out_hdf.create_dataset("seqLengths", data=filtered_seqLengths)
        for attr_key, attr_val in hdf_file["seqLengths"].attrs.items():
            out_hdf["seqLengths"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "targets", out_hdf)
