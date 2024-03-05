__all__ = ["PeakyAlignmentJob"]

from sisyphus import *

Path = setup_path(__package__)


class PeakyAlignmentJob(Job):
    def __init__(self, dataset_hdf: tk.Path) -> None:
        self.dataset_hdf = dataset_hdf
        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self) -> None:
        import h5py

        hdf_file = h5py.File(self.dataset_hdf)

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

        peaky_inputs = hdf_file["inputs"][:]
        for idx in range(len(peaky_inputs) - 1):
            if peaky_inputs[idx] == peaky_inputs[idx + 1]:
                peaky_inputs[idx] = 0

        out_hdf.create_dataset("inputs", data=peaky_inputs)
        for attr_key, attr_val in hdf_file["inputs"].attrs.items():
            out_hdf["inputs"].attrs[attr_key] = attr_val
        copy_data_group(hdf_file, "seqTags", out_hdf)
        copy_data_group(hdf_file, "seqLengths", out_hdf)
        if "targets" in hdf_file:
            copy_data_group(hdf_file, "targets", out_hdf)
