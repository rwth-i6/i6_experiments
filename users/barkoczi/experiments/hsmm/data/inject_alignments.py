import argparse
import h5py
import numpy as np
import glob
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_hdf", required=True, help="Input features HDF5 from Stage 2")
    parser.add_argument("--out_hdf", required=True, help="Output HDF5 path for Sisyphus")
    parser.add_argument("--align_dir", required=True, help="Directory containing alignment HDFs")
    parser.add_argument("--subsample", type=int, default=2, help="Subsampling factor (e.g., 2 for 10ms -> 20ms)")
    args = parser.parse_args()

    print(f"Copying {args.in_hdf} to {args.out_hdf} to preserve Sisyphus immutability...")
    shutil.copy(args.in_hdf, args.out_hdf)

    alignment_files = glob.glob(os.path.join(args.align_dir, "alignment_*.hdf"))
    print(f"Found {len(alignment_files)} alignment files in {args.align_dir}.")

    length_mismatches = 0
    injected_count = 0

    with h5py.File(args.out_hdf, "a") as f_out:
        missing_seq_keys = set(f_out.keys())
        
        for ali_file in alignment_files:
            with h5py.File(ali_file, "r") as f_in:
                for seq_tag in f_in.keys():
                    if seq_tag not in missing_seq_keys:
                        continue
                        
                    # Extract raw labels
                    if isinstance(f_in[seq_tag], h5py.Dataset):
                        raw_labels = f_in[seq_tag][:]
                    elif "data" in f_in[seq_tag]:
                        raw_labels = f_in[seq_tag]["data"][:]
                    else:
                        keys = list(f_in[seq_tag].keys())
                        raw_labels = f_in[seq_tag][keys[0]][:]

                    raw_labels = raw_labels.flatten()

                    # Compare lengths and subsample
                    features = f_out[seq_tag]["data"][:]
                    T_feat = features.shape[0]
                    
                    subsampled_labels = raw_labels[::args.subsample]
                    T_label = subsampled_labels.shape[0]
                    
                    if T_feat != T_label:
                        length_mismatches += 1
                        min_len = min(T_feat, T_label)
                        subsampled_labels = subsampled_labels[:min_len]
                        
                    if "labels" in f_out[seq_tag]:
                        del f_out[seq_tag]["labels"]
                        
                    f_out[seq_tag].create_dataset("labels", data=subsampled_labels.astype(np.int32), compression="gzip")
                    injected_count += 1
                    missing_seq_keys.remove(seq_tag)

    print("\n--- Merge Summary ---")
    print(f"Total sequences updated with labels: {injected_count}")
    print(f"Sequences missing alignments entirely: {len(missing_seq_keys)}")
    print(f"Length mismatches (truncated to match): {length_mismatches}")

if __name__ == "__main__":
    main()