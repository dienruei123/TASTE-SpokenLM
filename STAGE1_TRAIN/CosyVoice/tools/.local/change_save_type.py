import os
import glob
import torch
import numpy as np

root_dir = "/proj/gpu_d_09023_MR_dataset_augmented/emilia/en/0-32"

_search_pattern = os.path.join(root_dir, "*.pt")
pt_files = glob.glob(_search_pattern)
print(pt_files)


def dict_to_npy_dict(_dict):
    new_dict = {}
    for key, val in _dict.items():
        new_val = np.array(val, dtype=np.uint16)
        new_dict[key] = new_val
    return new_dict

for pt_file in pt_files:
    _dict = torch.load(pt_file)
    npy_dict = dict_to_npy_dict(_dict)
    new_fpath = pt_file.replace(".pt", ".npz")
    np.savez(new_fpath, **npy_dict)
