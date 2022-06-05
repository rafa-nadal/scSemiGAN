import csv
import numpy as np
import pandas as pd
import cmath as cm
import h5py
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight


def read_cell_for_h5(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        print(f["obs"].keys())

        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        print(X)
    return X, cell_label, cell_type, cell_type.shape[0], obs, var


if __name__ == "__main__":

    data_array, data_label, cell_type, cell_class, obs, var = read_cell_for_h5(
        "data/Quake_Smart-seq2_Diaphragm/data.h5")

    print(data_array)
    print(data_label)
    print(cell_type)
    print(cell_type.dtype)
    print(cell_class)
    print(obs)
    print(var)

    print(data_array.shape)
    print(data_label.shape)
    print(cell_type.shape)
    print(cell_class)
    print(obs.shape)
    print(var.shape)


