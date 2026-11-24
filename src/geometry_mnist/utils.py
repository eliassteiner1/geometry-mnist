import h5py
import numpy as np

from geometry_mnist.paths import ROOT


def _read_array(f: h5py.File, key: str) -> np.ndarray:
    """ helper for typechecker to handle reading h5py data. makes sure output always is np.ndarray """
    
    ds = f[key]
    if not isinstance(ds, h5py.Dataset):
        raise TypeError(f"'{key}' is not a dataset, got {type(ds).__name__}")
    return ds[...]


def load_data(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    
    print(f"> loading dataset {dataset_name} ...")
    
    with h5py.File(ROOT / "data" / dataset_name, "r") as f:
        data = _read_array(f, "data")
        targets = _read_array(f, "targets")
        
    print("> successfully loaded data!")
    return data, targets


def write_data(dataset_name: str, data: np.ndarray, targets: np.ndarray) -> None:
    
    print(f"> writing dataset {dataset_name}")
    
    with h5py.File(ROOT / "data" / dataset_name, "w") as f:
        f["data"] = data
        f["targets"] = targets
        
    print(f"> done writing dataset!")