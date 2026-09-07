"""
PyTorch Dataset for 1D PIC HDF5 runs.
Usage:
    from dataset import PICDataset
    ds = PICDataset("data/runs", input_key="rho", target_key="E")
    loader = DataLoader(ds, batch_size=8, shuffle=True)
"""
import os
import glob
import numpy as np
from runner import load_result_hdf5


class PICDataset:
    """
    Dataset over HDF5 run files.
    Each file = one simulation. Can slice by time step or use full history.
    """

    def __init__(
        self,
        data_dir,
        input_key="rho",
        target_key="E",
        time_slice=None,
        transform=None,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Directory with run_*.h5 files
        input_key : str
            Key for model input (e.g. "rho", "phi", "ne")
        target_key : str
            Key for target (e.g. "E")
        time_slice : int or slice, optional
            If int, use that time index. If slice, use range. If None, use last.
        transform : callable, optional
            (x, y) -> (x', y') applied to each sample
        """
        self.data_dir = data_dir
        self.input_key = input_key
        self.target_key = target_key
        self.time_slice = time_slice
        self.transform = transform

        self.files = sorted(glob.glob(os.path.join(data_dir, "run_*.h5")))
        if not self.files:
            raise FileNotFoundError(f"No run_*.h5 in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = load_result_hdf5(self.files[idx])
        x = data[self.input_key]
        y = data[self.target_key]

        if self.time_slice is None:
            x = x[-1]
            y = y[-1]
        elif isinstance(self.time_slice, int):
            x = x[self.time_slice]
            y = y[self.time_slice]
        else:
            x = x[self.time_slice]
            y = y[self.time_slice]

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.transform:
            x, y = self.transform(x, y)
        return x, y


def get_dataloader(data_dir, batch_size=4, **kwargs):
    """Convenience: DataLoader with PICDataset."""
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("PyTorch required for DataLoader")
    ds = PICDataset(data_dir, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
