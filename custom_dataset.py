import torch
import h5py

class CustomDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path: str,
    ):
        self.dataset  = h5py.File(dataset_path, 'r')
        
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> list:  
        data_group = self.dataset[str(index)]
        return [data_group[key][:] for key in data_group.keys()]

    def __len__(self):
        return len(self.dataset)