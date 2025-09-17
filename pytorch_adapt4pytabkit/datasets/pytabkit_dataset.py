from torch.utils.data import Dataset

class DictToDataset(Dataset):
    def __init__(self, pytabkit_ds):
        self.tensors = pytabkit_ds.tensors
        self.length = pytabkit_ds.n_samples 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tensors.items()}