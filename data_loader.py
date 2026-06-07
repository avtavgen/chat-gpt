import torch


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y