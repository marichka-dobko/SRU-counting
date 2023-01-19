from torch.utils.data import Dataset
import torch
import numpy as np


class CountingDataset(Dataset):

    def __init__(self, count_len):
        self.count_len = count_len

    def __len__(self):
        return self.count_len - 3

    @staticmethod
    def one_hot_encode(sequence, dict_size, seq_len):
        features = np.zeros((seq_len, dict_size), dtype=np.float32)
        for u in range(seq_len):
            features[u, sequence[u]] = 1

        return features

    def __getitem__(self, index):

        seq = [index, index + 1, index + 2]
        label = [index + 3]

        input_seq = self.one_hot_encode(seq, self.count_len, len(seq))
        label_seq = self.one_hot_encode(label, self.count_len, len(label))

        return {
            'data': torch.from_numpy(input_seq),
            'label': torch.from_numpy(label_seq)
        }
