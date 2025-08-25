import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 50

class GRMDataset(Dataset):
    def __init__(self, sequences):
        self.samples = []
        for seq in sequences:
            for i in range(1, len(seq)):
                input_seq = seq[max(0, i - MAX_SEQ_LEN):i]
                target = seq[i]
                pad_len = MAX_SEQ_LEN - len(input_seq)
                input_seq = [0] * pad_len + input_seq
                self.samples.append((input_seq, target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.LongTensor(x), torch.LongTensor([y])
