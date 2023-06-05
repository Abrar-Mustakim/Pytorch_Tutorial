from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        targets = self.y[index]

        return features, targets