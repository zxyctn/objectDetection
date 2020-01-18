from torch.utils.data import Dataset, DataLoader
import torch

class_categories = ['eagle', 'dog', 'cat', 'tiger', 'starfish',
            'zebra', 'bison', 'antelope', 'chimpanzee', 'elephant']

class CustomDataset(Dataset):
    def __init__(self, train, features, labels):
        super(CustomDataset, self).__init__()
        self.train = train
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if (torch.is_tensor(idx)):
            idx = idx.toList()
        features = self.features[idx]
        label = self.labels[idx]
        
        feature = []
        feature.append(features)

        sample = {'features': feature, 'label': class_categories.index(label)}

        return sample