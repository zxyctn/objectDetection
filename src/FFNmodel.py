import torch.nn as nn

class FFNModel(nn.Module):
    def __init__(self, hidden_dim):
        super(FFNModel, self).__init__()
        self.fc1 = nn.Linear(2048, hidden_dim) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out