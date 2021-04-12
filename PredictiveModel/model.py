import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_dim=6, output_dim=4):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 32)
        self.output_layer = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.output_layer(x)

        return (x)