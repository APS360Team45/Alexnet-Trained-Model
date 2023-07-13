import torch.nn as nn
import torch.nn.functional as F
import torch

class Al_Net(nn.Module):
    def __init__(self, name):
        super(Al_Net, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512*4*4, 328)
        self.fc2 = nn.Linear(328, 96)
        self.fc3 = nn.Linear(96, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x
    
# loading pretrained weights
parameters = torch.load('model_al_net_bs64_lr0.001_epoch30')
model = Al_Net('al_net')
model.load_state_dict(parameters)
