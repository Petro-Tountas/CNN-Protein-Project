# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactCNN(nn.Module):
    def __init__(self):
        super(ContactCNN, self).__init__()

        self.conv1 = nn.Conv2d(21, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # binary output
        return x