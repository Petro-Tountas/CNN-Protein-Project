import torch
import torch.nn as nn
import torch.nn.functional as F

# This class defines the neural network
class ContactCNN(nn.Module):

    def __init__(self, input_channels=21):
        super(ContactCNN, self).__init__()

        # First convolution layer:
        # Takes input channels (features) and outputs 64 feature maps
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        # Second convolution layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Third convolution layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fourth convolution layer
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Final 1x1 convolution:
        # Compresses all features into a single output (contact probability)
        self.output = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):

        # Pass input through each layer with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Apply sigmoid so output is between 0 and 1 (probability)
        x = torch.sigmoid(self.output(x))

        return x