import torch
import torch.nn as nn

from model import ContactCNN
from dataset import sequence_to_pair_features, pdb_to_contact_map
from evaluate import precision_at_L5


# Example protein sequence (you can replace this)
sequence = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

# You must download a real PDB file and put path here
pdb_file = "1CRN.pdb"


# Convert sequence → input tensor
x = sequence_to_pair_features(sequence).unsqueeze(0)

# Convert PDB → ground truth contact map
y = pdb_to_contact_map(pdb_file).unsqueeze(0)


# Initialize model
model = ContactCNN()

# Binary classification loss
criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
for epoch in range(50):

    # Forward pass
    prediction = model(x)

    # Compute loss
    loss = criterion(prediction, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())


# Evaluate model after training
precision = precision_at_L5(prediction[0][0], y[0][0])
print("Precision@L/5:", precision)