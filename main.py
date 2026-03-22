import torch
import torch.nn as nn

from model import ContactCNN
from dataset import sequence_to_pair_features, pdb_to_contact_map
from evaluate import precision_at_L5


def main():

    # -----------------------------
    # 1. DEFINE INPUT DATA
    train_pdb = "1CRN.pdb"
    test_pdb = "1UBQ.pdb"
    # -----------------------------
    print("TRAIN PDB:", train_pdb)
    print("TEST PDB:", test_pdb)
    # Example protein: Crambin (1CRN)
    sequence = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

    pdb_file = "1CRN.pdb"  # Make sure this file exists in your folder



    print("Loading training data...")
    # TRAIN DATA
    X_train = sequence_to_features(train_pdb).unsqueeze(0)
    y_train = pdb_to_contact_map(train_pdb).unsqueeze(0)
    # TEST DATA
    X_test = sequence_to_features(test_pdb).unsqueeze(0)
    y_test = pdb_to_contact_map(test_pdb).unsqueeze(0)
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    # -----------------------------
    # 2. INITIALIZE MODEL
    # -----------------------------

    model = ContactCNN()

    # Binary classification loss
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # 3. TRAINING LOOP
    # -----------------------------

    epochs = 50

    print("\nStarting training...\n")
    output = model(X_train)
    loss = criterion(output, y_train)
    for epoch in range(epochs):

        # Forward pass
        prediction = model(x)

        # Compute loss
        loss = criterion(prediction, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    print("\nTraining complete.\n")

    print("\nLoading test data...")

   

    # -----------------------------
    # 4. EVALUATION
    # -----------------------------

    model.eval()

with torch.no_grad():
    prediction = model(X_test)

# Remove batch + channel dimensions
prediction = prediction.squeeze()
y_test = y_test.squeeze()

precision = precision_at_L5(prediction, y_test)

print("\nFinal Results (TEST SET):")
print("Precision @ L/5:", precision)

