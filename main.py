# main.py

import torch
import torch.nn as nn
import torch.optim as optim

# Import your modules
from model import ContactCNN
from dataset import sequence_to_pair_features, pdb_to_contact_map
from evaluate import precision_at_L5


def main():

   
    # 1. CHOOSE PROTEINS
    
    train_pdb = "1CRN.pdb"
    test_pdb = "1UBQ.pdb"

    print("Training on:", train_pdb)
    print("Testing on:", test_pdb)

  
    # 2. LOAD DATA
   
    print("\nLoading training data...")

    X_train = sequence_to_pair_features(train_pdb).unsqueeze(0)
    y_train = pdb_to_contact_map(train_pdb).unsqueeze(0)

    print("Train input shape:", X_train.shape)
    print("Train target shape:", y_train.shape)

    print("\nLoading test data...")

    X_test = sequence_to_pair_features(test_pdb).unsqueeze(0)
    y_test = pdb_to_contact_map(test_pdb).unsqueeze(0)

    print("Test input shape:", X_test.shape)
    print("Test target shape:", y_test.shape)

    
    model = ContactCNN()

    # Loss function (binary classification)
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 
    # 4. TRAIN MODEL
    #
    print("\nStarting training...\n")

    epochs = 50

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_train)

        # Compute loss
        loss = criterion(predictions, y_train)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    print("\nTraining complete.")

    
    # 5. EVALUATE MODEL
    
    print("\nEvaluating model...")

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

    # Compute precision@L/5
    precision = precision_at_L5(predictions, y_test)

    print("\nFinal Results:")
    print(f"Precision @ L/5: {precision:.4f}")


# RUN PROGRAM

if __name__ == "__main__":
    main()