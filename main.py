import torch
import torch.nn as nn
import torch.optim as optim

from dataset import sequence_to_features, pdb_to_contact_map
from model import ContactCNN
from evaluate import precision_at_L5


def main():
    
    # SELECT FILES
    
    train_pdb = "1CRN.pdb"
    test_pdb = "1UBQ.pdb"

    print(f"Training on: {train_pdb}")
    print(f"Testing on: {test_pdb}")

  
 # LOAD TRAIN DATA

    X_train = sequence_to_features(train_pdb).unsqueeze(0)
    y_train = pdb_to_contact_map(train_pdb)

    if y_train.dim() == 3:
        y_train = y_train.unsqueeze(1)

    print("Train input shape:", X_train.shape)
    print("Train target shape:", y_train.shape)



 # LOAD TEST DATA

    X_test = sequence_to_features(test_pdb).unsqueeze(0)
    y_test = pdb_to_contact_map(test_pdb)

    if y_test.dim() == 3:
        y_test = y_test.unsqueeze(1)

    print("Test input shape:", X_test.shape)
    print("Test target shape:", y_test.shape)



 # MODEL

    model = ContactCNN()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

   
 # TRAINING LOOP
    
    print("\nStarting training...\n")

    epochs = 50

    for epoch in range(epochs):
        model.train()

        preds = model(X_train)

        loss = criterion(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    print("\nTraining complete.\n")

    
    # EVALUATION
    
    model.eval()

    with torch.no_grad():
        preds = model(X_test)

    # remove batch + channel for evaluation
    preds = preds.squeeze()
    y_test_eval = y_test.squeeze()

    precision = precision_at_L5(preds, y_test_eval)

    print("Final Results:")
    print("Precision @ L/5:", precision)


if __name__ == "__main__":
    main()