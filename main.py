import torch
import torch.nn as nn
import torch.optim as optim
 
from dataset import sequence_to_features, pdb_to_contact_map
from model import ContactCNN
from evaluate import precision_at_L5
 
 
def main():
    
    # SELECT FILES
 
    train_pdbs = ["1CRN.pdb", "1OQV.pdb", "1AY2.pdb", "2PIL.pdb"]  # add or remove training PDBs here
    test_pdb = "1HPW.pdb"
 
    print(f"Training on: {train_pdbs}")
    print(f"Testing on: {test_pdb}")
 
 
    # LOAD ALL TRAIN DATA
 
    train_data = []
 
    for pdb in train_pdbs:
        X = sequence_to_features(pdb).unsqueeze(0)
        y = pdb_to_contact_map(pdb)
 
        if y.dim() == 3:
            y = y.unsqueeze(1)
 
        train_data.append((X, y))
        print(f"Loaded {pdb} | Input: {X.shape} | Target: {y.shape}")
 
 
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
 
        optimizer.zero_grad()
 
        epoch_loss = 0.0
 
        # Accumulate gradients across all training proteins
        for X_train, y_train in train_data:
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            epoch_loss += loss.item()
 
        optimizer.step()
 
        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(train_data)
            print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")
 
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
 