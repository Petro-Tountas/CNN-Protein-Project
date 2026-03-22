import torch
import numpy as np
from Bio.PDB import PDBParser


# Convert amino acid sequence into one-hot encoding
def one_hot_encode(sequence):

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = []

    for aa in sequence:
        vector = [0]*20

        # Set the correct index to 1
        if aa in amino_acids:
            vector[amino_acids.index(aa)] = 1

        encoding.append(vector)

    return np.array(encoding)


# Create pairwise feature tensor (NxN)
def sequence_to_pair_features(sequence):

    seq_encoded = one_hot_encode(sequence)
    L = len(sequence)

    # 21 channels: 20 for amino acids + 1 for distance
    pair_tensor = np.zeros((21, L, L))

    for i in range(L):
        for j in range(L):

            # Copy features of residue i
            pair_tensor[:20, i, j] = seq_encoded[i]

            # Add normalized sequence distance
            pair_tensor[20, i, j] = abs(i - j) / L

    return torch.tensor(pair_tensor).float()


# Convert PDB file into contact map
def pdb_to_contact_map(pdb_file, threshold=8.0):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = list(structure.get_residues())
    coords = []

    # Extract alpha carbon coordinates
    for res in residues:
        if "CA" in res:
            coords.append(res["CA"].coord)

    coords = np.array(coords)
    L = len(coords)

    contact_map = np.zeros((L, L))

    # Compute pairwise distances
    for i in range(L):
        for j in range(L):
            dist = np.linalg.norm(coords[i] - coords[j])

            # If distance < 8 Å → contact
            if dist < threshold:
                contact_map[i, j] = 1

    # Add channel dimension (1, L, L)
    return torch.tensor(contact_map).unsqueeze(0).float()