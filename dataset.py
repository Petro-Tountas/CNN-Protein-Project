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
def sequence_to_features(pdb_file):
    sequence = get_sequence_from_pdb(pdb_file)  
    L = len(sequence)

    # 21 channels (20 amino acids + 1 padding)
    features = torch.zeros((21, L, L))

    # simple encoding: pairwise identity
    for i in range(L):
        for j in range(L):
            features[0, i, j] = 1 if sequence[i] == sequence[j] else 0

    return features

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