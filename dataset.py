# dataset.py

import torch
import numpy as np
from Bio.PDB import PDBParser


def get_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_id()[0] == " ":
                    residues.append(res)

    sequence = []
    for res in residues:
        sequence.append(res.get_resname())

    return sequence, residues


def sequence_to_features(pdb_file):
    sequence, _ = get_sequence_from_pdb(pdb_file)
    L = len(sequence)

    # 21 channels (simple encoding)
    features = torch.zeros((21, L, L))

    for i in range(L):
        for j in range(L):
            # simple identity feature
            features[0, i, j] = 1 if sequence[i] == sequence[j] else 0

    return features


def pdb_to_contact_map(pdb_file, threshold=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_id()[0] == " ":
                    residues.append(res)

    L = len(residues)
    contact_map = np.zeros((L, L))

    coords = []

    for res in residues:
        if "CB" in res:
            coords.append(res["CB"].get_coord())
        elif "CA" in res:
            coords.append(res["CA"].get_coord())
        else:
            coords.append(None)

    for i in range(L):
        for j in range(L):
            if coords[i] is not None and coords[j] is not None:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < threshold:
                    contact_map[i, j] = 1

    return torch.tensor(contact_map, dtype=torch.float32).unsqueeze(0)