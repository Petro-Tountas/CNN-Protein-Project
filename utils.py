import torch


def mask_diagonal(matrix):

    L = matrix.shape[-1]

    for i in range(L):
        matrix[:,:,i,i] = 0

    return matrix