import numpy as np


def precision_at_L5(prediction, truth):

    # Flatten matrices
    prediction = prediction.detach().numpy().flatten()
    truth = truth.numpy().flatten()

    # Compute protein length
    L = int(np.sqrt(len(prediction)))

    # Top L/5 predictions
    top_k = L // 5

    # Sort predictions from highest to lowest probability
    sorted_indices = np.argsort(prediction)[::-1]

    top_indices = sorted_indices[:top_k]

    # Count correct predictions
    correct = 0

    for idx in top_indices:
        if truth[idx] == 1:
            correct += 1

    return correct / top_k