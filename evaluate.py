import numpy as np


def precision_at_L5(prediction, truth):

    prediction = prediction.detach().numpy()
    truth = truth.numpy()

    L = prediction.shape[0]
    top_k = L // 5

    pairs = []

    for i in range(L):
        for j in range(L):

            # ONLY long-range contacts
            if abs(i - j) >= 24:
                pairs.append((prediction[i][j], truth[i][j]))

    # Sort by predicted probability
    pairs.sort(key=lambda x: x[0], reverse=True)

    top_pairs = pairs[:top_k]

    correct = sum(1 for p in top_pairs if p[1] == 1)

    return correct / top_k