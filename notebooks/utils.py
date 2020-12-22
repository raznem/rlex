import numpy as np
import matplotlib.pyplot as plt

def plot_evals_list(evals_list, color="blue", *args, **kwargs):
    outputs = np.array(evals_list)
    indexes = outputs[0, :, 0]
    values = outputs[:, :, 1]
    scores = np.hstack((indexes.reshape(-1, 1), values.T))

    mean_scores = np.mean(scores[:, 1:], axis=1)
    stds = np.std(scores[:, 1:], axis=1)
    plt.plot(scores[:, 0], mean_scores, color=color)
    plt.fill_between(scores[:, 0], mean_scores - stds, mean_scores + stds, alpha=0.2, color=color)