"""
Visualization Experiments on the MIT-BIH Arrhythmia Dataset in Python.

Reference:
Matrix Profile III: The Matrix Profile Allows Visualization of Salient Subsequences in Massive Time Series
Chin-Chia Michael Yeh, Helga Van Herle, and Eamonn Keogh
https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from tsvisualize import TimeSeriesVisualizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
np.random.seed(0)


# Z-normalization to zero mean and unit variance.
def znormalize(subsequence):
    return (subsequence - np.mean(subsequence)) / np.std(subsequence)


if __name__ == '__main__':

    # Load record 106 from the MIT-BIH Arrhythmia Dataset.
    record = np.load('testdata/heartbeat.npz')

    # Load data and labelled indices.
    timeseries = record['timeseries']

    labelled_indices = record['labelled_indices']
    true_labels = record['true_labels']
    matrix_profile = record['matrix_profile']
    matrix_profile_indices = record['matrix_profile_indices']
    subsequence_length = int(record['subsequence_length'])

    # Select Z-normalized subsequences. These are subsequences chosen and labelled by an expert.
    subsequences = np.array([znormalize(timeseries[index: index + subsequence_length]) for index in labelled_indices])

    # Labels subsequences based on their proximity to the true labels.
    def get_labels(indices):

        def label(index):
            if np.min(np.abs(labelled_indices - index)) < subsequence_length:
                if true_labels[np.argmin(np.abs(labelled_indices - index))] == 2:
                    return 0
                else:
                    return 1
            else:
                return -1

        return np.array([label(index) for index in indices])

    # Project down to 2D space, and plot.
    projected_subsequences = PCA(n_components=2).fit_transform(subsequences)
    labels = get_labels(labelled_indices)
    plt.scatter(projected_subsequences[labels ==  0][:, 0], projected_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
    plt.scatter(projected_subsequences[labels ==  1][:, 0], projected_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
    plt.scatter(projected_subsequences[labels == -1][:, 0], projected_subsequences[labels == -1][:, 1], c='green', label='No Label')
    plt.legend()
    plt.title('PCA Plot with Expert-Extracted Subsequences')
    plt.show()

    # Select random subsequences, same in number as for the previous plot.
    num_points = len(true_labels)
    random_indices = np.random.permutation(timeseries.size - subsequence_length + 1)[:num_points]
    random_subsequences = np.array([znormalize(timeseries[index: index + subsequence_length]) for index in random_indices])

    # Project down to 2D space, and plot.
    projected_random_subsequences = PCA(n_components=2).fit_transform(random_subsequences)
    labels = get_labels(random_indices)
    plt.scatter(projected_random_subsequences[labels ==  0][:, 0], projected_random_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
    plt.scatter(projected_random_subsequences[labels ==  1][:, 0], projected_random_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
    plt.scatter(projected_random_subsequences[labels == -1][:, 0], projected_random_subsequences[labels == -1][:, 1], c='green', label='No Label')
    plt.legend()
    plt.title('PCA Plot with Randomly Sampled Subsequences')
    plt.show()

    # Select subsequences, chosen to minimize our MDL cost function!
    tsv = TimeSeriesVisualizer(timeseries, subsequence_length, discretization_bits=8, candidates_per_round=5,
                               matrix_profile_noise=0, matrix_profile_run_time=300)
    normalized_subsequences, subsequence_indices = tsv.select_subsequences()

    # Project down to 2D space, and plot.
    projected_chosen_subsequences = PCA(n_components=2).fit_transform(normalized_subsequences)
    labels = get_labels(subsequence_indices)
    plt.scatter(projected_chosen_subsequences[labels ==  0][:, 0], projected_chosen_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
    plt.scatter(projected_chosen_subsequences[labels ==  1][:, 0], projected_chosen_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
    plt.scatter(projected_chosen_subsequences[labels == -1][:, 0], projected_chosen_subsequences[labels == -1][:, 1], c='green', label='No Label')
    plt.legend()
    plt.title('PCA Plot with Specifically Selected Subsequences')
    plt.show()

    # Select subsequences, but using a precomputed matrix profile.
    tsv.original_matrix_profile = matrix_profile
    tsv.original_matrix_profile_indices = matrix_profile_indices

    normalized_subsequences, subsequence_indices = tsv.select_subsequences()

    # Project down to 2D space, and plot.
    projected_chosen_subsequences = PCA(n_components=2).fit_transform(normalized_subsequences)
    labels = get_labels(subsequence_indices)
    plt.scatter(projected_chosen_subsequences[labels ==  0][:, 0], projected_chosen_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
    plt.scatter(projected_chosen_subsequences[labels ==  1][:, 0], projected_chosen_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
    plt.scatter(projected_chosen_subsequences[labels == -1][:, 0], projected_chosen_subsequences[labels == -1][:, 1], c='green', label='No Label')
    plt.legend()
    plt.title('PCA Plot with Specifically Selected Subsequences \n (Precomputed Matrix Profile)')
    plt.show()

