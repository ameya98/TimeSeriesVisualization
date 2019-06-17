"""
Recreating Visualization Experiments from the MIT-BIH Arrhythmia Dataset in Python.

Reference:
Matrix Profile III: The Matrix Profile Allows Visualization of Salient Subsequences in Massive Time Series
Chin-Chia Michael Yeh, Helga Van Herle, and Eamonn Keogh
https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from tsvisualize import TimeSeriesVisualizer
import scipy.io
from sklearn.manifold import MDS


if __name__ == '__main__':

    # Load time-series. This is from record 106 from the MIT-BIH Arrhythmia Dataset.
    mat = scipy.io.loadmat('testdata/106.mat')

    # The actual time-series of the heartbeat.
    timeseries = mat['data'].flatten()

    # Window size of the heartbeats.
    windowsize = np.squeeze(mat['winSize']).astype(int)

    # Recreate Figure 1, from the paper.
    subsequence_indices = mat['coordIdx'].flatten()
    labels = mat['coordLab'].flatten()
    normal_heartbeats = mat['coord'][np.where(labels == 3)[0], :]
    abnormal_heartbeats = mat['coord'][np.where(labels == 4)[0], :]

    plt.scatter(normal_heartbeats[:, 0], normal_heartbeats[:, 1], c='gold')
    plt.scatter(abnormal_heartbeats[:, 0], abnormal_heartbeats[:, 1], c='darkblue')
    plt.title('MDS Plot with Correctly Sampled Subsequences')
    plt.show()

    # Recreate Figure 2, from the paper.
    num_points = labels.size
    random_indices = np.random.permutation(timeseries.size - windowsize + 1)[:num_points]
    random_subsequences = np.zeros((num_points, windowsize))
    random_labels = np.zeros(num_points)

    for actual_index, random_index in enumerate(random_indices):
        # Get the subsequence.
        random_subsequences[actual_index] = timeseries[random_index: random_index + windowsize]

        # Standardize to zero mean and unit variance.
        random_subsequences[actual_index] -= np.mean(random_subsequences[actual_index])
        random_subsequences[actual_index] /= np.std(random_subsequences[actual_index])

        # Get the label for this subsequence.
        if np.any(np.abs(subsequence_indices - random_index) < 0.2 * windowsize):
            random_labels[actual_index] = labels[np.argmin(np.abs(subsequence_indices - random_index))]

    transformed_subsequences = MDS(n_components=2).fit_transform(random_subsequences)
    normal_subsequences = transformed_subsequences[np.where(random_labels == 3)[0]]
    abnormal_subsequences = transformed_subsequences[np.where(random_labels == 4)[0]]
    overlap_subsequences = transformed_subsequences[np.where(random_labels == 0)[0]]

    plt.scatter(normal_subsequences[:, 0], normal_subsequences[:, 1], c='gold')
    plt.scatter(abnormal_subsequences[:, 0], abnormal_subsequences[:, 1], c='darkblue')
    plt.scatter(overlap_subsequences[:, 0], overlap_subsequences[:, 1], c='green')
    plt.title('MDS Plot with Randomly Sampled Subsequences')
    plt.show()
