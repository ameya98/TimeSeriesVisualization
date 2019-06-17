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
