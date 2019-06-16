import matplotlib.pyplot as plt
import numpy as np
from visualization import TimeSeriesVisualizer

if __name__ == '__main__':

    # Construct artificial time-series of given size.
    size = 1000
    piece_length = size // 3
    piece1 = np.sin(np.arange(0, piece_length))
    piece2 = 3 + np.sin(np.arange(0, piece_length))
    piece3 = np.sin(np.arange(0, piece_length + size % 3))
    timeseries = np.hstack((piece1, piece2, piece3))[:size] + np.random.randn(size) * 0.2

    # Get transformed time-series, as well as indexes of the subsequences used.
    timeseries_transformed, mds_indices = TimeSeriesVisualizer(timeseries, 10).fit_transform()

    # This picks the right color for the points in the plots.
    def colorpicker(index):
        return 'red' if piece_length <= index < 2 * piece_length else 'blue'

    # Plot time-series as well as MDS plot.
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(timeseries)
    ax[0].set_ylabel('Original Time-series')

    mds_colors = map(colorpicker, mds_indices)
    ax[1].scatter(timeseries_transformed[:, 0], timeseries_transformed[:, 1], color=mds_colors)
    ax[1].set_ylabel('MDS Plot')

    plt.title('Time-Series Visualization', y=-0.35)
    plt.show()