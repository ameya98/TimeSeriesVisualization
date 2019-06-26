# TimeSeriesVisualization
Visualization of Time-Series with the Matrix Profile by Salience Subsequence Selection.

<p align="center">
  <img width="460" height="300" src="/images/tsne_precomputed.png" />
</p>

## Introduction
**tsvisualize** is a Python implementation of the algorithm referenced in [1], with the additional capability to create t-SNE, PCA, and metric MDS plots in two dimensions.

Further, this example is based on the first experiment in [1], and the data is taken from the supporting [page](https://sites.google.com/site/salientsubs/) of this publication.
The complete code for this example is available as `visualization_arrythmia.py` in the `tests` directory.
```bash
pip install tsvisualize
```

A variety of dimensionality-reducing techniques are extremely popular for exploratory data analysis, data visualization and as preprocessing steps for other algorithms (such as in classification tasks).
At a first glance, this seems applicable to time-series as well, where we want to capture similar regions across the entire time-series. 
We can do this by considering subsequences of the time-series, and clustering these. 
However, since we do not know apriori where our regions of interest are, we can simply select all subsequences of a given length from the original time-series, and apply our favorite dimensionality reduction technique on these.
Time-series tend to be extremely large though and most of these techniques scale non-linearly with the size of the input. This means applying these techniques to the set of all subsequences will take forever. 
Instead, what if we selected a random subset of the subsequences? Let's try this out on our data!

For our example here, we have access to expert-labelled and selected heartbeat subsequences from the MIT-BIH Arrhythmia Dataset [2]. 

First, import some useful packages.
```python
import matplotlib.pyplot as plt
import numpy as np
from tsvisualize import TimeSeriesVisualizer
from sklearn.manifold import TSNE
```

Load the time-series, as well as some auxiliary data.
```python
# Load record 106 from the MIT-BIH Arrhythmia Dataset.
record = np.load('testdata/heartbeat.npz')

# Load data and labelled indices.
timeseries = record['timeseries']
labelled_indices = record['labelled_indices']
true_labels = record['true_labels']
matrix_profile = record['matrix_profile']
matrix_profile_indices = record['matrix_profile_indices']
subsequence_length = int(record['subsequence_length'])
```

Obtaining all the subsequences of fixed length from this timeseries, and Z-normalizing them:
```python
# Select Z-normalized subsequences. These are subsequences chosen and labelled by an expert.
subsequences = np.array([znormalize(timeseries[index: index + subsequence_length]) for index in labelled_indices])
```
Note that the definitions of some functions here are in `visualization_arrythmia.py` in the `tests` directory.

If we project these subsequences down to 2D, and plot:
```python
# Project down to 2D space, and plot.
projected_subsequences = PCA(n_components=2).fit_transform(subsequences)
labels = get_labels(labelled_indices)
plt.scatter(projected_subsequences[labels ==  0][:, 0], projected_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
plt.scatter(projected_subsequences[labels ==  1][:, 0], projected_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
plt.scatter(projected_subsequences[labels == -1][:, 0], projected_subsequences[labels == -1][:, 1], c='green', label='No Label')
plt.legend()
plt.title('t-SNE Plot with Expert-Extracted Subsequences')
plt.show()
```
<p align="center">
  <img width="460" height="300" src="/images/pca_expert.png" />
</p>

The separation between abnormal heartbeats and normal heartbeats is clear! This time-series is 650000 time-steps long, and we've got an effective summary of the kind of data that we have in this time-series.
Note that the labels have been supplied only for clarity in the plots here, but they are not required by the dimensionality reduction technique.
 
What if we didn't have any ideas about where the interesting phenomena (like the heartbeats here) are, and we selected a random subset of the subsequences to create a similar plot?
```python
# Select random subsequences, same in number as for the previous plot.
num_points = true_labels.size
random_indices = np.random.permutation(timeseries.size - subsequence_length + 1)[:num_points]
random_subsequences = np.array([znormalize(timeseries[index: index + subsequence_length]) for index in random_indices])

# Project down to 2D space, and plot.
projected_random_subsequences = PCA(n_components=2).fit_transform(random_subsequences)
labels = get_labels(random_indices)
plt.scatter(projected_random_subsequences[labels ==  0][:, 0], projected_random_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
plt.scatter(projected_random_subsequences[labels ==  1][:, 0], projected_random_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
plt.scatter(projected_random_subsequences[labels == -1][:, 0], projected_random_subsequences[labels == -1][:, 1], c='green', label='No Label')
plt.legend()
plt.title('t-SNE Plot with Randomly Sampled Subsequences')
plt.show()
```
<p align="center">
  <img width="460" height="300" src="/images/pca_random.png" />
</p>

Whoops. This doesn't look anything as useful. The labels are added based on proximity to the actually labelled subsequences. Subsequences which do not have any overlap with any of the labelled subsequences are given no label.
Unfortunately, all of the subsequences here are mixed together, and it is really hard to see any meaningful structure.

Was this all by chance? If you repeat this experiment, you will not get much better results, even if you use the entire time-series. 
This has to do with the fact that subsequences of time-series tend to have significant correlation with adjacent subsequences, which results in the phenomena of trivial matches.
All of this means that there is significant redundancy within these subsequences, which makes it hard to separate them in just 2 dimensions.

We can solve this by selecting only salient subsequences, according to a cost function based on the minimum description length of the time-series. The details are in [1], but the idea is to select subsequences that can explain other subsequences (by encoding the difference).
**tsvisualize** provides exactly this functionality, with *.select_subsequences()*.

```python
tsv = TimeSeriesVisualizer(timeseries, subsequence_length, discretization_bits=8, candidates_per_round=5,
                           matrix_profile_noise=0, matrix_profile_run_time=300)

# Select subsequences, chosen to minimize our MDL cost function!                           
normalized_subsequences, subsequence_indices = tsv.select_subsequences()

# Project down to 2D space, and plot.
projected_chosen_subsequences = PCA(n_components=2).fit_transform(normalized_subsequences)
labels = get_labels(subsequence_indices)
plt.scatter(projected_chosen_subsequences[labels ==  0][:, 0], projected_chosen_subsequences[labels ==  0][:, 1], c='gold', label='Normal Heartbeat')
plt.scatter(projected_chosen_subsequences[labels ==  1][:, 0], projected_chosen_subsequences[labels ==  1][:, 1], c='darkblue', label='Abnormal Heartbeat')
plt.scatter(projected_chosen_subsequences[labels == -1][:, 0], projected_chosen_subsequences[labels == -1][:, 1], c='green', label='No Label')
plt.legend()
plt.title('t-SNE Plot with Specifically Selected Subsequences')
plt.show()
```

<p align="center">
  <img width="460" height="300" src="/images/pca_actual.png" />
</p>

We get a significantly more meaningful plot - the abnormalities are all on the outside. The cluster of normal heartbeats are all homogeneous, except one. Note that the matrix profile can take a long time to compute, so we allow termination of the matrix profile computation after a fixed amount of time (5 minutes in the above example.) to get an approximation of it.

**tsvisualize** uses the SCRIMP++ algorithm to compute the matrix profile. 
If you have a faster implementation of the matrix profile, or even better, a precomputed matrix profile (like we do here), we can supply that directly, too. This speeds up the selection process significantly.
```python
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
plt.title('t-SNE Plot with Specifically Selected Subsequences \n (Precomputed Matrix Profile)')
plt.show()
```
<p align="center">
  <img width="460" height="300" src="/images/pca_precomputed.png" />
</p>

This is an even better plot! The abnormal heartbeats are pretty close to each other, barring one. The clusters themselves are pretty homogeneous too.

## Usage
The **TimeSeriesVisualizer** class provides all of the functionality here.
```python
TimeSeriesVisualizer(sequence, subsequence_length, discretization_bits=6, candidates_per_round=10, matrix_profile_noise=0, matrix_profile_run_time=None, method='pca')
```
* **sequence**: The timeseries to analyse.
* **subsequence_length**: Length of subsequences to extract.
* **discretization_bits**: Number of bits to represent discretized subsequences, in the MDL cost function.
* **candidates_per_round**: Number of candidates to select from the matrix profile. This generally affects runtime.
* **matrix_profile_noise**: Noise standard deviation which is a correction for the matrix profile. See [4].
* **matrix_profile_run_time**: How many seconds to spend evaluating the matrix profile.
* **method**: Visualization method. Choose from PCA ('pca'), metric MDS ('mds'), or t-SNE ('tsne').

The *.select_subsequences()* method returns a tuple of normalized subsequences (a 2D numpy array), and their actual indices in the timeseries (a 1D numpy array).
```python
normalized_subsequences, subsequence_indices = tsv.select_subsequences()
```

The *.fit_tranform()* method returns a tuple of the transformed subsequences (a 2D numpy array), and their actual indices in the timeseries (a 1D numpy array).
It first selects the subsequences and then applies the required dimensionality reducing embedding.
```python
transformed_subsequences, subsequence_indices = tsv.fit_transform()
```

## References
[1] **Matrix Profile III: The Matrix Profile Allows Visualization of Salient Subsequences in Massive Time Series**    
Chin-Chia Michael Yeh, Helga Van Herle, and Eamonn Keogh. IEEE Industrial Conference on Data Mining, 2016.

[2] **The Impact of the MIT-BIH Arrhythmia Database**  
Moody GB and Mark RG. IEEE Engineering in Medicine and Biology, 2001.

[3] **Clustering of Time Series Subsequences is Meaningless: Implications for Previous and Future Research**  
Eamonn Keogh and Jessica Lin. Knowledge and Information Systems, 2005.

[4] **Eliminating Noise in the Matrix Profile**
Dieter De Paepe, Olivier Janssens and Sofie Van Hoecke. International Conference on Pattern Recognition Applications and Methods, 2019.