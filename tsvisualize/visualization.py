"""
Time-series Visualization with the Matrix Profile.

Author: Ameya Daigavane
Reference:
Matrix Profile III: The Matrix Profile Allows Visualization of Salient Subsequences in Massive Time Series
Chin-Chia Michael Yeh, Helga Van Herle, and Eamonn Keogh
https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf
"""

from __future__ import division
import numpy as np
from matrixprofile import matrixProfile as mp
from mdl import description_length, reduced_description_length


class TimeSeriesVisualizer:
    """
    Constructs the matrix profile and optimizes for a MDL cost function when selecting subsequences for MDS.
    """
    def __init__(self, sequence, subsequence_length, discretization_bits=6, candidates_per_round=10, matrix_profile_noise=0, matrix_profile_run_time=None, method='pca'):
        self.sequence = np.array(sequence)
        self.sequence_length = self.sequence.shape[0]
        self.subsequence_length = subsequence_length
        self.num_subsequences = self.sequence_length - self.subsequence_length + 1

        # Visualization method.
        if method not in ['pca', 'tsne', 'mds']:
            raise ValueError('Visualization must be one of \'pca\', \'tsne\', \'mds\'.')
        self.visualization_method = method

        # Number of bits used to discretize for MDL.
        self.num_bits = discretization_bits

        # How many candidates to consider every round.
        self.candidates_per_round = candidates_per_round

        # The minimum and maximum of Z-normalized subsequences. Required for discretization.
        self.subsequence_min = np.inf
        self.subsequence_max = -np.inf

        # Matrix profile - to be evaluated and filled in later.
        self.original_matrix_profile = None
        self.original_matrix_profile_indices = None
        self.matrix_profile = None
        self.matrix_profile_indices = None
        self.std_noise = matrix_profile_noise
        self.matrix_profile_run_time = matrix_profile_run_time

        # The three sets that will be used to construct the overall list of subsequences.
        self.compressible_set = []
        self.hypothesis_set = []
        self.unexplored_set = set(np.arange(self.num_subsequences))

        self.actual_chosen = None

    # Z-normalize a subsequence.
    @staticmethod
    def znormalize(subsequence):
        """
        :param subsequence: Subsequence of a time-series, as a numpy array.
        :return: Z-normalized subsequence with zero mean and unit variance.
        """
        return (subsequence - np.mean(subsequence)) / np.std(subsequence)

    # Get the znormalized version of a subsequence at index.
    def get_znormalized_subsequence(self, index):
        subsequence = self.sequence[index: index + self.subsequence_length]
        return self.znormalize(subsequence)

    # Discretize a subsequence, using a fixed number of bits.
    def discretize(self, subsequence):
        """
        :param subsequence: Subsequence to be discretized, as a numpy array.
        :return: Discretized subsequence, with each element represented by a number of bits.
        """
        return np.round(((self.znormalize(subsequence) - self.subsequence_min) / (self.subsequence_max - self.subsequence_min)) * (2 ** self.num_bits - 1)).astype(int) + 1

    # Get discretized subsequence at index.
    def get_discrete_subsequence(self, index):
        return self.discretize(self.sequence[index: index + self.subsequence_length])

    # Compute Z-normalized subsequence minimum and maximum, for discretization later.
    def set_discretization_thresholds(self):
        for index in range(self.num_subsequences):
            subsequence = self.get_znormalized_subsequence(index)
            self.subsequence_min = min(self.subsequence_min, np.min(subsequence))
            self.subsequence_max = max(self.subsequence_max, np.max(subsequence))

    # Bit cost of the current state of the three sets.
    def bit_cost(self):
        bit_cost = 0

        # Add compressed cost for compressible set.
        for compressible_sequence_index in self.compressible_set:
            compressible_sequence = self.get_discrete_subsequence(compressible_sequence_index)

            min_rdl = np.inf
            for hypothesis_sequence_index in self.hypothesis_set:
                hypothesis_sequence = self.get_discrete_subsequence(hypothesis_sequence_index)
                rdl = reduced_description_length(compressible_sequence, hypothesis_sequence, self.num_bits) + np.log2(len(self.hypothesis_set))
                min_rdl = min(rdl, min_rdl)

            if min_rdl < np.inf:
                bit_cost += min_rdl

        # Add uncompressed cost for hypothesis and unexplored sets.
        bit_cost += (len(self.hypothesis_set) + len(self.unexplored_set)) * description_length(self.get_discrete_subsequence(0), self.num_bits)

        return bit_cost

    # Gets the next list of candidates to consider.
    def get_candidates(self, num_candidates):
        candidates = []
        for _ in range(num_candidates):

            # If no remaining values in the matrix profile, quit.
            if np.min(self.matrix_profile) == np.inf:
                break

            # Get the position and corresponding subsequence of the smallest value from the matrix profile.
            candidate_index = np.argmin(self.matrix_profile)

            # Discretize this, and add to candidates set.
            candidate = self.get_discrete_subsequence(candidate_index)
            candidates.append((candidate, candidate_index))

            # Mask out trivial matches from the matrix profile.
            mask_start = max(candidate_index - self.subsequence_length + 1, 0)
            mask_end = min(candidate_index + self.subsequence_length, self.matrix_profile.size)
            self.matrix_profile[mask_start: mask_end] = np.inf

        return candidates

    # Picks the best candidate along with its type ('hypothesis'/'compressible') according to the MDL criteria.
    def best_candidate(self, candidates):
        best_bit_save = -np.inf
        best_candidate = None
        best_candidate_index = None

        for candidate, candidate_index in candidates:

            # Test as hypothesis, by looking at nearest neighbour.
            nearest_neighbour = self.get_discrete_subsequence(int(self.matrix_profile_indices[candidate_index]))
            bit_save = description_length(nearest_neighbour, self.num_bits) - reduced_description_length(nearest_neighbour, candidate, self.num_bits)

            if bit_save > best_bit_save:
                best_bit_save = bit_save
                best_candidate = candidate
                best_candidate_index = candidate_index
                candidate_is_hypothesis = True

            # Test as compressible, by looking at hypotheses.
            for hypothesis_index in self.hypothesis_set:
                hypothesis = self.get_discrete_subsequence(hypothesis_index)
                bit_save = description_length(candidate, self.num_bits) - reduced_description_length(candidate, hypothesis, self.num_bits)

                if bit_save > best_bit_save:
                    best_bit_save = bit_save
                    best_candidate = candidate
                    best_candidate_index = candidate_index
                    candidate_is_hypothesis = False

        return best_candidate, best_candidate_index, candidate_is_hypothesis

    # Computes the matrix profile with the SCRIMP method, returning the indices as well.
    # Noise correction for the matrix profile has been implemented.
    def get_matrix_profile(self):
        return mp.scrimp_plus_plus(self.sequence, self.subsequence_length, step_size_fraction=0.25, std_noise=self.std_noise, runtime=self.matrix_profile_run_time, exclusion_zone_fraction=1)

    # Selects subsequences to be used for the MDS plot.
    def select_subsequences(self):

        # Set thresholds for discretization.
        self.set_discretization_thresholds()

        # Re-initialize sets.
        self.compressible_set = set()
        self.hypothesis_set = set()
        self.unexplored_set = set(np.arange(self.num_subsequences))

        # Compute initial bit cost.
        bit_cost_old = self.bit_cost()

        # Load matrix profile again, if we've already computed.
        if self.original_matrix_profile is None:
            self.matrix_profile, self.matrix_profile_indices = self.get_matrix_profile()
            self.original_matrix_profile = np.copy(self.matrix_profile)
            self.original_matrix_profile_indices = np.copy(self.matrix_profile_indices)
        else:
            self.matrix_profile = self.original_matrix_profile
            self.matrix_profile_indices = self.original_matrix_profile_indices

        while True:
            # Get all the candidate subsequences.
            num_candidates = self.candidates_per_round
            candidates = self.get_candidates(num_candidates)

            # If no candidates, quit.
            if len(candidates) == 0:
                break

            # Choose the best according to the MDL criteria.
            candidate, candidate_index, is_hypothesis = self.best_candidate(candidates)

            # Update matrix profile to remove trivial matches.
            mask_start = max(candidate_index - self.subsequence_length + 1, 0)
            mask_end = min(candidate_index + self.subsequence_length, self.matrix_profile.size)
            self.matrix_profile[mask_start: mask_end] = np.inf

            # Remove candidate from unexplored.
            self.unexplored_set.remove(candidate_index)

            # Depending on the type, add to the right set.
            if is_hypothesis:
                self.hypothesis_set.add(candidate_index)
            else:
                self.compressible_set.add(candidate_index)

                # Compute new bit cost.
                bit_cost_new = self.bit_cost()

                if bit_cost_new > bit_cost_old:
                    break
                else:
                    bit_cost_old = bit_cost_new

        # Return the union of the compressible set and the hypothesis set.
        subsequence_indices = list(self.compressible_set.union(self.hypothesis_set))
        normalized_subsequences = np.array([self.get_znormalized_subsequence(index) for index in subsequence_indices])

        return normalized_subsequences, subsequence_indices

    # Fits the embedding for the visualization method, using the selected Z-normalized subsequences.
    # Returns the transformed subsequences, along with their actual indices!
    def fit_transform(self):

        # Choose method and required embedding.
        if self.visualization_method == 'pca':
            from sklearn.decomposition import PCA
            embedding = PCA(n_components=2)

        elif self.visualization_method == 'mds':
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2)

        elif self.visualization_method == 'mds':
            from sklearn.manifold import MDS
            embedding = MDS(n_components=2)

        normalized_subsequences, subsequence_indices = self.select_subsequences()
        transformed_subsequences = embedding.fit_transform(normalized_subsequences)
        return transformed_subsequences, subsequence_indices