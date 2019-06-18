"""
Time-series Visualization with the Matrix Profile and Multidimensional Scaling.

Author: Ameya Daigavane

Reference:
Matrix Profile III: The Matrix Profile Allows Visualization of Salient Subsequences in Massive Time Series
Chin-Chia Michael Yeh, Helga Van Herle, and Eamonn Keogh
https://www.cs.ucr.edu/~eamonn/PID4481999_Matrix%20Profile_III.pdf
"""

from __future__ import division
import numpy as np
from matrixprofile import matrixProfile as mp
from sklearn.manifold import MDS


class MDL:
    @staticmethod
    def discretize(subsequence, num_bits):
        """
        :param subsequence: Subsequence to be discretized.
        :param num_bits: Number of bits used to represent each bit of the discretized subsequence.
        :return: Discretized subsequence.

        >>> s1 = np.array([1, 3, 4, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 4, 2, 6, 10, 11, 9, 3])
        >>> '%s' % MDL.discretize(s1, num_bits=4) # doctest: +NORMALIZE_WHITESPACE
        '[ 2 4 5 6 5 4 7 16 15 14 13 1 3 5 3 7 11 12 10 4]'
        >>> s2 = np.array([1, 3, 4, 5, 3])
        >>> '%s' % MDL.discretize(s2, num_bits=4) # doctest: +NORMALIZE_WHITESPACE
        '[ 1 9 12 16 9]'
        """
        subsequence_min = np.min(subsequence)
        subsequence_max = np.max(subsequence)
        return np.round((subsequence - subsequence_min) / (subsequence_max - subsequence_min)).astype(int) * (2 ** num_bits - 1) + 1

    @staticmethod
    def description_length(subsequence, num_bits):
        """
        :param subsequence: Subsequence - can be thought of as a vector of elements each represented with some number of bits.
        :param num_bits: Number of bits used to represent each element in this subsequence.
        :return: The description length of the subsequence. This corresponds to the space required to store this sequence uncompressed.

        >>> s1 = np.array([1, 3, 4, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 4, 2, 6, 10, 11, 9, 3])
        >>> '%0.2f' % MDL.description_length(s1, num_bits=4)
        '80.00'
        """
        return subsequence.shape[0] * num_bits

    @staticmethod
    def reduced_description_length(compressible_sequence, hypothesis_sequence, num_bits):
        """
        :param compressible_sequence: The subsequence we want to compress by representing it with the hypothesis subsequence.
        :param hypothesis_sequence: The subsequence we will use to compress the other sequence by noting dissimilarities.
        :param num_bits: The number of bits used to store each element in the compressed format.
        :return: The reduced description length of the subsequence. This corresponds to the space required to store the compressible sequence with the hypothesis sequence.

        >>> s1 = np.array([1, 3, 4, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 4, 2, 6, 10, 11, 9, 3])
        >>> s2 = np.array([1, 3, 3, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 3, 2, 6, 4, 3, 1, 0])
        >>> '%0.2f' % MDL.reduced_description_length(s1, s2, num_bits=4)
        '49.93'
        """
        return np.where((compressible_sequence != hypothesis_sequence))[0].size * (np.log2(compressible_sequence.shape[0]) + num_bits)


class TimeSeriesVisualizer:

    def __init__(self, sequence, subsequence_length, discretization_bits=6, candidates_per_round=10, std_noise=0):
        self.sequence = sequence
        self.sequence_length = self.sequence.shape[0]
        self.subsequence_length = subsequence_length

        # Number of bits used to discretize for MDL.
        self.num_bits = discretization_bits

        # How many candidates to consider every round.
        self.candidates_per_round = candidates_per_round

        # Compute the description length for individual subsequences.
        self.subsequence_description_length = MDL.description_length(self.sequence[:subsequence_length], self.num_bits)

        # Matrix profile - to be evaluated and filled in later.
        self.matrix_profile = None
        self.matrix_profile_indices = None
        self.std_noise = std_noise

        # The three sets that will be used to construct the overall list of subsequences.
        self.compressible_set = []
        self.hypothesis_set = []
        self.unexplored_set = set(np.arange(self.sequence_length - self.subsequence_length + 1))

    # Bit cost of the current state of the three sets.
    def bit_cost(self):

        # Add compressed cost for compressible set.
        bit_cost = 0
        for compressible_sequence_index in self.compressible_set:
            compressible_sequence = self.sequence[compressible_sequence_index]
            min_rdl = np.inf
            for hypothesis_sequence_index in self.hypothesis_set:
                hypothesis_sequence = self.sequence[hypothesis_sequence_index]
                rdl = MDL.reduced_description_length(compressible_sequence, hypothesis_sequence, self.num_bits)
                min_rdl = min(rdl, min_rdl)

            if min_rdl < np.inf:
                bit_cost += min_rdl

        # Add uncompressed cost for hypothesis and unexplored sets.
        bit_cost += len(self.hypothesis_set) * self.subsequence_description_length
        bit_cost += len(self.unexplored_set) * self.subsequence_description_length

        return bit_cost

    # Gets the next list of candidates to consider.
    def get_candidates(self, num_candidates):
        candidates = []
        matrix_profile = np.copy(self.matrix_profile)
        for _ in range(num_candidates):
            # If no remaining values in the matrix profile, quit.
            if np.min(matrix_profile) == np.inf:
                break

            # Get the position and corresponding subsequence of the smallest value from the matrix profile.
            candidate_index = np.argmin(matrix_profile)
            candidate = self.sequence[candidate_index: candidate_index + self.subsequence_length]

            # Discretize this.
            candidate = MDL.discretize(candidate, self.num_bits)
            candidates.append((candidate, candidate_index))

            # Mask out trivial matches from the matrix profile.
            mask_start = max(candidate_index - self.subsequence_length, 0)
            mask_end = min(candidate_index + self.subsequence_length, matrix_profile.size)
            for index in range(mask_start, mask_end):
                matrix_profile[index] = np.inf

        return candidates

    # Picks the best candidate according to the MDL criteria along with the type ('hypothesis'/'compressible') it belongs to.
    def best_candidate(self, candidates):
        best_bit_save = -np.inf
        best_candidate = None
        best_candidate_index = None

        for candidate, candidate_index in candidates:

            # Test as hypothesis.
            nearest_neighbour_index = int(self.matrix_profile_indices[candidate_index])
            nearest_neighbour = self.sequence[nearest_neighbour_index: nearest_neighbour_index + self.subsequence_length]
            nearest_neighbour = MDL.discretize(nearest_neighbour, self.num_bits)
            bit_save = MDL.description_length(nearest_neighbour, self.num_bits) - MDL.reduced_description_length(nearest_neighbour, candidate, self.num_bits)

            if bit_save > best_bit_save:
                best_bit_save = bit_save
                best_candidate = candidate
                best_candidate_index = candidate_index
                candidate_is_hypothesis = False

            # Test as compressible.
            for hypothesis_index in self.hypothesis_set:
                hypothesis = self.sequence[hypothesis_index]
                bit_save = self.description_length(candidate) - MDL.reduced_description_length(candidate, hypothesis, self.num_bits)

                if bit_save > best_bit_save:
                    best_bit_save = bit_save
                    best_candidate = candidate
                    best_candidate_index = candidate_index
                    candidate_is_hypothesis = True

        return best_candidate, best_candidate_index, candidate_is_hypothesis

    # Computes the matrix profile with the STOMP method, returning the indices as well.
    def get_matrix_profile(self):
        return mp.stomp(self.sequence, self.subsequence_length, std_noise=self.std_noise)

    # Selects subsequences to be used for the MDS plot.
    def select_subsequences(self):
        # Re-initialize sets.
        self.compressible_set = set()
        self.hypothesis_set = set()
        self.unexplored_set = set(np.arange(self.sequence_length - self.subsequence_length + 1))

        # Compute initial bit cost.
        bit_cost_old = self.bit_cost()

        # Compute matrix profile with indices. Noise correction for the matrix profile has been implemented.
        self.matrix_profile, self.matrix_profile_indices = self.get_matrix_profile()

        while True:
            # Get all the candidate subsequences.
            num_candidates = self.candidates_per_round
            candidates = self.get_candidates(num_candidates)

            # If no candidates, quit.
            if len(candidates) == 0:
                break

            # Choose the best according to the MDL criteria.
            candidate, candidate_index, is_hypothesis = self.best_candidate(candidates)

            # Remove candidate from unexplored.
            self.unexplored_set.remove(candidate_index)

            # Update matrix profile to remove trivial matches.
            mask_start = max(candidate_index - self.subsequence_length, 0)
            mask_end = min(candidate_index + self.subsequence_length, self.matrix_profile.size)
            for index in range(mask_start, mask_end):
                self.matrix_profile[index] = np.inf

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
        return list(self.compressible_set.union(self.hypothesis_set))

    # Fits the embedding for the MDS, and returns the transformed sequence, along with the subsequence indices!
    def fit_transform(self):
        embedding = MDS(n_components=2)
        subsequence_indices = self.select_subsequences()
        subsequences = [self.sequence[subsequence_index: subsequence_index + self.subsequence_length] for subsequence_index in subsequence_indices]
        transformed_sequence = embedding.fit_transform(subsequences)
        return transformed_sequence, subsequence_indices
