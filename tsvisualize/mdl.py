import numpy as np


def description_length(subsequence, num_bits):
    """
    :param subsequence: Subsequence - a vector of elements each represented with some number of bits.
    :param num_bits: Number of bits used to represent each element in this subsequence.
    :return: The description length of the subsequence. This corresponds to the space required to store this sequence uncompressed.

    >>> s1 = np.array([1, 3, 4, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 4, 2, 6, 10, 11, 9, 3])
    >>> '%0.2f' % description_length(s1, num_bits=4)
    '80.00'
    """
    return subsequence.shape[0] * num_bits


def reduced_description_length(compressible_sequence, hypothesis_sequence, num_bits):
    """
    :param compressible_sequence: The subsequence we want to compress by representing it with the hypothesis subsequence.
    :param hypothesis_sequence: The subsequence we will use to compress the other sequence by noting dissimilarities.
    :param num_bits: The number of bits used to store each element in the compressed format.
    :return: The reduced description length of the subsequence. This corresponds to the space required to store the compressible sequence with the hypothesis sequence.

    >>> s1 = np.array([1, 3, 4, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 4, 2, 6, 10, 11, 9, 3])
    >>> s2 = np.array([1, 3, 3, 5, 4, 3, 6, 15, 14, 13, 12, 0, 2, 3, 2, 6, 4, 3, 1, 0])
    >>> '%0.2f' % reduced_description_length(s1, s2, num_bits=4)
    '49.93'
    """
    return np.where(compressible_sequence != hypothesis_sequence)[0].size * (np.log2(compressible_sequence.shape[0]) + num_bits)