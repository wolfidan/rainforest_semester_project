import numpy as np

# first some helper functions
@njit
def _find_group_indices(groups):
    """
    Compute start and end indices for contiguous groups.

    This function assumes that `groups` is a 1D array of integer group
    identifiers that are already sorted and contiguous. It returns the
    start and end indices for each unique group.

    Parameters
    ----------
    groups : ndarray of shape (n_samples,)
        Sorted array of group identifiers.

    Returns
    -------
    start_idx : ndarray of shape (n_groups,)
        Start indices of each group in `groups`.

    end_idx : ndarray of shape (n_groups,)
        End indices (exclusive) of each group in `groups`.
    """
    indices = np.searchsorted(groups, np.arange(groups[-1]+1)) # might have to be +1? because arange not inclusive 
    return indices[:-1], np.append(indices[1:], len(groups))

@njit
def _process_groups_fast(features, start_idx, end_idx, padded_sequences):
    """
    Populate a padded sequence array using precomputed group indices.

    For each group, feature vectors are copied into a preallocated
    padded array. Remaining entries are left as zero-padding.

    Parameters
    ----------
    features : ndarray of shape (n_samples, n_features)
        Input feature matrix sorted by group.
    start_idx : ndarray of shape (n_groups,)
        Start indices for each group.
    end_idx : ndarray of shape (n_groups,)
        End indices (exclusive) for each group.
    padded_sequences : ndarray of shape (n_groups, max_seq_len, n_features)
        Preallocated output array for padded sequences.

    Returns
    -------
    padded_sequences : ndarray
        The input array filled with grouped feature sequences.
    """
    for i in range(len(start_idx)):
        start, end = start_idx[i], end_idx[i]  # Get start and end indices
        group = features[start:end]  # Extract group (already sorted)
        seq_len = group.shape[0]  # Sequence length
        padded_sequences[i, :seq_len, :] = group  # Copy with padding

    return padded_sequences