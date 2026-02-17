import numpy as np
from collections import OrderedDict
from scipy.stats import energy_distance


def perfscores(est_data, ref_data, bounds=None, array=False):
    """
    Computes a set of precipitation performance scores, on different data ranges.
    The scores are
        - scatter: 0.5 * (Qw84(x) - Qw16(x)), where Qw is a quantile weighted
          by ref_data / sum(ref_data) and x is est_data / ref_data in dB scale
        - RMSE: root mean  square error (linear error)
        - bias:  (ME/mean(ref_data) + 1) in dB
        - ED: the energy distance which is a measure of the distance between
          two distributions (https://en.wikipedia.org/wiki/Energy_distance)

    Parameters
    ----------
    est_data : ndarray
        array of estimates (ex. precip from QPE)
    ref_data : ndarray
        array of reference (ex. precip from gauge)
    bounds : list (optional)
        list of bounds on ref_data for which to compute the error metrics,
        by default all data will be used (unbounded), note that even if you
        prescribe bounds the scores for the overall data will always be
        added in the output
    array: boolean (optional)
        Whether or not to convert the output dict to a numpy array

    Returns
    -------
    all_metrics : dict or ndarray
        a dictionary containing all the scores, organized in the following way
        all_metrics[bound][score]
    """
    all_metrics = OrderedDict()

    valid = np.logical_and(est_data >= 0, ref_data >= 0)
    est_data = est_data[valid > 0]
    ref_data = ref_data[valid > 0]

    est = est_data
    ref = ref_data

    all_metrics["all"] = _perfscores(est, ref)

    if bounds is not None:
        for i in range(len(bounds) - 1):
            bound_str = "{:2.1f}-{:2.1f}".format(bounds[i], bounds[i + 1])
            cond = np.logical_and(ref_data < bounds[i + 1], ref_data >= bounds[i])
            if np.sum(cond) > 0:
                est = est_data[cond]
                ref = ref_data[cond]

                all_metrics[bound_str] = _perfscores(est, ref)

    if array:
        arr = []
        for k in all_metrics:
            arr.append(list(all_metrics[k].values()))
        arr = np.array(arr)
        all_metrics = np.array(arr)

    return all_metrics


def _perfscores(est_data, ref_data, doublecond_thresh=0.1):
    """An unbounded version of the previous function"""
    doublecond = np.logical_and(
        ref_data > doublecond_thresh, est_data > doublecond_thresh
    )
    rmse = np.sqrt(np.nanmean((est_data[doublecond] - ref_data[doublecond]) ** 2))
    db_err = 10 * np.log10(est_data[doublecond] / ref_data[doublecond])
    weights = ref_data[doublecond] / np.sum(ref_data[doublecond])
    scatter = 0.5 * (quantile(db_err, weights, 0.84) - quantile(db_err, weights, 0.16))
    bias_db = 10 * np.log10(np.sum(est_data[doublecond]) / np.sum(ref_data[doublecond]))
    ed = energy_distance(
        est_data[np.isfinite(est_data)], ref_data[np.isfinite(est_data)]
    )

    mest = np.nanmean(est_data[doublecond])
    mref = np.nanmean(ref_data[doublecond])
    stdest = np.nanstd(est_data[doublecond])
    stdref = np.nanstd(ref_data[doublecond])

    metrics = {
        "RMSE": rmse,
        "scatter": scatter,
        "logBias": bias_db,
        "ED": ed,
        "N": len(ref_data[doublecond]),
        "N_all": len(ref_data),
        "est_mean": mest,
        "ref_mean": mref,
        "est_std": stdest,
        "ref_std": stdref,
    }

    return metrics


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quantile > 1.0) or (quantile < 0.0):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def quantile(data, weights, quantile):
    """
    Weighted quantile of an array with respect to the last axis.

    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last
        axis of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile : float
        The output value.
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    if nd == 0:
        TypeError("data must have at least one dimension")
    elif nd == 1:
        return quantile_1D(data, weights, quantile)
    elif nd > 1:
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1D, -1, imr, weights, quantile)
        return result.reshape(n[:-1])
