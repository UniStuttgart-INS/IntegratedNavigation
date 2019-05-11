# Author: Nikolay Mayorov <nikolay.mayorov@zoho.com>

from __future__ import division
from collections import OrderedDict
import numpy as np
from scipy.optimize import nnls


def allan_variance(x, dt=1, min_cluster_size=1, min_cluster_count='auto',
                   n_clusters=100, input_type="increment"):
    """Compute Allan variance (AVAR).

    Consider an underlying measurement y(t). Our sensors output integrals of
    y(t) over successive time intervals of length dt. These measurements
    x(k * dt) form the input to this function.

    Allan variance is defined for different averaging times tau = m * dt as
    follows::

        AVAR(tau) = 1/2 * <(Y(k + m) - Y(k))>,

    where Y(j) is the time average value of y(t) over [k * dt, (k + m) * dt]
    (call it a cluster), and < ... > means averaging over different clusters.
    If we define X(j) being an integral of x(s) from 0 to dt * j,
    we can rewrite the AVAR as  follows::

        AVAR(tau) = 1/(2 * tau**2) * <X(k + 2 * m) - 2 * X(k + m) + X(k)>

    We implement < ... > by averaging over different clusters of a given sample
    with overlapping, and X(j) is readily available from x.

    Parameters
    ----------
    x : ndarray, shape (n, ...)
        Integrating sensor readings, i. e. its cumulative sum gives an
        integral of a signal. Assumed to vary along the 0-th axis.
    dt : float, optional
        Sampling period. Default is 1.
    min_cluster_size : int, optional
        Minimum size of a cluster to use. Determines a lower bound on the
        averaging time as ``dt * min_cluster_size``. Default is 1.
    min_cluster_count : int or 'auto', optional
        Minimum number of clusters required to compute the average. Determines
        an upper bound of the averaging time as
        ``dt * (n - min_cluster_count) // 2``. If 'auto' (default) it is taken
        to be ``min(1000, n - 2)``
    n_clusters : int, optional
        Number of clusters to compute Allan variance for. The averaging times
        will be spread approximately uniform in a log scale. Default is 100.
    input_type : 'increment' or 'mean', optional
        How to interpret the input data. If 'increment' (default), then
        `x` is assumed to contain integral increments over successive time
        intervals (as described above). If 'mean', then `x` is assumed to
        contain mean values of y over successive time intervals, i.e.
        increments divided by `dt`.

    Returns
    -------
    tau : ndarray
        Averaging times for which Allan variance was computed, 1-d array.
    avar : ndarray
        Values of AVAR. The 0-th dimension is the same as for `tau`. The
        trailing dimensions match ones for `x`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Allan_variance
    """
    if input_type not in ('increment', 'mean'):
        raise ValueError("`input_type` must be either 'increment' or 'mean'.")

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    X = np.cumsum(x, axis=0)

    if min_cluster_count == 'auto':
        min_cluster_count = min(1000, n - 2)

    log_min = np.log2(min_cluster_size)
    log_max = np.log2((n - min_cluster_count) // 2)

    cluster_sizes = np.logspace(log_min, log_max, n_clusters, base=2)
    cluster_sizes = np.unique(np.round(cluster_sizes)).astype(np.int64)

    avar = np.empty(cluster_sizes.shape + X.shape[1:])
    for i, k in enumerate(cluster_sizes):
        c = X[2*k:] - 2 * X[k:-k] + X[:-2*k]
        avar[i] = np.mean(c**2, axis=0) / k / k

    if input_type == 'increment':
        avar *= 0.5 / dt**2
    elif input_type == 'mean':
        avar *= 0.5

    return cluster_sizes * dt, avar

