import numpy as np

# Python 3.5's standard library has versions for these two in `statistics`
def mean(data, axis = 0):
    return data.sum(axis = axis) / data.shape[axis]

# TODO: check out whether this naive version is numerically sound
def variance(data, xbar = None, axis = 0):
    N = data.shape[axis]
    if xbar is None:
        xbar = mean(data, axis = axis)
    return ( (data - xbar) ** 2 ).sum(axis = axis) / (N - 1)

def autocorrelation_function(observable, k, xbar, measurement_variance):
    """
    Compute the autocorrelation function for an observable with the argument k.

    Sample mean and variance have to given.
    """

    if k == 0:
        if observable.ndim > 1:
            return np.ones(observable.shape[1])
        else:
            return 1.

    m = mean(observable[:-k] * observable[k:])
    return (m - xbar**2) / measurement_variance

def autocorrelation_analysis(observable,
                             xbar = None, measurement_variance = None):
    """
    Run complete autocorrelation analysis on the given time series.
    Returns the average, its error, the autocorrelation time, its error and the
    effective sampling size in a numpy array.
    """

    if xbar is None:
        xbar = mean(observable)
    if measurement_variance is None:
        measurement_variance = variance(observable, xbar = xbar)

    N = len(observable)
    if observable.ndim > 1:
        n = observable.shape[1]
        tau = np.empty(n)
        k_max = np.empty(n)
        tau[:] = .5
        for i in range(n):
            k = 1
            while k < 6 * tau[i] or k > N:
                tau[i] += autocorrelation_function(observable[:,i], k, xbar[i],
                                                   measurement_variance[i])
                k += 1

            k_max[i] = k
    else:
        tau = .5
        k = 1
        while k < 6 * tau or k > N:
            tau += autocorrelation_function(observable, k, xbar,
                                            measurement_variance)
            k += 1

        k_max = k

    N_eff = N / tau
    e_obs = measurement_variance / N_eff
    e_tau = np.sqrt(2 * (2*k_max + 1) / N)

    return np.array( (xbar, e_obs, tau, e_tau, N_eff) )

def binning_analysis(observable, bin_width, xbar = None, measurement_variance = None):
    """
    Computes binning \\tau and error estimate \\epsilon^2 from time series and
    given bin width, returns (\\tau, \\epsilon^2).

    If the bin width does not divide the series length, the series will be
    truncated to the next smallest length that is a multiple of the given width.
    If sample average and variance are already computed, they can be given as
    keyword arguments. It is not checked whether these values are actually
    correct.
    """

    k = bin_width
    N = series_length = len(observable)
    N_b = N // k

    bins = np.array([observable[i * k : (i + 1) * k] for i in range(N_b)])
    bin_averages = mean(bins, axis = 1)
    if xbar is None:
        xbar = mean(bin_averages)

    bin_variance = variance(bin_averages, xbar = xbar)
    if measurement_variance is None:
        measurement_variance = variance(observable, xbar = xbar)

    auto_correlation_time = k / 2 * (bin_variance / measurement_variance)
    binning_error = bin_variance / N_b
    return auto_correlation_time, binning_error

def jackknife_analysis(observable, bin_width, xbar = None):
    """
    Compute jackknife error for an observable.
    `observable` can be a two dimensional array in which case the analysis is
    performed for two or more time series of the same length simultaneously.

    If the bin width does not divide the series length, the series will be
    truncated to the next smallest length that is a multiple of the given width.
    If sample average and variance are already computed, they can be given as
    keyword arguments. It is not checked whether these values are actually
    correct.
    """

    k = bin_width
    N = series_length = len(observable)
    N_b = N // k

    bins = np.array([observable[i * k : (i + 1) * k] for i in range(N_b)])
    bin_averages = mean(bins, axis = 1)

    if xbar is None:
        xbar = mean(bin_averages)

    jackknife_averages = (N * xbar - k * bin_averages) / (N - k)
    jackknife_error = (N_b - 1) / float(N_b) \
                    * ( (jackknife_averages - xbar)**2 ).sum(axis = 0)
    return jackknife_error
