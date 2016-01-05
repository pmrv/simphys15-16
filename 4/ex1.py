from matplotlib import pyplot as plt
import numpy as np
import pickle

import error

with open("series.dat") as fdata:
    series = pickle.load(fdata).T

averages  = error.mean(series)
variances = error.variance(series)

# jackknife stuff
jackknife_errors = np.array([error.jackknife_analysis(series, k, averages)
                                for k in range(1, 2001)])
plt.subplot(211)
plt.plot(jackknife_errors.T[0])
plt.plot(jackknife_errors.T[1])
plt.plot(jackknife_errors.T[2])
plt.plot(jackknife_errors.T[3])
plt.plot(jackknife_errors.T[4])
print "Jackknife:", jackknife_errors[-1]

# binning stuff
binning_results = np.array([error.binning_analysis(series, k, averages,
                                                   variances)
                                for k in range(1, 2001)])
binning_times  = binning_results[:,0,:]
binning_errors = binning_results[:,1,:]

plt.subplot(212)
plt.plot(binning_errors.T[0])
plt.plot(binning_errors.T[1])
plt.plot(binning_errors.T[2])
plt.plot(binning_errors.T[3])
plt.plot(binning_errors.T[4])
print "Binning:", binning_errors[-1]

plt.show()

# autocorrelation stuff
autocorrelation_results = error.autocorrelation_analysis(series, averages, variances)
autocorrelation_errors = autocorrelation_results[1]
print "Autocorrelation:", autocorrelation_errors
