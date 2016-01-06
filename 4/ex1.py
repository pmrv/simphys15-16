#!/usr/bin/env python2
import sys
from matplotlib import pyplot as plt
import numpy as np
import pickle

import error

def calc():
    with open("series.dat") as fdata:
        series = pickle.load(fdata).T

    averages  = error.mean(series)
    variances = error.variance(series)

    # jackknife stuff
    jackknife_errors = np.array([error.jackknife_analysis(series, k, averages)
                                    for k in range(1, 2001)])

    # binning sutff
    binning_results = np.array([error.binning_analysis(series, k, averages,
                                                    variances)
                                    for k in range(1, 2001)])
    binning_errors = binning_results[:,1,:]

    # autocorrelation stuff
    autocorrelation_results = error.autocorrelation_analysis(series, averages, variances)
    autocorrelation_errors = autocorrelation_results[1]

    np.save("jackknife.npy", jackknife_errors)
    np.save("binning.npy", binning_results)
    np.save("autocorrelation.npy", autocorrelation_results)

    print "Binning:", binning_errors[-1]
    print "Jackknife:", jackknife_errors[-1]
    print "Autocorrelation:", autocorrelation_errors

def plot():

    binning_results = np.load("binning.npy")
    jackknife_errors = np.load("jackknife.npy")
    autocorrelation_results = np.load("autocorrelation.npy")
    autocorrelation_errors = autocorrelation_results[1]

    plt.subplot(211)
    plt.xlabel("$k$")
    plt.ylabel("$\epsilon^2$")
    plt.title("jackknife")
    plt.plot(jackknife_errors.T[0], label = "1")
    plt.plot(jackknife_errors.T[1], label = "2")
    plt.plot(jackknife_errors.T[2], label = "3")
    plt.plot(jackknife_errors.T[4], label = "5")

    plt.legend(loc = "upper left")

    binning_times  = binning_results[:,0,:]
    binning_errors = binning_results[:,1,:]

    plt.subplot(212)
    plt.xlabel("$k$")
    plt.ylabel("$\epsilon^2$")
    plt.title("binning")
    plt.plot(binning_errors.T[0], label = "1")
    plt.plot(binning_errors.T[1], label = "2")
    plt.plot(binning_errors.T[2], label = "3")
    plt.plot(binning_errors.T[4], label = "5")

    plt.legend(loc = "upper left")
    plt.savefig("errors_big.pdf")
    plt.clf()

    # plot last data set seperately because it's pretty small compared to the
    # others
    plt.subplot(211)
    plt.xlabel("$k$")
    plt.ylabel("$\epsilon^2$")
    plt.title("jackknife")
    plt.plot(jackknife_errors.T[3], label = "4")

    plt.legend(loc = "upper left")

    plt.subplot(212)
    plt.xlabel("$k$")
    plt.ylabel("$\epsilon^2$")
    plt.title("binning")
    plt.plot(binning_errors.T[3], label = "4")

    plt.legend(loc = "upper left")
    plt.savefig("errors_small.pdf")
    plt.clf()

    plt.subplot(211)
    plt.xlabel("$k$")
    plt.ylabel(r"$\tau$")
    plt.plot(binning_times.T[0], label = "1")
    plt.plot(binning_times.T[1], label = "2")
    plt.plot(binning_times.T[2], label = "3")
    plt.plot(binning_times.T[4], label = "5")

    plt.legend(loc = "upper left")

    plt.subplot(212)
    plt.xlabel("$k$")
    plt.ylabel(r"$\tau$")
    plt.plot(binning_times.T[3], label = "4")

    plt.legend(loc = "upper left")
    plt.savefig("blocking-tau.pdf")
    plt.clf()

if __name__ == "__main__":
    if "calc" in sys.argv: calc()
    else:
        try: plot()
        except IOError:
            print "No results of previous calculations found. " \
                  "Run `{} calc` first".format(sys.argv[0])
