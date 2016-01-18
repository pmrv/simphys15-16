import sys
import numpy as np
import matplotlib.pyplot as plt

ver = sys.version_info
if ver.major == 3 and ver.minor > 4:
    from statistics import mean, variance
else:
    # re-use code from last exercise
    from error import mean, variance

def runge(x):
    return 1. / (1 + x**2)

def exact_runge_integral():
    return np.arctan(5) - np.arctan(-5)

def simple_sampling(f, a, b, N):
    samples = (b - a) * f(np.random.rand(N) * (b - a) + a)
    m = mean(samples)
    return m, np.sqrt(variance(samples, xbar = m))

if __name__ == "__main__":

    exact = exact_runge_integral()
    I          = np.empty(19)
    stat_error = np.empty(19)
    real_error = np.empty(19)
    Ns = 1 << np.arange(2, 21)
    for i, N in enumerate(Ns):
        I[i], stat_error[i] = simple_sampling(runge, -5, 5, N)
        real_error[i] = abs(I[i] - exact)

    plt.plot(np.log2(Ns), stat_error, label = "statistical error")
    plt.plot(np.log2(Ns), real_error, label = "actual error")
    plt.legend(loc = "center right")
    plt.xlabel("$\log_2 N$")
    plt.ylabel("error")
    plt.savefig("error.pdf")
    plt.show()
