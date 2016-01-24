from numpy import *
from matplotlib.pyplot import *

import gsl
import mc
from mc import compute_energy

def compute_magnetization(sigma):
    return sigma.sum()

def compute_act_error(x):
    x = asarray(x)
    N = len(x)
    xmean = x.mean()
    xvar = x.var()
    acfs = []
    tau_ints = []
    k = 0
    tau_int = 0.5
    while k < 6*tau_int:
        acf = ((x[:N-k]*x[k:]).mean() - xmean*xmean) / xvar
        tau_int += acf
        N_eff = N/(2*tau_int)
        err_tau = tau_int*sqrt(12./N_eff)

        acfs.append(acf)
        tau_ints.append(tau_int)
        k += 1

    err_x = sqrt(xvar/N*2.0*tau_int)
    return xmean, xvar, err_x, tau_int, err_tau, N_eff, \
        array(acfs), array(tau_ints)

##################################################
## MONTE CARLO
##################################################
def monte_carlo_ising(L, T, num_sweeps):
    V = L*L
    beta = 1.0/T

    # generate random configuration
    sigma = random.randint(0, 2, (L, L))
    sigma *= 2
    sigma -= 1

    E  = compute_energy(sigma)
    mu = sigma.sum()

    Es = empty(num_sweeps, int)
    ms = empty(num_sweeps, int)

    mc.core(E, mu, Es, ms, num_sweeps, sigma, beta)

    Es = Es.astype(float) / V
    ms = abs(ms)/float(V)
    Emean, _, Eerr, tauE, _, _, _, _ = compute_act_error(Es)
    mmean, _, merr, tauM, _, _, _, _ = compute_act_error(ms)
    print "\rT = {} tau_E = {} tau_M = {} E = {}+/-{} m = {}+/-{}"\
        .format(T, tauE, tauM, Emean, Eerr, mmean, merr)

    return Emean, Eerr, mmean, merr, sigma

# Main program
num_sweeps = 10000
Ts = arange(1.0, 5.0, 0.1)

# Main program
for L in [16]:
    print "MC (L={})".format(L)

    Emeans = []
    Eerrs = []
    mmeans = []
    merrs = []
    sigmas = []

    for T in Ts:
        Emean, Eerr, mmean, merr, sigma = monte_carlo_ising(L, T, num_sweeps)
        Emeans.append(Emean)
        Eerrs.append(Eerr)
        mmeans.append(mmean)
        merrs.append(merr)
        sigmas.append(sigma)

"""
    figure(0)
    subplot(211, title='Energy vs. Temperature')
    errorbar(Ts, Emeans, yerr=Eerrs, fmt='o-', label='MC L={}'.format(L))
    legend()

    subplot(212, title='Magnetization vs. Temperature')
    errorbar(Ts, mmeans, yerr=merrs, fmt='o-', label='MC L={}'.format(L))
    legend()

figure('Final states')
numplots = len(sigmas)
cols = int(ceil(sqrt(numplots)))
rows = int(ceil(numplots/float(cols)))
for i in range(numplots):
    subplot(rows, cols, i+1, title='T={}'.format(Ts[i]))
    axis('off')
    imshow(sigmas[i], interpolation='nearest')

show()
"""