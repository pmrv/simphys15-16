from numpy import *
from matplotlib.pyplot import *
from argparse import ArgumentParser as AP
import multiprocessing.pool
from datetime import datetime
import pickle
import gzip

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

        acfs.append(acf)
        tau_ints.append(tau_int)
        k += 1

    N_eff = N/(2*tau_int)
    err_tau = tau_int*sqrt(12./N_eff)
    err_x = sqrt(xvar/N*2.0*tau_int)
    return xmean, xvar, err_x, tau_int, err_tau, N_eff, \
        array(acfs), array(tau_ints)

##################################################
## MONTE CARLO
##################################################
def monte_carlo_ising( (L, T, num_sweeps) ):
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

    Es = Es/float(V)
    ms = abs(ms)/float(V)
    Emean, _, Eerr, tauE, _, _, _, _ = compute_act_error(Es)
    mmean, _, merr, tauM, _, _, _, _ = compute_act_error(ms)
    print "\rT = {} tau_E = {} tau_M = {} E = {}+/-{} m = {}+/-{}"\
        .format(T, tauE, tauM, Emean, Eerr, mmean, merr)

    return Emean, Eerr, mmean, merr, sigma, Es, ms

if __name__ == "__main__":
    parser = AP()
    parser.add_argument("-n", "--sweeps", type = int, default = 1000,
                        help = "number of iterations to be done in MC calculation")
    parser.add_argument("-L", "--size", type = int, nargs = "+", default = [16],
                        help = "system size")
    parser.add_argument("-p", "--plot", action = "store_true",
                        help = "whether to plot data at the end or not")
    parser.add_argument("-s", "--store", action = "store_true",
                        help = "whether to write out data at the end or not")
    parser.add_argument("-T", "--temperature", type = float, nargs = 3,
                        default = (1.0, 5.0, 0.1), help = "temperature range")
    parser.add_argument("-c", "--cores", type = int, default = 1,
                        help = "how many cores should be used")
    parser.add_argument("-b", "--binder", action = "store_true",
                        help = "whether to calculate Binder U at each temperature or not")
    args = parser.parse_args()

    # Main program
    T0, T1, Tstep = args.temperature
    Ts = linspace(T0, T1, round((T1 - T0) / Tstep) + 1)
    pool = multiprocessing.pool.Pool(args.cores)

    # Main program
    for L in args.size:
        print "MC (L={})".format(L)

        Emeans = []
        Eerrs = []
        mmeans = []
        merrs = []
        sigmas = []
        if args.binder:
            binders = []

        # why are we using py2 again? I want pool.starmap!
        for Emean, Eerr, mmean, merr, sigma, _, ms in pool.map(
                monte_carlo_ising, zip([L]*len(Ts), Ts, [args.sweeps]*len(Ts))):
            Emeans.append(Emean)
            Eerrs.append(Eerr)
            mmeans.append(mmean)
            merrs.append(merr)
            sigmas.append(sigma)
            if args.binder:
                binders.append(1 - 1./3 * (ms**4).mean() / (ms**2).mean()**2)

        Emeans  = array(Emeans)
        Eerrs   = array(Eerrs)
        mmeans  = array(mmeans)
        merrs   = array(merrs)
        binders = array(binders)
        if args.plot:
            figure(0)
            subplot(211, title='Energy vs. Temperature')
            errorbar(Ts, Emeans, yerr=Eerrs, fmt='o-', label='MC L={}'.format(L))
            legend()

            subplot(212, title='Magnetization vs. Temperature')
            errorbar(Ts, mmeans, yerr=merrs, fmt='o-', label='MC L={}'.format(L))
            legend()
        if args.store:
            with gzip.open("ising{}-{}-{}-{}.dat".format(
                    "-binder" * args.binder, L, args.sweeps, datetime.now()), 'w') as fdat:
                if args.binder:
                    pickle.dump([Ts, Emeans, Eerrs, mmeans, merrs, binders], fdat)
                else:
                    pickle.dump([Ts, Emeans, Eerrs, mmeans, merrs], fdat)

    if args.plot:
        figure('Final states')
        numplots = len(sigmas)
        cols = int(ceil(sqrt(numplots)))
        rows = int(ceil(numplots/float(cols)))
        for i in range(numplots):
            subplot(rows, cols, i+1, title='T={}'.format(Ts[i]))
            axis('off')
            imshow(sigmas[i], interpolation='nearest')

        show()
