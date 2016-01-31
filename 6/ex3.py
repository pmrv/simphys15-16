#!/usr/bin/env python3 

import sys
import gzip
import pickle
from matplotlib.pyplot import *

from ising import exact_sum

figure(0)
xlabel("$E$")
ylabel("$T$")

figure(1)
xlabel("$m$")
ylabel("$T$")

for L, f in zip((16, 64), sys.argv[1:]):
    with gzip.open(f) as fdat:
        Ts, Emeans, Eerrs, mmeans, merrs = pickle.load(fdat)[:6]

    figure(0)
    errorbar(Ts, Emeans, yerr=Eerrs, fmt='x', label='MC L={}'.format(L))
    legend()

    figure(1)
    errorbar(Ts, mmeans, yerr=merrs, fmt='x', label='MC L={}'.format(L))
    legend()

Emeans_exact, mmeans_exact = exact_sum(4, Ts)
figure(0)
plot(Ts, Emeans, '+', label='MC L=4 (exact)')
legend()
savefig("energy.pdf")

figure(1)
plot(Ts, mmeans, '+', label='MC L=4 (exact)')
legend()
savefig("magnetization.pdf")
