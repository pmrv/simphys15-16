import sys
import gzip
import pickle
from matplotlib.pyplot import *

for L, f in zip((4, 16, 64), sys.argv[1:]):
    with gzip.open(f) as fdat:
        Ts, binders = pickle.load(fdat)[::5]

    plot(Ts, binders, 'o', label = "L = {}".format(L))

legend(loc = "lower left")
xlabel("$T$")
ylabel("Binder $U$")
savefig("binder.pdf")
