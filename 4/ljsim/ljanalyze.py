from __future__ import print_function
from numpy import *
from matplotlib.pyplot import *
import sys, os
import pickle
import lj

t_equi = 800

# OPEN SIMULATION FILES
if len(sys.argv) != 2:
    print("Usage: python %s ID" % sys.argv[0])
    sys.exit(2)
datafilename = sys.argv[1]

# check whether data file exists
if not os.path.exists(datafilename):
    print("ERROR: %s doesn't exist." % datafilename)
    sys.exit(1)

print("Reading data from %s." % datafilename)
datafile = open(datafilename, 'r')
_, _, x, _, ts, Es, Tms, rdf_min, rdf_max, rdf_bins, rdfs = pickle.load(datafile)
datafile.close()

ts = array(ts)
Es = array(Es)
Tms = array(Tms)
rdfs = array(rdfs)

def running_average(Os, window):
    Os_ra = empty_like(Os)
    Os_ra[:] = NaN
    for i in range(window/2, len(Os)-window/2):
        Os_ra[i] = Os[i-window/2:i+window/2].sum()/window
    return Os_ra
        
figure()
plot(ts, Es[:,0], label='$E_\mathrm{tot}$')
plot(ts, Es[:,1], label='$E_\mathrm{pot}$')
plot(ts, running_average(Es[:,1], 10))
plot(ts, running_average(Es[:,1], 100))
plot(ts, Es[:,2], label='$E_\mathrm{kin}$')
plot(ts, running_average(Es[:,2], 10))
plot(ts, running_average(Es[:,2], 10))
plot(ts, running_average(Es[:,2], 100))
title('Energy')
legend()

figure()
plot(ts, Tms)
plot(ts, running_average(Tms, 10))
title('Temperature')

if ts[-1] < t_equi:
    print("Not equilibrated yet.")
else:
    equi_step = 0
    while ts[equi_step] < t_equi:
        equi_step += 1

    print("Equilibrium values:")
    E = Es[equi_step:,:].mean(axis=0)
    print("  E =", E)
    Tm = Tms[equi_step:].mean(axis=0)
    print("  Tm =", Tm)
    rdf = rdfs[equi_step:].mean(axis=0)

    figure()
    rs = linspace(rdf_min, rdf_max, rdf_bins, endpoint=False)
    plot(rs, rdf)

show()
