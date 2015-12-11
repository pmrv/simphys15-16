from __future__ import print_function
from numpy import *
from matplotlib.pyplot import *
import sys, os
import pickle 
import lj

# SYSTEM CONSTANTS
# density
density = 0.316
# timestep
dt = 0.01
# length of run
trun = 10.0
# desired temperature
T = 0.3

# SIMULATION CONSTANTS
# warmup?
warmup_on = True
force_cap = 20.0
force_cap_factor = 1.1

# velocity rescaling thermostat on?
rescaling_on = True
# skin size
skin = 0.4
# number of steps to do before the next measurement
measurement_stride = 100
# cutoff length
rcut = 2.5
# potential shift
shift = -0.016316891136

# Measure RDF?
rdf_on = True
# minimal and maximal distance for the rdf histogram
rdf_min = 0.8
rdf_max = 5.0
# number of bins for the rdf histogram
rdf_bins = 100

# COMPUTED CONSTANTS
# total number of particles
N = 1000
# volume of the system
volume = N/density
# side length of the system
L = volume**(1./3.)

# OPEN SIMULATION FILES
if len(sys.argv) != 2:
    print("Usage: python {} ID".format(sys.argv[0]))
    sys.exit(2)
simulation_id = sys.argv[1]

vtffilename = '{}.vtf'.format(simulation_id)
datafilename = '{}.dat'.format(simulation_id)

def compute_temperature(v):
    _, N = v.shape
    Tm = (v*v).sum()/(3*N)
    return Tm

def rescale_velocities(v, Tm):
    global T
    v *= sqrt(T/Tm)
    return v

def compute_forces(x):
    global force_cap, warmup_on, warmup_finished
    f = lj.compute_forces(x)
    if warmup_on:
        # cap forces
        warmup_finished = all(f < force_cap) and all(f > -force_cap)
        for i in range(3):
            for j in range(N):
                if f[i,j] > force_cap: 
                    f[i,j] = force_cap
                elif f[i,j] < -force_cap: 
                    f[i,j] = -force_cap
    return f

def step_vv(x, v, f, dt, xup):
    global rcut, skin

    # update positions
    x += v*dt + 0.5*f * dt*dt

    # compute maximal position update
    # vectorial
    dx = x - xup
    # square
    dx *= dx
    # sum up 
    dx = dx.sum(axis=0)
    # test whether the neighbor list needs to be rebuilt
    if max(dx) > (0.5*skin)**2:
        lj.rebuild_neighbor_lists(x, rcut+skin)
        xup = x.copy()
    
    # half update of the velocity
    v += 0.5*f * dt
        
    # compute new forces
    f = compute_forces(x)

    # second half update of the velocity
    v += 0.5*f * dt

    return x, v, f, xup

# SET UP SYSTEM OR LOAD IT
# check whether data file already exists
if os.path.exists(datafilename):
    print("Reading data from {}.".format(datafilename))
    datafile = open(datafilename, 'r')
    step, t, x, v, ts, Es, Tms, rdf_min, rdf_max, rdf_bins, rdfs = pickle.load(datafile)
    datafile.close()
    print("Restarting simulation at t={}...".format(t))
else:
    print("Starting simulation...")
    t = 0.0
    step = 0

    # random particle positions
    x = random.random((3,N))*L
    # random particle velocities
    v = 0.1*(2.0*random.random((3,N))-1.0)

    # variables to cumulate data
    ts = []
    Es = []
    Tms = []
    rdfs = []

print("density={}, L={}, N={}".format(density, L, N))

# check whether vtf file already exists
if os.path.exists(vtffilename):
    print("Opening {} to append new timesteps...".format(vtffilename))
    vtffile = open(vtffilename, 'a')
else:
    print("Creating {}...".format(vtffilename))
    # create a new file and write the structure
    vtffile = open(vtffilename, 'a')

    # write the structure of the system into the file: 
    # N particles ("atoms") with a radius of 0.5
    vtffile.write('atom 0:{} radius 0.5\n'.format(N-1))
    vtffile.write('pbc {} {} {}\n'.format(L, L, L))
    
    # write out that a new timestep starts
    vtffile.write('timestep\n')
    # write out the coordinates of the particles
    for i in range(N):
        vtffile.write("{} {} {}\n".format(x[0,i], x[1,i], x[2,i]))

# main loop
lj.set_globals(L, N, rcut, shift)
lj.rebuild_neighbor_lists(x, rcut+skin)
xup = x.copy()
f = compute_forces(x)
tmax = t+trun
if warmup_on:
    print("Warmup simulation...")
    warmup_finished = False
else:
    print("Simulating until tmax={}...".format(tmax))

while (not warmup_on and t < tmax) or (warmup_on and not warmup_finished):
    x, v, f, xup = step_vv(x, v, f, dt, xup)
    t += dt
    step += 1
    if warmup_on: 
        Tm = compute_temperature(v)
        v = rescale_velocities(v, Tm)

    if step % measurement_stride == 0:
        E = lj.compute_energy(x, v)
        Tm = compute_temperature(v)
        print("t={}, E={}, T_m={}".format(t, E, Tm))

        if warmup_on:
            force_cap *= force_cap_factor
            print("force_cap={}".format(force_cap))
        else:
            ts.append(t)
            Es.append(E)
            Tms.append(Tm)

            # RDF
            ds = lj.compute_distances(x)
            rdf_histo, xs = histogram(ds, rdf_bins, (rdf_min, rdf_max))
            rdf = rdf_histo/(N*density*2.0*pi*xs[1:]*xs[1:]*((rdf_max-rdf_min)/rdf_bins))
            rdfs.append(rdf)
        
        # write out that a new timestep starts
        vtffile.write('timestep\n')
        # write out the coordinates of the particles
        for i in range(N):
            vtffile.write("{} {} {}\n".format(x[0,i], x[1,i], x[2,i]))
            
        if rescaling_on:
            v = rescale_velocities(v, Tm)

# at the end of the simulation, write out the final state
print("Writing simulation data to {}.".format(datafilename))
datafile = open(datafilename, 'w')
pickle.dump([step, t, x, v, ts, Es, Tms, rdf_min, rdf_max, rdf_bins, rdfs], datafile)
datafile.close()

# close vtf file
print("Closing {}.".format(vtffilename))
vtffile.close()
if warmup_on:
    print("Finished warmup.")
else:
    print("Finished simulation.")
