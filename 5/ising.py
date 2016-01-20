import numpy as np
import sys
from copy import copy
import math
from matplotlib import pyplot as plt

ver = sys.version_info
if ver.major == 3 and ver.minor > 4:
    from statistics import mean, variance
else:
    # re-use code from last exercise
    from error import mean, variance



def boltzmann_distribution(T):
    return lambda energy: np.exp(-energy/T)

def metropolis(N, P, trial_move, sigma):
    if (sigma.__class__.__name__ == 'ndarray'):
        Lx, Ly = sigma.shape
    else:
        print('Something went to shit! sigma should be numpy array!\nExiting...')
        exit(1)
    samples = np.zeros((N,Lx,Ly))
    rejects = 0
    """ used for the calculation of the acceptance rate """
    x_maps = []
    for i in range(Lx):
        for j in range(Ly):
            x_maps.append(compute_neighbour_map(sigma, i, j))
    """ x_maps like before for the computation of the energies.
    Note: We were late for the excercise and didn't care for the recomputation of the energies """
    last_energy = compute_energy(sigma, x_maps)
    for i in range(N):
        samples[i] = sigma
        sigma = copy(sigma)
        i,j = trial_move(sigma)
        """ Perform trial move """
        r = np.random.random()
        """ Draw uniformly distributed random number in the interval [0,1[ """
        energy = compute_energy(sigma, x_maps)
        last_energy = compute_energy(samples[i], x_maps)
        p_ratio = P(energy-last_energy)
        p_ratio = np.min([1., p_ratio])
        if (r >= p_ratio):
            """ Reject """
            sigma = samples[i]
            rejects += 1
#        else:
#            last_energy = energy

    acceptance_rate = float(N-1-rejects) / (N-1)
    """ N-1 because the initial state should not be included """
    return samples, acceptance_rate

def flip_random_spin(x):
    """ Randomly flip a spin """
    N,M = x.shape
    i = np.random.randint(N)
    j = np.random.randint(M)
    x[i,j] *= -1
    return i,j

def exact_summation(Lx, Ly):
    """ An unoptimized routine that sums over all possible states (2**(Lx*Ly)) of 
    the 2D Ising modell. The number of summations could be drastically reduced
    by grouping terms with the same energy and counting the number of permutations. """
    x_base = np.ones((Lx,Ly)) * (-1)
    sites = Lx * Ly
    N = 2**sites
    samples = []
    energies = []
    sum_spins = []
    """ sites is the total number of particles on the grid. N is the number of different states.
    Initialisation of x_maps follows. """
    x_maps = []
    for i in range(Lx):
        for j in range(Ly):
            x_maps.append(compute_neighbour_map(x_base, i, j))

    log2 = np.log(2)
    for i in range(N):
        x = copy(x_base.reshape(sites))
        if i == 0:
            highest_power_of_two = 0
        else:
            highest_power_of_two = int(math.log(i) / log2)
        for exponent in range(highest_power_of_two+1):
            """ Basic idea: Each bit represents a single particle/spin. If the bit is 1 it corresponds
            to spin up (+1) otherwise spin down (-1)"""
            if i & (1<<exponent):
                x[exponent] = 1
        x = x.reshape((Lx,Ly))
        energy = compute_energy(x, x_maps)
        sum_spin = compute_magnetization(x)
        samples.append(x)
        energies.append(energy)
        sum_spins.append(sum_spin)

    return samples, energies, sum_spins

def compute_energy(x, x_maps):
    """ Computes the total energy for the state x.
    x_maps is a list of x_maps for every particle. Its' length should be NxM.
    For more info on x_map see compute_energy_particle(..) and compute_neighbour_map(..)"""
    energy = 0.
    for x_map in x_maps:
        energy += compute_energy_particle(x, x_map)
    energy *= 0.5
    return energy

def compute_magnetization(x):
    """ Computes magnetization of state x. """
    N,M = x.shape
    m = 0.
    for i in range(N):
        for j in range(M):
            m += x[i,j]
    m /= N*M
    return m
        
def compute_energy_particle(x, x_map):
    """ Computes energy of particle (i,j) on the 2d grid specified by x_map
    i is the x-coordinate of the top/bottom point,
    j is the y-coordinate of the left/right point"""
    spin_map = lambda indices: x[indices[0], indices[1]]
    i = x_map[1][0]
    j = x_map[0][1]
    xij = x[i,j]
    sum_spin =np.sum( [spin_map(indices) for indices in x_map ] )
    energy = -xij * sum_spin
    return energy
    
def compute_neighbour_map(x, i, j):
    """ x_map holds a list of the indices of the 4 closest neighbours in the following order:
    left, top, right, bottom. Usage map[2] yields the tuple (k,l) that adresses 
    the right neighbour. Spin of right neighbour would be x[map[2][0],map[2][1]] """
    
    N,M = x.shape
    x_map = [[i-1,j], [i,j-1], [i+1,j], [i,j+1]]
    
    """ Make sure that indices are in range """
    if x_map[0][0] == -1:
        x_map[0][0] += N
    elif x_map[2][0] == N:
        x_map[2][0] = 0
    if x_map[1][1] == -1:
        x_map[1][1] += M
    elif x_map[3][1] == M:
        x_map[3][1] = 0

    return x_map
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        L = int(sys.argv[1])
    else:
        L = 4
    """ Metropolis Sampling!
    Initialising starting state with 1 and -1 """
    Ts = np.arange(1.,5.1,0.1)
    x = np.random.random((L,L)) *2 -1
    sigma = (x < 0) * (-1) + (x > 0) * 1
    x_maps = []
    for i in range(L):
        for j in range(L):
            x_maps.append(compute_neighbour_map(x, i, j))

    samples_metropolis = [ metropolis(10000, boltzmann_distribution(T), flip_random_spin, sigma)[0] for T in Ts ]
    sum_spins_metropolis = [ [ compute_magnetization(sample) for sample in samples ] for samples in samples_metropolis ]
#    mean_energy_per_site_metropolis = np.array([ [mean(compute_energy(sample, x_maps)) for sample in samples ] for samples in samples_metropolis ]) / (L*L)
    mean_energy_per_site_metropolis = np.array([ mean( np.array([ compute_energy(sample, x_maps) for sample in samples ]) ) for samples in samples_metropolis ]) / (L*L)
    mean_magnetization_per_site_metropolis = [ mean(np.abs(sum_spin)) for sum_spin in sum_spins_metropolis ]

    """ Exact summation! """
    samples_exact, energies_exact, sum_spins_exact = exact_summation(L,L)
    N = len(samples_exact)
    Ts = np.arange(1.,5.1,0.1)
    probabilities = [ [ boltzmann_distribution(T)(energy) for energy in energies_exact ] for T in Ts ]
    Zs = [ np.sum( p ) for p in probabilities ]
    """ Zs is the partition function for the exact summation case.
    This is used for the computation of the mean values of the energy and magnetization"""
    mean_energy_per_site_exact = np.array([ np.average(energies_exact, weights=(probabilities[i]/Zs[i])) for i in range(len(Ts)) ]) / (L*L)
    mean_magnetization_per_site_exact = np.array([ np.average(np.abs(sum_spins_exact), weights=(probabilities[i]/Zs[i])) for i in range(len(Ts)) ])

    """ Plotting """
    plt.xlabel("Temperature (T)")
    plt.ylabel("mean energy per site (e)")
    plt.plot(Ts, mean_energy_per_site_exact, label="Exact summation")
    plt.plot(Ts, mean_energy_per_site_metropolis, label="Metropolis Monte Carlo")
    plt.legend()
    plt.savefig("energy_per_site.pdf")
    plt.clf()
    plt.xlabel("Temperature (T)")
    plt.ylabel("mean magnetization per site (m)")
    plt.plot(Ts, mean_magnetization_per_site_exact, label="Exact summation")
    plt.plot(Ts, mean_magnetization_per_site_metropolis, label="Metropolis Monte Carlo")
    plt.legend()
    plt.savefig("magnetization_per_site.pdf")
