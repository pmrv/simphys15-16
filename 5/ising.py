import numpy as np
import sys
from metropolis import metropolis

ver = sys.version_info
if ver.major == 3 and ver.minor > 4:
    from statistics import mean, variance
else:
    # re-use code from last exercise
    from error import mean, variance


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
    """ x_map holds a list of the 4 closest neighbours in the following order:
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
    """ initialising starting state with 1 and -1 """
    x = np.random.random((L,L)) *2 -1
    sigma = (x < 0) * (-1) + (x > 0) * 1
