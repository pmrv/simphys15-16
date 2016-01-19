import sys
import numpy as np
from ex3 import runge
from matplotlib import pyplot as plt

def metropolis(N, P, trial_move, phi0, dphi = 0.1):
    if (phi0.__class__.__name__ == 'ndarray'):
        d = phi0.shape[0]
    elif (phi0.__class__.__name__ == 'list'):
        d = len(phi0)
    elif (phi0.__class__.__name__ == 'int') or (phi0.__class__.__name__ == 'float'):
        d = 1
    samples = np.zeros((N,d))
    phi = np.array(phi0)
    rejects = 0
    """ used for the calculation of the acceptance rate """
    for i in range(N):
        samples[i] = phi
        phi = trial_move(phi, dphi)
        """ Perform trial move """
        r = np.random.random()
        """ Draw uniformly distributed random number in the interval [0,1[ """
        p_ratio = P(phi)/P(samples[i])
        p_ratio = np.min([1., p_ratio])
        if (r >= p_ratio):
            """ Reject """
            phi = samples[i]
            rejects += 1

    acceptance_rate = float(N-1-rejects) / (N-1)
    """ N-1 because the initial state phi0 should not be included """
    return samples, acceptance_rate

def trial_move(phi, dphi = 0.1): 
    factor = dphi * (np.random.random()*2 -1)
    """ Uniform in [-dphi, dphi[ """
    new_phi = phi + factor * np.ones_like(phi)
    return new_phi

if __name__ == '__main__':
    phi0 = 0.

    sam1, acc1 = metropolis(100000, runge, trial_move, phi0, 0.1)
    sam2, acc2 = metropolis(100000, runge, trial_move, phi0, 1.)
    sam3, acc3 = metropolis(100000, runge, trial_move, phi0, 10.)
    sam4, acc4 = metropolis(100000, runge, trial_move, phi0, 100.)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    axes = [ax1, ax2, ax3, ax4]
    dphi = [0.1, 1., 10., 100.]
    acc = [acc1, acc2, acc3, acc4]
    """ for "convenient" setting of labels """
    for i in range(len(axes)):
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('#')
        axes[i].set_title('$\Delta x = {}$'.format(dphi[i]))
        print('{} dx={} acceptance rate={}\n'.format(i, dphi[i], acc[i]))
        
    ax1.hist(sam1, bins=100, range=(-5.,5.))
    ax2.hist(sam2, bins=100, range=(-5.,5.))
    ax3.hist(sam3, bins=100, range=(-5.,5.))
    ax4.hist(sam4, bins=100, range=(-5.,5.))
    
    f.savefig('hist_runge.pdf')

