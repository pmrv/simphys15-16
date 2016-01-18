import sys
import numpy as np
import copy

def metropolis(N, P, trial_move, phi0, dphi = 0.1):
    if (phi0.__class__.__name__ == 'ndarray'):
        d = phi0.shape[0]
    elif (phi0.__class__.__name__ == 'list'):
        d = len(phi0)
    elif (phi0.__class__.__name__ == 'int') or (phi0.__class__.__name__ == 'float'):
        d = 1
    samples = np.zeros((N,d))
    phi = np.array(phi0)
    rejects = 0 """ used for the calculation of the acceptance rate """
    for i in range(N):
        samples[i] = phi
        phi = trial_move(phi, dphi) """ Perform trial move """
        r = np.random.random() """ Draw uniformly distributed random number in the interval [0,1[ """
        p_ratio = P(phi)/P(samples[i])
        p_ratio = np.min([1., p_ratio])
        if (r >= p_ratio): """ Reject """
            phi = samples[i]
            rejects += 1

    acceptance_rate = float(N-1-rejects) / (N-1) """ N-1 because the initial state phi0 should not be included """
    return samples, acceptance_rate

def trial_move(phi, dphi = 0.1): 
    factor = dphi * (np.random.random()*2 -1) """ Uniform in [-dphi, dphi[ """
    new_phi = phi + factor * np.ones_like(phi)
    return new_phi
