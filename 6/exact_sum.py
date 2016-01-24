from numpy import *
from matplotlib.pyplot import *

from ising import compute_energy, compute_magnetization

##################################################
## EXACT SUMMATION
##################################################
def exact_sum(L, Ts):
    # we compute the mean energy and magnetization for all
    # temperatures at once!
    ws = zeros_like(Ts)
    Es = zeros_like(Ts)
    ms = zeros_like(Ts)
    # beta is a NumPy array with len(Ts) elements
    beta = 1./Ts

    V = float(L*L)

    sigma = ones((L, L), dtype=int)
    # the bit pattern of the integer "state" is used to generate all
    # possible states of sigma
    for state in range(2**(L*L)):
        # read out the bitpattern
        for i in range(L):
            for j in range(L):
                k = i*L + j
                if state & 2**k > 0:
                    sigma[i,j] = 1
                else:
                    sigma[i,j] = -1

        if state%10000==0: print state

        # compute energy and magnetization of this state
        E = compute_energy(sigma)
        mu = compute_magnetization(sigma)

        # this is a vector operation, as beta is a vector
        w = exp(-beta*E)
        ws += w
        Es += E/V*w
        ms += abs(mu)/V*w

    Emeans = Es/ws
    mmeans = ms/ws
    return Emeans, mmeans

print "exact summation"
Ts = arange(1.0, 5.1, 0.1)
Emeans, mmeans = exact_sum(4, Ts)
for i in range(len(Ts)):
    print "\tT = {} E = {} m = {}".format(Ts[i], Emeans[i], mmeans[i])

figure(0)
subplot(211, title='Energy vs. Temperature')
plot(Ts, Emeans, 'o-', label='exact')
legend()

subplot(212, title='Magnetization vs. Temperature')
plot(Ts, mmeans, 'o-', label='exact')
legend()
