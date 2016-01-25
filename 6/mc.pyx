import cython
import numpy
cimport numpy

cimport gsl
from gsl import randint, rand

cdef extern from "math.h":
    double exp(double x)

def compute_energy(numpy.ndarray[long, ndim = 2] sigma not None):

    cdef int L = sigma.shape[0]
    cdef long int E = 0
    cdef int i, j
    for i in range(L):
        for j in range(L):
            E -= sigma[i, j] * sigma[i, (j + 1) % L]
            E -= sigma[i, j] * sigma[(i + 1) % L, j]

    return E

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def core(long E, long mu,
         numpy.ndarray[long, ndim = 1] Es not None,
         numpy.ndarray[long, ndim = 1] ms not None,
         int num_sweeps,
         numpy.ndarray[long, ndim = 2] sigma not None,
         double beta):

    cdef int L = sigma.shape[0], V = L * L
    cdef int dE = 0, dmu = 0, deltaE
    cdef unsigned int i, j

    for sweep in range(num_sweeps):
        for step in range(V):
            # flip single spin
            i, j = gsl.randint(L), gsl.randint(L)
            sigma[i,j] *= -1

            deltaE = -2*sigma[i,j]*(sigma[(i-1)%L, j] +
                                    sigma[(i+1)%L, j] +
                                    sigma[i, (j-1)%L] +
                                    sigma[i, (j+1)%L])

            if gsl.rand() < exp(-beta*deltaE):
                # accept move
                E  += deltaE
                mu += 2*sigma[i,j]
            else:
                # reject move, i.e. restore spin
                sigma[i,j] *= -1

        Es[sweep] = E
        ms[sweep] = mu
