import cython
import numpy
cimport numpy

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng:
        pass

    ctypedef struct gsl_rng_type:
        pass

    gsl_rng_type *gsl_rng_taus
    gsl_rng *gsl_rng_alloc(const gsl_rng_type * T)
    double gsl_rng_uniform (const gsl_rng * r)
    unsigned long int gsl_rng_uniform_int (const gsl_rng * r, unsigned long int n)

cdef gsl_rng *r = NULL

# Initialize the random number generator.
cdef void init():
  global r
  r = gsl_rng_alloc(gsl_rng_taus)

# Generate a random integer between 0 and N-1.
cpdef long randint(long N):
  return gsl_rng_uniform_int(r, N)

# Generate a random float between 0.0 and 1.0.
cpdef double rand():
  return gsl_rng_uniform(r)

# Cython interface to C function
def random_list(numpy.ndarray[double, ndim=1, mode='c'] rs not None):
    # pass the array to the C function
    cdef long N = rs.shape[0]
    for i in range(N):
        rs[i] = rand()

    return rs

# call init() when the module is loaded to init the gsl RNG
init()
