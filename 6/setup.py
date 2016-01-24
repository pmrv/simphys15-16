from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("gsl",
                  sources=["gsl.pyx"],
                  include_dirs=[numpy.get_include()],
                  libraries=["gsl", "gslcblas"]
                  ),
        Extension("mc",
                  sources=["mc.pyx"],
                  include_dirs=[numpy.get_include()],
                  libraries=["m"]
                  )
    ],
)
