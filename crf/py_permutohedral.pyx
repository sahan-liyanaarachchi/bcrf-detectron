# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool

from crf.Permutohedral cimport Permutohedral

cdef class PyPermutohedral:
    cdef Permutohedral c_obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def init(self, np.ndarray[float, ndim=3, mode="c"] features not None, int num_dimensions, int num_points):
        self.c_obj.init(&features[0, 0, 0], num_dimensions, num_points)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute(self,
                np.ndarray[float, ndim=3, mode="c"] output not None,
                np.ndarray[float, ndim=3, mode="c"] inp not None,
                int value_size, bool reverse):
        self.c_obj.compute(&output[0, 0, 0], &inp[0, 0, 0], value_size, reverse)
