from libcpp cimport bool


cdef extern from "permutohedral.cpp":
    pass


cdef extern from "permutohedral.h":
    cdef cppclass Permutohedral:
        Permutohedral() except +
        void init(const float* features, int num_dimensions, int num_points)
        void compute(float* out, const float* inp, int value_size, bool reverse)
