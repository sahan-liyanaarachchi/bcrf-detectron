from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='bcrf',
    ext_modules=cythonize(
        Extension(
            "py_permutohedral",
            sources=["py_permutohedral.pyx"],
            include_dirs=[np.get_include()]
        ),
        # include_path=['crf']
    ),
    install_requires=["numpy"]
)
