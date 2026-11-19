import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

if len(sys.argv) == 1:
    sys.argv.append("build_ext")
    sys.argv.append("--inplace")

extra_compile_args = []
extra_link_args = []
if os.environ.get("OMP_ENABLED", "1") != "0":
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

ext_modules = [
    Pybind11Extension(
        "_hyperbolic_embedding",
        ["src/hyperbolic_embedding_binding.cc"],
        depends=[
            "src/hyperbolic_embedding.cc",
            "src/hyperbolic_embedding.hh",
            "src/quad_tree_hyperbolic.hh",
        ],
        include_dirs=["src", pybind11.get_include()],
        language="c++",
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="hyperbolic_embedding",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
