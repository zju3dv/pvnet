import os

ceres_include='./include'          # contains Ceres/ceres.h
ceres_library='./lib/libceres.so'
eigen_include='./include/eigen3'                     # contains Eigen/Core
glog_library='./lib/libglog.so'
cuda_include='/usr/local/cuda-8.0/include'
cudart = '/usr/local/cuda-8.0/lib64/libcudart.so'

os.system('gcc -shared src/mesh_rasterization.cpp -c -o src/mesh_rasterization.cpp.o -fopenmp -fPIC -O2 -std=c++11')
os.system('gcc -shared src/farthest_point_sampling.cpp -c -o src/farthest_point_sampling.cpp.o -fopenmp -fPIC -O2 -std=c++11')
os.system('gcc -shared src/uncertainty_pnp.cpp -c -o src/uncertainty_pnp.cpp.o -fopenmp -fPIC -O2 -std=c++11 -I {} -I {}'.
          format(ceres_include,eigen_include))
os.system('nvcc src/nearest_neighborhood.cu -c -o src/nearest_neighborhood.cu.o -x cu -Xcompiler -fPIC -O2 -arch=sm_52 -I {} -D_FORCE_INLINES'.
          format(cuda_include))

from cffi import FFI
ffibuilder = FFI()


# cdef() expects a string listing the C types, functions and
# globals needed from Python. The string follows the C syntax.
with open(os.path.join(os.path.dirname(__file__), "src/utils_python_binding.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source("_extend_utils",
                      """
                             #include "src/utils_python_binding.h"   // the C header of the library
                      """,
                      extra_objects=['src/mesh_rasterization.cpp.o','src/farthest_point_sampling.cpp.o',
                                     'src/uncertainty_pnp.cpp.o','src/nearest_neighborhood.cu.o',
                                     # 'src/post_process.cpp.o',
                                     ceres_library, glog_library,
                                     cudart],
                      libraries=['stdc++']
                      )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    os.system("rm src/*.o")
    # os.system("rm *.o")
