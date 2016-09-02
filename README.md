PRIMAL-DUAL TOOLBOX
===================================================
C++/Cuda implementation of various Total Variation (TV) and second-order Total Generalized Variation (TGV) [1,2] problems using the primal-dual algorithm [3], including Python and Matlab wrappers. This toolbox is used for TGV-based MRI reconstruction presented in [4]. If you find this software useful for your academic work, please cite the related publications.

#### Currently supported operators
- Denoising (real-valued and complex-valued)
- Cartesian MRI reconstruction (2D)
- Radial MRI reconstruction (2D, requires [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)) [5,6]

The single operators can be used in the different optimizers (TV, TGV) or standalone.

Dependencies
------------
- [ImageUtilities](https://github.com/VLOGroup/imageutilities)
- **Optional**: [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT)

These modules are added as submodules to this repository. You can either clone
this repository with the flag `--recursive` or do a
~~~
$ git submodule update --init --recursive --remote
~~~
*Note: The framework was tested using Ubuntu 16.04, gcc 5.4 and cuda 8.0.
We support both python2 and python3. We highly recommend to use an [Anaconda](https://www.anaconda.com/download/) environment.*

Installation
-------------

- Set up environment variable `COMPUTE_CAPABILITY` with the CC of your CUDA-enabled GPU
- Set up environment variable `CUDA_SDK_ROOT_DIR` to point to the NVidia CUDA examples (required to find <matlab_helper.h>)
- Set up environment variable `IMAGEUTILITIES_ROOT` to point to the path of the ImageUtilities root directory
- Set up environment variable `MATLAB_ROOT` to point to the matlab root directory
- **Optional**: Set up environment variable `GPUNUFFT_ROOT_DIR` to point to the path of the gpuNUFFT root directory
- Make sure that you have boost-python installed.
  We highly recommend to use an Anaconda environment. To install boost-python here,
  simply do
  ~~~
  $ conda install boost
  ~~~
  *Note: Please make sure to use the same environment for building this software and running your code. If you update boost or numpy, you might have to re-build this
  software, because the versions have to match. If you wish to build this software
  for a specific Anaconda environment, activate this environment before the building
  process using `source activate <your_python_environment>`.*

  If you want to use your system python,
  you can install boost-python using:
  ~~~
  $ sudo apt-get install libboost-python
  ~~~

To build the primal-dual toolbox including the dependencies, simply perform the following steps:

- **ImageUtilities** (requires `libboost-python, libopenexr-dev`)
  ~~~
  $ cd imageutilities/build
  $ cmake .. -DWITH_PYTHON=ON -DWITH_MATLAB=ON
  $ make
  $ make install
  $ cd ../../
  ~~~
  If you wish to exclude the Matlab wrapper, simply set `-DWITH_MATLAB=OFF` in above building steps.

- **gpuNUFFT (optional)**
  ~~~
  $ cd gpuNUFFT/CUDA/build
  $ cmake .. -DGEN_MEX_FILES=ON -DMATLAB_ROOT_DIR=$MATLAB_ROOT
  $ make
  $ cd ../../../
  ~~~
  If you wish to build gpuNUFFT without the mex files, replace above CMake command by:
  ~~~
  $ cmake .. -DGEN_MEX_FILES=OFF
  ~~~

- **Primal-Dual-Toolbox**
  ~~~
  $ mkdir build
  $ cd build
  $ cmake .. -DWITH_GPUNUFFT=ON
  $ make
  $ cd ../
  ~~~
   To build without gpuNUFFT, use `cmake .. -DWITH_GPUNUFFT=OFF` instead of `cmake .. -DWITH_GPUNUFFT=ON`.

After building the C-code, you can build and install the Python package from the root directory of this repository as follows:
~~~
$ python setup.py bdist_wheel
$ pip install dist/<your-PrimalDualToolbox-wheel-package>.whl
~~~

*Please make sure to use the same python environment that you built this software with.*

Documentation
-------------
To build the documentation (requires doxygen), additionally execute
following command in the build directory:
~~~
$ make apidoc
~~~

Test Python Module
-----
To test the python module simply check if following works without error:
~~~
$ ipython
$ import primaldualtoolbox
~~~

Run tests
-----
Go into the `bin` directory. To run the python samples, simply type
~~~
$ python denoising_test.py
$ python denoising_complex_test.py
$ python mri_cartesian_test.py
$ python mri_radial_test.py
~~~
To run the Matlab samples, open Matlab and run `mri_cartesian_test.m` or `mri_radial_test.m`. The data used for the radial tests is the same as provided
in the [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT). The examples show simple use of the implemented
operators and how to run TGV reconstructions.

Common issues:
----------
~~~
Invalid MEX-file '~/pd_toolbox/lib/gpuMriCartesianRemoveROOSAdj.mexa64': ~/MATLAB/R2014a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by
~/pd_toolbox/lib/gpuMriCartesianFwd.mexa64)
~~~
Start your Matlab with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab`.

References
----------
- [1] K Bredies, K Kunisch, T Pock. *Total generalized variation*. SIAM Journal on Imaging Sciences 3 (3), pp. 492-526, 2010.
- [2] A Chambolle, T Pock. *An  introduction  to  continuous  optimization  for  imaging*.  Acta Numerica, 25, pp. 161-319, 2016.
- [3] A Chambolle, T Pock. *A first-order primal-dual algorithm for convex problems with applications to imaging*. Journal of Mathematical Imaging and Vision 40 (1), pp. 120-145, 2011.
- [4] K Hammernik, T Klatzer, E Kobler, MP Recht, DK Sodickson, T Pock, F Knoll. *Learning a Variational Network for Reconstruction of Accelerated MRI Data*. Magnetic Resonance in Medicine, 2017 (early view).
- [5] F Knoll, K Bredies, T Pock, R Stollberger. *Second order total generalized variation (TGV) for MRI*. Magnetic Resonance in Medicine 65 (2), pp. 480-491, 2011.
- [6] F Knoll, A Schwarzl, C Diwoky, DK Sodickson. *gpuNUFFT-an open source GPU library for 3D regridding with direct Matlab interface*. Proceedings of the 22nd Annual Meeting of ISMRM, Milan, Italy, p. 4297, 2014.
