# This file is part of primal-dual-toolbox.
#
# Copyright (C) 2018 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
# Institute of Computer Graphics and Vision, Graz University of Technology
# https://www.tugraz.at/institute/icg/research/team-pock/
#
# primal-dual-toolbox is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# primal-dual-toolbox is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import os.path
import subprocess
import shutil

from setuptools import setup, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

shutil.copy2('lib/libprimaldualtoolbox.so', 'primaldualtoolbox/')
shutil.copy2('lib/pymrireconstruction.so', 'primaldualtoolbox/')
shutil.copy2('lib/pydenoising.so', 'primaldualtoolbox/')

setup(name='PrimalDualToolbox',
  version='1.0.0',
  description='Primal-Dual Toolbox.',
  long_description=readme,
  author='Kerstin Hammernik',
  author_email='hammernik@icg.tugraz.com',
  url='https://github.com/VLOGroup',
  packages=['primaldualtoolbox'],
  license="GNU GPL v3",
  package_data={'primaldualtoolbox':'lib/*.so'},
  distclass=BinaryDistribution,
  classifiers=[
      'Development Status :: 1 - Planning',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License (GPL)',
      'Operating System :: Linux',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: C++',
      'Programming Language :: CUDA',
      ],
  #python_requires='~=2.7'
 )
