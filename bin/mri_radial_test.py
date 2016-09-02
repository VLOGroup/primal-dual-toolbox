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


import primaldualtoolbox
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title):
    plt.imshow(np.abs(img), cmap='gray')
    plt.title(title)
    plt.axis('off')

# load data
k = np.load('../data/brain_trajectory.npy')
sensitivities = np.load('../data/brain_sensitivities.npy')
rawdata = np.load('../data/brain_rawdata.npy')

# setup trajectory k and dcf weighting w
[nCh, nFE, nSpokes] = rawdata.shape
rawdata = np.reshape(rawdata, [nCh, nFE*nSpokes])
k_col = k.flatten()
k_col = np.array([np.imag(k_col), np.real(k_col)])
w = np.sqrt(np.abs(k.flatten()))[np.newaxis,:]

# init operator
config = {'osf' : 2,
          'sector_width' : 8,
          'kernel_width' : 3,
          'img_dim' : nFE/2}
op = primaldualtoolbox.mri.MriRadialOperator(config)

# set operator constants. This has to be done in the right order.
op.setTrajectory(k_col)
op.setDcf(w)
op.setCoilSens(sensitivities)

# Compute input0
input0 = op.adjoint(rawdata)
plt.figure(1)
imshow(input0, 'input0')

# init optimizer
optimizer = primaldualtoolbox.mri.TgvMriOptimizer_2c2c()

# set TGV parameters (parameters from Knoll et al.)
optimizer.getParameters().reduction = 2**(-8)
optimizer.getParameters().max_iter = 1000
optimizer.getParameters().alpha1 = 1e-5/optimizer.getParameters().reduction
optimizer.getParameters().alpha0 = 2*optimizer.getParameters().alpha1

# set operator
optimizer.setOperator(op)

# set optimizer data
optimizer.setNoisyData(rawdata)

# solve
optimizer.solve()

# get TGV result
output = optimizer.getResult()

# plot results
plt.figure(2)
imshow(output, 'TGV reconstruction')

plt.show()