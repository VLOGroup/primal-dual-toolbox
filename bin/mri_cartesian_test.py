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
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title):
    plt.imshow(np.flipud(np.fliplr(np.abs(img))), cmap='gray')
    plt.title(title)
    plt.axis('off')

# load rawdata and sensitivities
rawdata = np.load('../data/knee_cartesian_rawdata.npy')
sensitivities = np.load('../data/knee_cartesian_sensitivities.npy')

# load reference
reference = np.load('../data/knee_cartesian_reference.npy')

# load sampling mask
mask = np.load('../data/mask_cartesian_random4.npy')
# mask = np.load('../data/mask_cartesian_regular4.npy')

# our data was acquired with readout oversampling (ROOS). We implemented two operators, one that keeps the ROOS and
# one that removes the ROOS. We did not observe any differences in image quality.
remove_roos  = True

# init operator
if remove_roos:
    op = primaldualtoolbox.mri.MriCartesianRemoveROOSOperator()
    nFE = reference.shape[0]
    reference = reference[int(nFE*0.25+1):int(nFE*0.75+1),:] # remove ROOS from reference.
else:
    op = primaldualtoolbox.mri.MriCartesianOperator()

# set operator constants. This has to be done in the right order.
op.setCoilSens(sensitivities)
op.setMask(mask)

# Compute and display zero filling solution
zero_filling = op.adjoint(rawdata)
plt.figure(1)
imshow(zero_filling, 'Zero filling')

plt.figure(2)
imshow(reference, 'Reference')

# init optimizer
optimizer = primaldualtoolbox.mri.TgvMriOptimizer_2c3c()

# set TGV parameters
optimizer.getParameters().reduction = 2e-8
optimizer.getParameters().max_iter = 1000
optimizer.getParameters().alpha1 = 0.05
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
plt.figure(3)
imshow(output, 'TGV reconstruction')

plt.show()