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

import numpy as np
import matplotlib.pyplot as plt
import primaldualtoolbox

def imshow(img, title):
    plt.imshow(np.flipud(np.fliplr(np.abs(img))), cmap='gray')
    plt.title(title)
    plt.axis('off')

# load complex-valued input image
noisy = np.load('../data/knee_cartesian_reference.npy')

# crop center of image
nFE = noisy.shape[0]
noisy = noisy[int(nFE * 0.25 + 1):int(nFE * 0.75 + 1), :]  # remove ROOS from reference.

# normalize to 1
noisy /= np.max(np.abs(noisy))

# plot input image
plt.figure(1)
imshow(noisy, 'Noisy')

# setup optimizer
optimizer = primaldualtoolbox.denoising.TvOptimizer_2c()
optimizer.setNoisyData(noisy)
optimizer.getParameters().Lambda = 1e2
optimizer.getParameters().max_iter = 1000

optimizer.solve()

denoised = optimizer.getResult()

# plot denoised image
plt.figure(2)
imshow(denoised, 'TV denoised')

plt.show()
