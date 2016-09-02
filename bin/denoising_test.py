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

import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import primaldualtoolbox

def imshow(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

# load input image
img = scipy.misc.face(gray=True).astype(np.float32)/255.0

# add noise
sigma = 0.1
noisy = img + sigma*np.random.randn(*img.shape)

# plot noisy
plt.figure(1)
imshow(noisy, 'Noisy')

# plot reference
plt.figure(2)
imshow(img, 'Reference')

# setup optimizer
optimizer = primaldualtoolbox.denoising.TvOptimizer_2f()
optimizer.setNoisyData(noisy.astype(np.float32))
optimizer.getParameters().Lambda = 10
optimizer.getParameters().max_iter = 1000

optimizer.solve()

denoised = optimizer.getResult()

# plot denoised image
plt.figure(3)
imshow(denoised, 'TV denoised')

plt.show()
