% This file is part of primal-dual-toolbox.
%
% Copyright (C) 2018 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
% Institute of Computer Graphics and Vision, Graz University of Technology
% https://www.tugraz.at/institute/icg/research/team-pock/
%
% primal-dual-toolbox is free software: you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or any later version.
%
% primal-dual-toolbox is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.

clear all;
close all;
clc;

addpath('../lib')

% load data
load('../data/brain_64spokes.mat');
load('../data/brain_sensitivities.mat');

[nFE,nSpokes,nCh]=size(rawdata);
rawdata = reshape(rawdata,[nFE*nSpokes,nCh]);

% simple density compensation for radial
w = sqrt(abs(k));

% define NUFFT parameters
nufft_params = struct;
nufft_params.osf = 2;
nufft_params.kernel_width = 3;
nufft_params.sector_width = 8;
nufft_params.img_dim = nFE/2;

% apply adjoint operator to get zero filling solution
input0 = gpuMriRadialAdj([real(col(k)), imag(col(k))],...
                         col(w),...
                         sensitivities,...
                         rawdata,...
                         nufft_params);
% show input0
figure(1);
imshow(abs(input0),[]);
title('Zero filling')
 
% TGV parameters (parameters from Knoll et al.)
tgv_params = struct;
tgv_params.reduction = 2^(-8);     % usually there is no need to change this
tgv_params.alpha1 = 1e-5/tgv_params.reduction; % usually there is no need to change this
tgv_params.alpha0 = 2 * tgv_params.alpha1;
tgv_params.max_iter = 1000;            % use 1000 Iterations for optimal image quality

% Solve
output = gpuTgvMriOptimizer_radial([real(col(k)), imag(col(k))],...
                         col(w),...
                         sensitivities,...
                         rawdata,...
                         nufft_params,...
                         tgv_params);

% Show output
figure(2);
imshow(abs(output),[])
title('TGV reconstruction')
