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

% load rawdata and sensitivities
load('../data/knee_cartesian_rawdata.mat');
load('../data/knee_cartesian_sensitivities.mat');

% load reference
load('../data/knee_cartesian_reference.mat');

% load sampling mask
load('../data/mask_cartesian_random4.mat')
% load('../data/mask_cartesian_regular4.mat')

% our data was acquired with readout oversampling (ROOS). We implemented two
% operators, one that keeps the ROOS and one that removes the ROOS.
% We did not observe any differences in image quality.
remove_roos = 1;

% apply adjoint operator to get zero filling solution
if remove_roos
  zero_filling = gpuMriCartesianRemoveROOSAdj(sensitivities, rawdata, mask);
  nFE = size(rawdata,1);
  reference = reference(nFE*0.25+2:nFE*0.75+1,:);
else
  zero_filling = gpuMriCartesianAdj(sensitivities, rawdata, mask);
end

figure(1);
imshow(flipud(fliplr(abs(zero_filling))),[]);
title('Zero filling')

% Show reference
figure(2);
imshow(flipud(fliplr(abs(reference))),[])
title('Reference')

% TGV parameters
tgv_params = struct;
tgv_params.reduction = 2e-8;
tgv_params.alpha1 = 0.05;
tgv_params.alpha0 = 2 * tgv_params.alpha1;
tgv_params.max_iter = 1000;

% Solve
if remove_roos
  output = gpuTgvMriOptimizer_cartesianRemoveROOS(sensitivities,...
    rawdata,...
    mask,...
    tgv_params);
else
  output = gpuTgvMriOptimizer_cartesian(sensitivities,...
                                      rawdata,...
                                      mask,...
                                      tgv_params);
end

% Show output
figure(3);
imshow(flipud(fliplr(abs(output))),[])
title('TGV reconstruction')